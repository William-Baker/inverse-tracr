# %%
import sys
sys.path.append('tracr/')
import jax.numpy as jnp
from flax.training import train_state
import optax
from utils.compile_with_compressed import compile_with_compressed, COMPILER_BOS
from utils.plot import *
import jax
from tqdm import tqdm
from itertools import product
import os
from argparse import Namespace
import torch
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
from datetime import datetime
from utils.time_sensitive import time_sensitive
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

process_args = Namespace(
    run_id =   str(datetime.now().strftime("%m-%d %H.%M.%S.%f"))
)

#%%
args = Namespace(
    generator = 'Random', # 'Vocabulary'
    compression = 1.0,
    idty = True, # True, # Whether to use a noisy identity to initialise the embedding
    LR = 1e-4, # 5e-2 worked so far but some nans
    EPOCHS = 20,
    trn_all = False, # True,
    loss = 'L2', #'L2', #  'L2', 'L1', 'SoftMax'
    add_soft = False, #True, # True, # True, # True, # False, True
    batch_size = 64,
    mult = False, #True, #True,
    sched = 'cosine',
    #mpow = 2,
    factor=0.01,
)

from torch.utils.tensorboard import SummaryWriter
log_dir = os.path.join("Compressed Tracr All" if args.trn_all == True else "Compressed Tracr emb_W", 
                        process_args.run_id)
logger = SummaryWriter(log_dir=log_dir)

logger.add_hparams(vars(process_args) | vars(args), dict() )

jax.config.update('jax_default_matmul_precision', 'float32') # 'bfloat16'


print("a")

# %% =================== init program and compile transformer programs ===========================
program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_op, ops_range = [None]*7

from data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
ops_range=(10, 15)
numeric_range=(5, 8)
vocab_size_range=(5, 8)
numeric_inputs_possible=True
max_seq_len = np.random.randint(4, 9)
CRAFT_TIMEOUT = 2# 0.2 + 0.00001 * max(ops_range) ** 4

def timed_func():
    assembled_model, compressed_assembled_model, actual_ops = None, None, None
    
    n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
    print(n_ops, vocab, TARGET_PROGRAM_LENGTH)
    try:
        program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
    except np.core._exceptions._ArrayMemoryError as E:
        print("mem alloc err")
        return None
    try:
        assembled_model, compressed_assembled_model, craft_model, rasp_model = compile_with_compressed(
            program, vocab, max_seq_len, compression=args.compression,
            CRAFT_TIMEOUT=CRAFT_TIMEOUT)
    except ValueError as E:
        print("val err")
        return None
    except KeyError as E:
        print("key err")
        return None
    except TimeoutError:
        print("craft timeout")
        return None

    return assembled_model, compressed_assembled_model, actual_ops, vocab, program



#%%


ret = None
for i in range(20):
    ret = time_sensitive(timed_func, 10)
    if ret is not None:
        break
if ret is None:
    exit(1)
    logger.add_scalar("progress",  0, 4)
    logger.add_scalar("fail", 1, 1)
    sys.exit(1)

assembled_model, compressed_assembled_model, actual_ops, vocab, program = ret

print("b")

    #craft_model, actual_ops = program_craft_generator(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_range=numeric_range, numeric_inputs_possible=numeric_inputs_possible)

logger.add_scalar("prog len", len(actual_ops), 1)
logger.add_scalar("progress", 0.1, 1)


if args.idty: # init embedding to be noisy identiy?
    compressed_assembled_model.params['compressed_transformer']['w_emb'] = jnp.eye(*compressed_assembled_model.params['compressed_transformer']['w_emb'].shape)
    compressed_assembled_model.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), compressed_assembled_model.params['compressed_transformer']['w_emb'].shape) / 10


def init_all_params(params):
    rng = jax.random.PRNGKey(0)
    initializer = jax.nn.initializers.glorot_uniform()
    for key, val in params.items():
        for comp, weight in val.items():
            if 'compressed_transformer' in key + comp:
                rng, nrng = jax.random.split(rng, 2)
                if len(params[key][comp].shape) > 1:
                    params[key][comp] =initializer(nrng, params[key][comp].shape, jnp.float32)
                else:
                    params[key][comp] = jax.random.normal(nrng, params[key][comp].shape) / 1000
    return params

if args.trn_all:
    compressed_assembled_model.params = init_all_params(compressed_assembled_model.params)

#%%

for key, val in compressed_assembled_model.params.items():
        for comp, weight in val.items():
            if 'compressed_transformer' in key + comp:
                if comp != 'w_emb':
                    print(key + ' ' + comp)
                    assert (weight == assembled_model.params[key.replace('compressed_transformer', 'transformer')][comp]).all()


# # normal
# encoded_tokens = assembled_model.encode_input(formatted_input)
# output = assembled_model.forward(assembled_model.params, encoded_tokens)
# decoded = assembled_model.decode_output(output)
# print(decoded)

# # compressed
# output = compressed_assembled_model.forward(compressed_assembled_model.params, encoded_tokens)
# decoded = compressed_assembled_model.decode_output(output)
# print(decoded)

print("c")
#%% ======================== Dataloader ======================================


class VocabDataset:
    def __init__(self, vocab, max_seq_len, encoder_fn) -> None:
        self.vocab = vocab
        self.inputs = list(product(*[vocab]*max_seq_len))
        self.encoder_fn = encoder_fn
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        formatted_input = [COMPILER_BOS] + list(self.inputs[idx])
        encoded_tokens =  self.encoder_fn(formatted_input)
        return formatted_input, np.array(encoded_tokens)
    def collate_fn(data):
            formatted_input = [d[0] for d in data]
            encoded_tokens = [d[1] for d in data]
            encoded_tokens = np.stack(encoded_tokens, axis=0).squeeze()
            return formatted_input, encoded_tokens
            
class RandomDataset:
    def __init__(self, max_seq_len, residual_size, length=25000) -> None:
        self.max_seq_len = max_seq_len
        self.residual_size = residual_size
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        rnd = np.random.rand(self.max_seq_len, self.residual_size)
        return None, rnd
    def collate_fn(data):
        formatted_input = [d[0] for d in data]
        encoded_tokens = [d[1] for d in data]
        encoded_tokens = np.stack(encoded_tokens, axis=0)
        return formatted_input, np.array(encoded_tokens)

            



print("d")

def make_teacher_call(teacher, teacher_forward):
    def fun(encoded_tokens):
        output = teacher_forward(teacher.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
        target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
        # decoded = np.array(teacher.decode_all_outputs(output))
        target_ids = jnp.argmax(teacher.residual_to_logits(output), axis=-1)
        return target_outs, target_ids
    return fun

def make_validation_teacher_call(teacher, teacher_forward):
    def fun(encoded_tokens):
        output = teacher_forward(teacher.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
        target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
        decoded = np.array(teacher.decode_all_outputs(output))
        target_ids = jnp.argmax(teacher.residual_to_logits(output), axis=-1)
        return target_outs, target_ids, decoded
    return fun

train_teacher_call = None

dataset = None
if args.generator == 'Vocabulary':
    
    train_dataloader = DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=args.batch_size, collate_fn=VocabDataset.collate_fn, shuffle=True, num_workers=1, prefetch_factor=2)
    train_teacher_call = make_teacher_call(assembled_model, assembled_model.forward)
elif args.generator == 'Random':
    train_dataloader = DataLoader(RandomDataset(max_seq_len, len(assembled_model.residual_labels)), batch_size=args.batch_size, collate_fn=RandomDataset.collate_fn, num_workers=1, prefetch_factor=2)
    train_teacher_call = make_teacher_call(assembled_model, assembled_model.forward_no_emb)

validation_dataloader = train_dataloader
if args.generator == 'Random':
    validation_dataloader =  DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=32, collate_fn=VocabDataset.collate_fn, shuffle=True, num_workers=1, prefetch_factor=2)
validation_teacher_call = make_validation_teacher_call(assembled_model, assembled_model.forward)

logger.add_scalar("progress", 0.2, 2)

#%% ==================== Schedulers ==========================================


class CustomSchedule:
    def __init__(self, LR) -> None:
        self.history = []
        self.LR = LR
        self.initial_LR = LR
    def log(self, loss):
        self.history.append(loss)
        if len(self.history) > 600:
            history = self.history
            avg_loss = np.mean(self.history[-100:])
            # less than 2% change in 2 epochs per epoch
            # if avg_loss < 1e-4:
            #     self.LR = self.initial_LR / 600*5
            # elif avg_loss < 5e-4:
            #     self.LR = self.initial_LR / 600
            # elif avg_loss < 1e-3:
            #     self.LR = self.initial_LR / 125
            # elif avg_loss < 5e-2:
            #     self.LR = self.initial_LR / 25
            # elif avg_loss < 1e-1:
            #     self.LR = self.initial_LR / 5
            # else:
            #     self.LR = self.initial_LR
            self.LR = self.initial_LR * min(0.01 * (np.exp(30 * avg_loss) - 1), 1)


cs = CustomSchedule(args.LR)




# cosine anealing + warmup scheduler
def create_learning_rate_fn(warmup_epochs, num_epochs, base_learning_rate, steps_per_epoch):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
  cosine_epochs = max(num_epochs - warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_epochs * steps_per_epoch])
  return schedule_fn




LR_fn = None
if args.sched == 'cosine': # cosine anealing scheduler
    LR_fn = create_learning_rate_fn(1, args.EPOCHS, args.LR, len(train_dataloader))    
elif args.sched == 'custom': # custom scheduler
    #  ensure you uncomment the line in the train loop to use
    LR_fn = lambda x: cs.LR

optimizer = optax.chain(
    # optax.clip_by_global_norm(0.01),  # Clip gradients at norm 1
    # optax.clip(1), # prevent nan ?
    #optax.adamw(args.LR, weight_decay=0.0001)
    #optax.sgd(learning_rate=args.LR)
    #optax.sgd(LR_fn)
    optax.adamw(LR_fn, weight_decay=0.0001) ,
    
)


#%% ================= setup frozen grads ==========================================

if not args.trn_all:
    # helpers for zero grads on all parameters other than compressed_transformer/w_emb
    from flax.core.frozen_dict import unfreeze
    from utils.jax_helpers import zero_grads, create_mask

    optimizer = optax.multi_transform({'adam': optimizer, 'zero': zero_grads()},
                            create_mask(compressed_assembled_model.params, lambda s: s != 'compressed_transformer'))
    compressed_assembled_model.params = unfreeze(compressed_assembled_model.params)


#%%

# jax.tree_map(lambda x: jnp.isnan(x).any(), compressed_assembled_model.params)
# jax.tree_map(lambda x: jnp.isnan(x).any(), assembled_model.params)

#%% ============== init train state ===============================


forward_fn = None
if args.generator == 'Vocabulary':
    forward_fn = compressed_assembled_model.forward
elif args.generator == 'Random':
    forward_fn = compressed_assembled_model.forward_no_emb

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=forward_fn, params=compressed_assembled_model.params, tx=optimizer)



def calculate_loss(params, batch):
    encoded_tokens, target_outs, target_ids = batch
    output = state.apply_fn(params, encoded_tokens)
    compressed_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()

    loss = 0.0
    # L2 Loss
    if args.loss == 'L2':
        loss = jnp.mean((target_outs - compressed_outs)** 2) 

    # L1 Loss
    elif args.loss == 'L1':
        loss = jnp.mean(jnp.abs(target_outs - compressed_outs))
    
    elif args.loss == 'SoftMax':
        loss = optax.softmax_cross_entropy(compressed_outs, target_outs).mean()

    # Additional logit error term
    if args.add_soft:
        logits = compressed_assembled_model.residual_to_logits(output)
        if args.mult:
            loss *= optax.softmax_cross_entropy_with_integer_labels(logits, target_ids).mean() ** args.mpow
        else:
            loss += optax.softmax_cross_entropy_with_integer_labels(logits, target_ids).mean() * args.factor
    
    return loss

def train_step(state, encoded_tokens):
    target_outs, target_ids =  train_teacher_call(encoded_tokens) 
    batch = (encoded_tokens, target_outs, target_ids)
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step = jax.jit(train_step)




#%% ======================= Train loop =====================================

logger.add_scalar("progress", 0.3, 3)



avg_loss = 0.0
        
global_idx = 0
for epoch in range(args.EPOCHS):
    with tqdm(total=len(train_dataloader), unit='batch') as tepoch:
        total_loss = 0.0
        for idx, batch in enumerate(train_dataloader):

            (formatted_input, encoded_tokens) = batch 


            state, loss = train_step(state, encoded_tokens)
            
            
            tepoch.set_postfix({'Batch': idx, 'Train Loss': loss})

            tepoch.update(1)
            total_loss += loss
            global_idx += 1
            # wandb.log({"loss": loss.item()}) # if more than 10 processes, exceeds rate limits
            # if (global_idx % 50) == 0:
            logger.add_scalar("loss", loss.item(), global_idx)

            if np.isnan( loss.item() ):
                logger.add_scalar("progress",  0, 4)
                logger.add_scalar("fail", 2, 2)
                sys.exit(1)

        
        avg_loss = total_loss / len(train_dataloader)
        tepoch.set_postfix({'Batch': idx, 'Avg Loss': avg_loss})
        logger.add_scalar("avg loss", avg_loss.item(), epoch)

 

 

logger.add_scalar("progress", 0.4, 4)

if avg_loss > 0.05:
    logger.add_scalar("progress",  0, 4)
    logger.add_scalar("fail", 3, 3)
    sys.exit(1)
    
VAL_SAMPLES = 10
with tqdm(total=VAL_SAMPLES, unit='batch') as tepoch:
    it = iter(validation_dataloader)
    avg_acc = 0.0
    for idx in range(VAL_SAMPLES):        
        batch = next(it)
        (formatted_input, encoded_tokens) = batch
        target_outs, target_ids, decoded = jax.jit(validation_teacher_call)(encoded_tokens)
        output = jax.jit(compressed_assembled_model.forward)(state.params, encoded_tokens)
        pred_decoded = compressed_assembled_model.decode_all_outputs(output)
        acc = np.equal(pred_decoded , decoded).mean()
        avg_acc += acc
        tepoch.set_postfix({'Batch': idx, 'Acc': acc})
        tepoch.update(1)
    avg_acc /= VAL_SAMPLES
    tepoch.set_postfix({'Avg Acc': avg_acc})
    logger.add_scalar("acc", avg_acc, 1)

# show_emb(state.params)


logger.add_scalar("progress", 0.5, 5)


# %%



def compress_params(params):
    # we first need to find the compression matrix
    w = params['compressed_transformer']['w_emb'].T
    compressed_params = dict()
    for key in params.keys():
        if 'compressed_transformer/' in key:
            p = params[key]['w']
            key = key.replace( 'compressed_transformer/', '')
            print(key)
            print(p.shape)
            if not (key.endswith('linear') or key.endswith('linear_2')):
                compressed_params[key] = np.array((p.T @ w).T)
            else:
                compressed_params[key] = np.array((p @ w))
            print(compressed_params[key].shape)
    return compressed_params

compressed = compress_params(state.params)

logger.add_scalar("progress", 0.6, 6)

# %%

from data.dataloaders import ProgramEncoder
from collections import defaultdict

prog_enc = ProgramEncoder(max(ops_range))
encoded_ops = ProgramEncoder.encode_ops(actual_ops)
tokenised_program = prog_enc.tokenise_program(encoded_ops)


def encode_jax_params(params):
    collected_by_block = defaultdict(lambda: dict())
    for key, val in params.items():
        layer_no, layer_type, param_name = key.split('/')
        collected_by_block[layer_no + layer_type][param_name] = val
    
    model_params = []
    for key, val in collected_by_block.items():
        if 'attn' in layer_type:
            model_params.append({'MHA': val})
        elif 'mlp' in layer_type:
            model_params.append({'MLP': val})
        else:
            raise NotImplementedError()
    return model_params

encoded_params = encode_jax_params(compressed)

logger.add_scalar("progress", 0.7, 7)

#%%

from zipfile import ZipFile
from cloudpickle import dumps
sample = (encoded_params, tokenised_program)

target_db_path = 'cp_dataset'
if args.trn_all == True:
    target_db_path += '_train_all'
else:
    target_db_path += '_train_w'

for i in range(10):
    try:
        zip = ZipFile(file=target_db_path+'.zip', mode='a')
        zip.writestr( process_args.run_id + '.pkl', dumps(sample))
        zip.close()
        break
    except:
        pass


logger.add_scalar("progress", 1, 10)


#%%


from data.parallelzipfilebetter import ParallelZipFile as ZipFile
import cloudpickle

class ZipStreamReader:
    def __init__(self, dir:str) -> None:
        self.zip = ZipFile(file=dir, mode='r')
        self.files = sorted(self.zip.namelist())
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        x = self.zip.read(self.files[idx])
        # loaded = np.load(BytesIO(x), allow_pickle=True)
        x,y = cloudpickle.loads(x)
        return x, y

#df = ZipStreamReader('cp_dataset_train_all.zip')

df = ZipStreamReader('cp_dataset_train_w.zip')
it = iter(df)
x,y = next(it)

print(len(df))

from data.dataloaders import ProgramEncoder


prog_enc = ProgramEncoder(15)
print(prog_enc.decode_pred(y))


