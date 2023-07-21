# %%
import sys
sys.path.append('tracr/')
import jax.numpy as jnp
from flax.training import train_state
import optax
from utils.compile_with_compressed import compile_with_compressed, COMPILER_BOS
import tracr.compiler.lib as lib
from tracr.rasp import rasp
from utils.plot import *
import jax
from tqdm import tqdm
from itertools import product
import os
from data.example_rasp_programs import get_program
from utils.plot import show_emb, show_images, show_image, figure_to_array
from argparse import Namespace
from utils.time_sensitive import time_sensitive
jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"



args = Namespace(
    generator = 'Random', # 'Vocabulary'
    program = 'random', #'sort_unique', "hist"#"sort"#"length"
    compression = 2.0,
    idty = False, # True, # Whether to use a noisy identity to initialise the embedding
    LR = 5e-2,
    EPOCHS = 20,
    trn_all = True, # True,
    loss = 'L2', #'L2', #  'L2', 'L1', 'SoftMax'
    add_soft = True, # True, # True, # True, # False, True
    batch_size = 64,
    mult = False, #True, #True,
    sched = 'cosine',
    #mpow = 1,
    factor=0.001,
)


jax.config.update('jax_default_matmul_precision', 'float32') # 'bfloat16'

# %% =================== init program and compile transformer programs ===========================
program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_op = [None]*6
if args.program != 'random':
    program, vocab, input_seq = get_program(args.program, 6)
    vocab = set(list(input_seq))
    max_seq_len = len(input_seq)+1

    assembled_model, compressed_assembled_model = compile_with_compressed(
                        program, vocab, max_seq_len, compression=args.compression)

else:
    from data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
    ops_range=(10, 15)
    numeric_range=(5, 8)
    vocab_size_range=(5, 8)
    numeric_inputs_possible=True
    n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
    program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
    max_seq_len = np.random.randint(5, 10)

    def timed_func():
        print("running func")
        assembled_model, compressed_assembled_model, actual_ops = None, None, None
        
        n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
        
        try:
            program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
        except np.core._exceptions._ArrayMemoryError as E:
            #print("mem alloc err")
            return None

        try:
            assembled_model, compressed_assembled_model = compile_with_compressed(
                program, vocab, max_seq_len, compression=args.compression)
        except ValueError as E:
            #print("val err")
            return None
        except KeyError as E:
            #print("key err")
            return None

        return assembled_model, compressed_assembled_model, actual_ops
    jax.profiler.start_trace("jax-profile")
    for i in range(3):
        ret = None
        while ret is None:
            #ret = time_sensitive(timed_func, 10)
            ret = timed_func()
        assembled_model, compressed_assembled_model, actual_ops = ret
        print(len(actual_ops))
    jax.profiler.stop_trace()

    #craft_model, actual_ops = program_craft_generator(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_range=numeric_range, numeric_inputs_possible=numeric_inputs_possible)


1 / 0




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

# # normal
# encoded_tokens = assembled_model.encode_input(formatted_input)
# output = assembled_model.forward(assembled_model.params, encoded_tokens)
# decoded = assembled_model.decode_output(output)
# print(decoded)

# # compressed
# output = compressed_assembled_model.forward(compressed_assembled_model.params, encoded_tokens)
# decoded = compressed_assembled_model.decode_output(output)
# print(decoded)


#%% ======================== Dataloader ======================================

import torch
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader

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

            


class TeacherDataset:
    def __init__(self, dataloader, teacher, teacher_call) -> None:
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.teacher = teacher
        self.teacher_call = jax.jit(teacher_call, device=jax.devices("cpu")[0]) #teacher.forward # jax.jit(teacher.forward)
        self.params = jax.device_put(self.teacher.params, device=jax.devices("cpu")[0])
    def __len__(self):
        return len(self.dataloader)-2
    def __getitem__(self, idx):
        with jax.default_device(jax.devices("cpu")[0]):
            formatted_input, encoded_tokens =  next(self.iter, (None, None))
            if formatted_input is None:
                self.iter = iter(self.dataloader)
                formatted_input, encoded_tokens =  next(self.iter)
            output = self.teacher_call(self.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
            target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
            decoded = np.array(self.teacher.decode_all_outputs(output))
            target_ids = jnp.argmax(self.teacher.residual_to_logits(output), axis=-1)
        return np.array(formatted_input), np.array(encoded_tokens), np.array(target_outs), decoded, np.array(target_ids)


dataset = None
if args.generator == 'Vocabulary':
    x_vocab_dataloader = DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=args.batch_size, collate_fn=VocabDataset.collate_fn, shuffle=True)
    dataset = TeacherDataset(x_vocab_dataloader, assembled_model, assembled_model.forward)
elif args.generator == 'Random':
    random_dataloader = DataLoader(RandomDataset(max_seq_len, len(assembled_model.residual_labels)), batch_size=args.batch_size, collate_fn=RandomDataset.collate_fn)
    dataset = TeacherDataset(random_dataloader, assembled_model, assembled_model.forward_no_emb)

next(iter(dataset)) # required otherwise seg fault in dataloader for some reason
train_dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=4, prefetch_factor=4)#, num_workers=8, prefetch_factor=18, shuffle=True)




val_dataset = TeacherDataset(DataLoader(VocabDataset(
                    vocab, max_seq_len, assembled_model.encode_input), batch_size=64, collate_fn=VocabDataset.collate_fn, shuffle=True), 
                assembled_model, assembled_model.forward)
next(iter(val_dataset)) # required otherwise seg fault in dataloader for some reason
validation_dataloader =  DataLoader(val_dataset, collate_fn=lambda x: x[0], num_workers=1, prefetch_factor=2)



next(iter(train_dataloader))
next(iter(val_dataset))


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



# class CustomSchedule:
#     def __init__(self, LR) -> None:
#         self.history = []
#         self.LR = LR
#         self.initial_LR = LR
#     def log(self, loss):
#         self.history.append(loss)
#         if len(self.history) >= 20:
#             history = self.history
#             # less than 2% change in 2 epochs per epoch
#             if (abs(history[-1] - history[-2]) + abs(history[-1] - history[-3])) / history[-1] < 0.04 or \
#                 abs(history[-1] - history[-3]) / history[-1] < 0.04 and history[-1] > history[-2]:
#                 self.LR = self.LR / 2
#                 print(f"updated LR: {self.LR}")
#                 self.history = []


LR_fn = None
if args.sched == 'cosine': # cosine anealing scheduler
    LR_fn = create_learning_rate_fn(1, args.EPOCHS, args.LR, len(train_dataloader))    
elif args.sched == 'custom': # custom scheduler
    #  ensure you uncomment the line in the train loop to use
    LR_fn = lambda x: cs.LR

optimizer = optax.chain(
    optax.clip_by_global_norm(0.01),  # Clip gradients at norm 1
    #optax.clip(1e-3),
    #optax.adamw(args.LR, weight_decay=0.0001)
    #optax.sgd(learning_rate=args.LR)
    #optax.sgd(make_schedule(args.LR))
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




#%% ============== init train state ===============================


forward_fn = None
if args.generator == 'Vocabulary':
    forward_fn = compressed_assembled_model.forward
elif args.generator == 'Random':
    forward_fn = compressed_assembled_model.forward_no_emb

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=jax.jit(forward_fn), params=compressed_assembled_model.params, tx=optimizer)



def calculate_loss(params, batch):
    encoded_tokens, targets, target_ids = batch
    output = state.apply_fn(params, encoded_tokens)
    compressed_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()

    loss = 0.0
    # L2 Loss
    if args.loss == 'L2':
        loss = jnp.mean((targets - compressed_outs)** 2) 

    # L1 Loss
    elif args.loss == 'L1':
        loss = jnp.mean(jnp.abs(targets - compressed_outs))
    
    elif args.loss == 'SoftMax':
        loss = optax.softmax_cross_entropy(compressed_outs, targets).mean()

    # Additional logit error term
    if args.add_soft:
        logits = compressed_assembled_model.residual_to_logits(output)
        if args.mult:
            loss *= optax.softmax_cross_entropy_with_integer_labels(logits, target_ids).mean() ** args.mpow
        else:
            loss += optax.softmax_cross_entropy_with_integer_labels(logits, target_ids).mean() * args.factor
    
    return loss

def train_step(state, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step = jax.jit(train_step)




#%% ======================= Train loop =====================================

from torch.utils.tensorboard import SummaryWriter
log_dir = os.path.join('.clogs', str(vars(args)).replace('\'', '')[1:-1])
#log_dir = os.path.join('.clogs', f'LR{args.LR} init')
logger = SummaryWriter(log_dir=log_dir)

        
global_idx = 0
for epoch in range(args.EPOCHS):
    with tqdm(total=len(train_dataloader), unit='batch') as tepoch:
        total_loss = 0.0
        for idx, batch in enumerate(train_dataloader):

            (formatted_input, encoded_tokens, target_outs, decoded, target_ids) = batch 


            state, loss = train_step(state, (encoded_tokens, target_outs, target_ids))
            
            
            tepoch.set_postfix({'Batch': idx, 'Train Loss': loss})
            logger.add_scalar('hf_loss', loss.item(), global_step=global_idx)
        
            tepoch.update(1)
            total_loss += loss
            global_idx += 1

            # enable if using custom scheduler
            #cs.log(loss)
            logger.add_scalar('LR', np.array(LR_fn(state.step)).item(), global_step=global_idx)
            

        
        avg_loss = total_loss / len(train_dataloader)
        tepoch.set_postfix({'Batch': idx, 'Avg Loss': avg_loss})
        logger.add_scalar('avg loss', avg_loss.item(), global_step=epoch)
        
        # ======================= Debug Info =====================================
        fig = show_emb(state.params, show=False)
        logger.add_figure('emb', fig, global_step=global_idx)
        output = state.apply_fn(state.params, encoded_tokens)
        compressed_outs = output.transformer_output.layer_outputs
        fig = show_images([x[0, :,:].T for x in compressed_outs], show=False)
        logger.add_figure('outs', fig, global_step=global_idx)

        
        
    
    VAL_SAMPLES = 10
    with tqdm(total=VAL_SAMPLES, unit='batch') as tepoch:
        it = iter(validation_dataloader)
        avg_acc = 0.0
        for idx in range(VAL_SAMPLES):        
            batch = next(it)
            (formatted_input, encoded_tokens, target_outs, decoded, target_ids) = batch
            output = jax.jit(compressed_assembled_model.forward)(state.params, encoded_tokens)
            pred_decoded = compressed_assembled_model.decode_all_outputs(output)
            acc = np.equal(pred_decoded , decoded).mean()
            avg_acc += acc
            logger.add_scalar('acc', acc, global_step=epoch * VAL_SAMPLES + idx)
            tepoch.set_postfix({'Batch': idx, 'Acc': acc})
            tepoch.update(1)
        avg_acc /= VAL_SAMPLES
        tepoch.set_postfix({'Avg Acc': avg_acc})
        logger.add_scalar('avg acc', avg_acc, global_step=epoch)

show_emb(state.params)

#%%
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']))
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']).T @ np.array(state.params['compressed_transformer']['w_emb']))

#%%



#%%
it = iter(dataset)

formatted_input, encoded_tokens, outs, targ_decoded, target_ids = next(it)

output = state.apply_fn(state.params, encoded_tokens)
decoded = compressed_assembled_model.decode_output(output)
#print(f"targ: {targ_decoded}, pred: {decoded}")
#assert (targ_decoded == decoded).all()


# %%

# Set the embedding to the identity to disable it
# state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)
