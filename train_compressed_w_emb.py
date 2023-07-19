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

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


args = Namespace(
    generator = 'Random', # 'Vocabulary'
    compression = 2.0,
    noisy_identity_emb = False, # True,
    LR = 5e-2,
    EPOCHS = 10,
    debug_grads = False,
    matmul_precision = 'float32', # 'bfloat16', # 
    train_all_params = False, # True,
    loss = 'L2', #  'L2', 'L1', 'SoftMax'
    add_logit_softmax = False, # False, True
    batch_size = 64,
)

jax.config.update('jax_default_matmul_precision', args.matmul_precision)

# %% =================== init program and compile transformer programs ===========================


prog_name = "sort_unique"#"hist"#"sort"#"length"
program, vocab, input_seq = get_program(prog_name, 6)
vocab = set(list(input_seq))
# formatted_input = [COMPILER_BOS] + list(input_seq)
max_seq_len = len(input_seq)+1

# vocab = {1,2,3,4,5}
# max_seq_len = 5



assembled_model, compressed_assembled_model = compile_with_compressed(
    program, vocab, max_seq_len, compression=args.compression)



if args.noisy_identity_emb: # init embedding to be noisy identiy?
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

if args.train_all_params:
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
            return formatted_input, np.array(encoded_tokens)
            
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

            


class TeacherDataset:
    def __init__(self, vocab_dataloader, teacher, teacher_call) -> None:
        self.vocab_dataloader = vocab_dataloader
        self.iter = iter(vocab_dataloader)
        self.teacher = teacher
        self.teacher_call = jax.jit(teacher_call, device=jax.devices("cpu")[0]) #teacher.forward # jax.jit(teacher.forward)
        self.params = jax.device_put(self.teacher.params, device=jax.devices("cpu")[0])
    def __len__(self):
        return len(self.vocab_dataloader)
    def __getitem__(self, idx):
        with jax.default_device(jax.devices("cpu")[0]):
            formatted_input, encoded_tokens =  next(self.iter, (None, None))
            if formatted_input is None:
                self.iter = iter(vocab_dataloader)
                formatted_input, encoded_tokens =  next(self.iter, (None, None))
            output = self.teacher_call(self.params, encoded_tokens) # todo improve performance by calling the teacher on a batch
            target_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()
            decoded = np.array(self.teacher.decode_all_outputs(output))
            target_ids = jnp.argmax(self.teacher.residual_to_logits(output), axis=-1)
        return np.array(formatted_input), np.array(encoded_tokens), np.array(target_outs), decoded, np.array(target_ids)

vocab_dataloader = DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=args.batch_size, collate_fn=VocabDataset.collate_fn, shuffle=True)
dataset = None
if args.generator == 'Vocabulary':
    dataset = TeacherDataset(vocab_dataloader, assembled_model, assembled_model.forward)
elif args.generator == 'Random':
    random_dataloader = DataLoader(RandomDataset(max_seq_len, len(assembled_model.residual_labels)), batch_size=args.batch_size, collate_fn=VocabDataset.collate_fn, shuffle=True)
    dataset = TeacherDataset(random_dataloader, assembled_model, assembled_model.forward_no_emb)



next(iter(dataset)) # required otherwise seg fault in dataloader for some reason
train_dataloader = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=lambda x: x[0], prefetch_factor=4)#, num_workers=8, prefetch_factor=18, shuffle=True)


val_dataset = TeacherDataset(vocab_dataloader, assembled_model, assembled_model.forward)
next(iter(val_dataset)) # required otherwise seg fault in dataloader for some reason
validation_dataloader =  DataLoader(val_dataset, collate_fn=lambda x: x[0], num_workers=4, prefetch_factor=2)


#%% ==================== Schedulers ==========================================





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



class CustomSchedule:
    def __init__(self, LR) -> None:
        self.history = []
        self.LR = LR
        self.initial_LR = LR
    def log(self, loss):
        self.history.append(loss)
        if len(self.history) >= 20:
            history = self.history
            # less than 2% change in 2 epochs per epoch
            if (abs(history[-1] - history[-2]) + abs(history[-1] - history[-3])) / history[-1] < 0.04 or \
                abs(history[-1] - history[-3]) / history[-1] < 0.04 and history[-1] > history[-2]:
                self.LR = self.LR / 2
                print(f"updated LR: {self.LR}")
                self.history = []

cs = CustomSchedule(args.LR)
def sched(count):
    return cs.LR

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
    #optax.adamw(args.LR, weight_decay=0.0001)
    #optax.sgd(learning_rate=args.LR)
    #optax.sgd(make_schedule(args.LR))
    
    # cosine anealing scheduler
    optax.adamw(create_learning_rate_fn(1, args.EPOCHS, args.LR, len(train_dataloader)), weight_decay=0.0001)
    
    # custom scheduler - halves every 20
    #  ensure you uncomment the line in the train loop to use
    #optax.adamw(sched, weight_decay=0.0001)
)


#%% ================= setup frozen grads ==========================================

if not args.train_all_params:
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
    if args.add_logit_softmax:
        logits = compressed_assembled_model.residual_to_logits(output)
        loss += optax.softmax_cross_entropy_with_integer_labels(logits, target_ids).mean()
    
    return loss

def train_step(state, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step = jax.jit(train_step)


def jit_grads(state, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads
jit_grads = jax.jit(jit_grads)




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
            if (global_idx % (len(train_dataloader)//4)) == 0:
                fig = show_emb(state.params, show=False)
                logger.add_figure('emb', fig, global_step=global_idx)
                output = state.apply_fn(state.params, encoded_tokens)
                pred_decoded = compressed_assembled_model.decode_output(output)
                # acc = (np.array(pred_decoded) == np.array(decoded)).mean()
                # logger.add_scalar('acc', acc, global_step=global_idx)
                
                compressed_outs = output.transformer_output.layer_outputs
                fig = show_images([x[0, :,:].T for x in compressed_outs], show=False)
                logger.add_figure('outs', fig, global_step=global_idx)
                
                if args.debug_grads:
                    loss, grads = jit_grads(state, (encoded_tokens, target_outs, target_ids))

                    logger.add_figure('d_emb',      show_image(np.array(grads['compressed_transformer']['w_emb'])),                     global_step=global_idx)#, dataformats='HW')
                    logger.add_figure('l0_att_lin', show_image(np.array(grads['compressed_transformer/layer_0/attn/linear']['w'] ))   , global_step=global_idx)#, dataformats='HW')
                    logger.add_figure('l0_mlp2',    show_image(np.array(grads['compressed_transformer/layer_0/mlp/linear_2']['w']))   , global_step=global_idx)#, dataformats='HW')
                    logger.add_figure('l1_mlp2',    show_image(np.array(grads['compressed_transformer/layer_1/mlp/linear_2']['w']))   , global_step=global_idx)#, dataformats='HW')

        
        avg_loss = total_loss / len(train_dataloader)
        tepoch.set_postfix({'Batch': idx, 'Avg Loss': avg_loss})
        logger.add_scalar('avg loss', avg_loss.item(), global_step=epoch)
        
        # enable if using custom scheduler
        # cs.log(avg_loss)
        
        logger.add_scalar('LR', cs.LR, global_step=epoch)
        
    with tqdm(total=10, unit='batch') as tepoch:
        it = iter(validation_dataloader)
        avg_acc = 0.0
        for idx in range(10):        
            batch = next(it)
            (formatted_input, encoded_tokens, target_outs, decoded, target_ids) = batch
            output = jax.jit(compressed_assembled_model.forward)(state.params, encoded_tokens)
            pred_decoded = compressed_assembled_model.decode_all_outputs(output)
            acc = np.equal(pred_decoded , decoded).mean()
            avg_acc += acc
            logger.add_scalar('acc', acc, global_step=global_idx)
            tepoch.set_postfix({'Batch': idx, 'Acc': acc})
            tepoch.update(1)
        avg_acc /= 10
        tepoch.set_postfix({'Avg Acc': avg_acc})
        logger.add_scalar('acc', avg_acc, global_step=global_idx)

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
print(f"targ: {targ_decoded}, pred: {decoded}")
#assert (targ_decoded == decoded).all()


# %%

# Set the embedding to the identity to disable it
# state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)
