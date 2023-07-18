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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


jax.config.update('jax_default_matmul_precision', 'float32')
#jax.config.update('jax_default_matmul_precision', 'fastest')


def get_program(program_name, max_seq_len):
    """Returns RASP program and corresponding token vocabulary."""
    if program_name == "length":
        vocab = {"a", "b", "c", "d"}
        program = lib.make_length()
        input_seq = "abbbc"
    elif program_name == "frac_prevs":
        vocab = {"a", "b", "c", "x"}
        program = lib.make_frac_prevs((rasp.tokens == "x").named("is_x"))
        input_seq = "abxxc"
    elif program_name == "dyck-2":
        vocab = {"(", ")", "{", "}"}
        program = lib.make_shuffle_dyck(pairs=["()", "{}"])
        input_seq = "{(})"
    elif program_name == "dyck-3":
        vocab = {"(", ")", "{", "}", "[", "]"}
        program = lib.make_shuffle_dyck(pairs=["()", "{}", "[]"])
        input_seq = "{(}[])"
    elif program_name == "sort":
        vocab = {1, 2, 3, 4, 5}
        program = lib.make_sort(
            rasp.tokens, rasp.tokens, max_seq_len=max_seq_len, min_key=1)
        input_seq = [3, 2, 3, 5, 2]
    elif program_name == "sort_unique":
        vocab = {1, 2, 3, 4, 5}
        program = lib.make_sort_unique(rasp.tokens, rasp.tokens)
        input_seq = [3, 2, 1, 5, 2]
    elif program_name == "hist":
        vocab = {"a", "b", "c", "d"}
        program = lib.make_hist()
        input_seq = "abccd"
    elif program_name == "sort_freq":
        vocab = {"a", "b", "c", "d"}
        program = lib.make_sort_freq(max_seq_len=max_seq_len)
        input_seq = "abcaba"
    elif program_name == "pair_balance":
        vocab = {"(", ")"}
        program = lib.make_pair_balance(
            sop=rasp.tokens, open_token="(", close_token=")")
        input_seq = "(()()"
    elif program_name == "map_test":
        vocab = {1, 2, 3, 4, 5}
        program = rasp.Map(lambda x: x > 4, rasp.tokens)
        input_seq = [1, 2]
    elif program_name == "map_test_b":
        vocab = {1, 2, 3, 4, 5}
        program = rasp.Map(lambda x: x < 1, rasp.Map(
            lambda x: x > 1, rasp.tokens))
        input_seq = [1, 2]
    elif program_name == "map_test_c":
        vocab = {1, 2, 3, 4, 5}
        input_seq = [1, 2]

        def p():
            a = rasp.Map(lambda x: x > 1, rasp.tokens)
            b = rasp.Map(lambda x: x > 2, a)
            c = rasp.Map(lambda x: x >= 3, b)
            d = rasp.Map(lambda x: x < 2, c)
            e = rasp.Map(lambda x: x >= 2, d)
            f = rasp.Map(lambda x: x <= 1, e)
            return f
        program = p()

    else:
        raise NotImplementedError(f"Program {program_name} not implemented.")
    return program, vocab, input_seq


def show_emb(params, show=True):
    emb = np.array(params['compressed_transformer']['w_emb'])
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(emb)
    ax2.imshow(emb.T @ emb)
    if show:
        fig.show()
    return fig

def show_images(arrays, show=True):
    fig, axs = plt.subplots(1,len(arrays))
    for i in range(len(arrays)): axs[i].imshow(arrays[i])
    if show:
        fig.show()
    return fig

def show_image(arrays, show=True):
    fig, axs = plt.subplots(1,1)
    axs.imshow(arrays)
    if show:
        fig.show()
    return fig
    
from PIL import Image
import io
import numpy as np

def figure_to_array(fig):
    picture_stream = io.BytesIO()
    fig.savefig(picture_stream, format='png')
    img = np.array(Image.open(picture_stream))
    return img

# %% =================== init program and compile transformer programs ===========================


prog_name = "sort_unique"#"hist"#"sort"#"length"
program, vocab, input_seq = get_program(prog_name, 6)
vocab = set(list(input_seq))
# formatted_input = [COMPILER_BOS] + list(input_seq)
max_seq_len = len(input_seq)+1

# vocab = {1,2,3,4,5}
# max_seq_len = 5



assembled_model, compressed_assembled_model = compile_with_compressed(
    program, vocab, max_seq_len, compression=1)



# # normal
# encoded_tokens = assembled_model.encode_input(formatted_input)
# output = assembled_model.forward(assembled_model.params, encoded_tokens)
# decoded = assembled_model.decode_output(output)
# print(decoded)

# # compressed
# output = compressed_assembled_model.forward(compressed_assembled_model.params, encoded_tokens)
# decoded = compressed_assembled_model.decode_output(output)
# print(decoded)

from tracr.craft import vectorspace_fns
res_to_out = vectorspace_fns.project(compressed_assembled_model.residual_space, compressed_assembled_model.output_space)

#%%

eg = list(product(*[vocab]*max_seq_len))[0]
emb = assembled_model.forward_emb.apply(assembled_model.params, jnp.array(eg))

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
        rnd = np.random.rand(*emb.shape)# * max(vocab)
        return formatted_input, rnd#np.array(encoded_tokens)
    def collate_fn(data):
            formatted_input = [d[0] for d in data]
            encoded_tokens = [np.array(d[1]) for d in data]
            encoded_tokens = np.stack(encoded_tokens, axis=0).squeeze()
            return formatted_input, np.array(encoded_tokens)
            


vocab_dataloader = DataLoader(VocabDataset(vocab, max_seq_len, assembled_model.encode_input), batch_size=64, collate_fn=VocabDataset.collate_fn, shuffle=True)

class TeacherDataset:
    def __init__(self, vocab_dataloader, teacher) -> None:
        self.vocab_dataloader = vocab_dataloader
        self.iter = iter(vocab_dataloader)
        self.teacher = teacher
        self.teacher_call = jax.jit(teacher.forward_no_emb.apply, device=jax.devices("cpu")[0]) #teacher.forward # jax.jit(teacher.forward)
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
            decoded = self.teacher.decode_output(output)
            logits = jnp.argmax(output.transformer_output.output @ res_to_out.matrix, axis=-1)
        return np.array(formatted_input), np.array(encoded_tokens), np.array(target_outs), decoded, np.array(logits)

dataset = TeacherDataset(vocab_dataloader, assembled_model)
it = iter(dataset)


formatted_input, encoded_tokens, outs, decoded, logits = next(it)
# pred = assembled_model.apply(list(formatted_input[0])).decoded
# assert (pred == decoded)
# print(formatted_input, decoded, pred)


def collate_fn(data):
        assert len(data) == 1
        return data[0]

train_dataloader = DataLoader(dataset, batch_size=1, num_workers=8, collate_fn=collate_fn, prefetch_factor=4)#, num_workers=8, prefetch_factor=18, shuffle=True)

# it = iter(train_dataloader)
# formatted_input, encoded_tokens, outs, decoded, logits = next(it)



#%% ==================== Schedulers ==========================================

LR = 1e-3
EPOCHS = 500



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

cs = CustomSchedule(LR)
def sched(count):
    return cs.LR

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
    #optax.adamw(LR, weight_decay=0.0001)
    #optax.sgd(learning_rate=LR)
    #optax.sgd(make_schedule(LR))
    
    # cosine anealing scheduler
    optax.adamw(create_learning_rate_fn(1, EPOCHS, LR, len(train_dataloader)), weight_decay=0.0001)
    
    # custom scheduler - halves every 20
    #  ensure you uncomment the line in the train loop to use
    #optax.adamw(sched, weight_decay=0.0001)
)


#%% ================= setup frozen grads ==========================================
# helpers for zero grads on all parameters other than compressed_transformer/w_emb
from flax.core.frozen_dict import FrozenDict

def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'adam'
    mask = {}
    _map(params, mask, label_fn)
    return mask

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


optimizer = optax.multi_transform({'adam': optimizer, 'zero': zero_grads()},
                           create_mask(compressed_assembled_model.params, lambda s: s != 'compressed_transformer'))


#%% ============== init train state ===============================

from flax.core.frozen_dict import unfreeze
# Initialize training state
state = train_state.TrainState.create(
    apply_fn=jax.jit(compressed_assembled_model.forward_no_emb.apply), params=unfreeze(compressed_assembled_model.params), tx=optimizer)


def calculate_loss(params, batch):
    encoded_tokens, targets, logits = batch
    output = state.apply_fn(params, encoded_tokens)
    compressed_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()

    # L2 Loss
    loss = jnp.mean((targets - compressed_outs)** 2) 

    # L1 Loss
    # loss = jnp.mean(jnp.abs(targets - compressed_outs))

    # Softmax Loss
    #loss = optax.softmax_cross_entropy(compressed_outs, targets).mean()

    # Additional logit error term
    #loss = optax.softmax_cross_entropy(output.transformer_output.output, logits).mean()#* 100.0
    pred_logits = output.transformer_output.output @ res_to_out.matrix
    #loss += optax.softmax_cross_entropy_with_integer_labels(pred_logits, logits).mean()
    
    return loss

def train_step(state, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step = jax.jit(train_step)


def jit_grads(params, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, grads
jit_grads = jax.jit(jit_grads)

#%% ======================== init embedding to be noisy identiy - skip to use random init ========================

state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)
state.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), state.params['compressed_transformer']['w_emb'].shape) / 10
# show_emb(state.params)
# plt.savefig

#%%


# for key, val in state.params.items():
#     for comp, weight in val.items():
#         if key + comp != 'compressed_transformerw_emb':
#             state.params[key][comp] += 0.00001

#%% ======================= Train loop =====================================

from torch.utils.tensorboard import SummaryWriter
log_dir = os.path.join('.clogs', f'L2 rand input')
#log_dir = os.path.join('.clogs', f'LR{LR} init')
logger = SummaryWriter(log_dir=log_dir)

        
global_idx = 0
for epoch in range(EPOCHS):
    with tqdm(total=len(train_dataloader), unit='batch') as tepoch:
        total_loss = 0.0
        for idx, batch in enumerate(train_dataloader):

            (formatted_input, encoded_tokens, target_outs, decoded, logits) = batch 

            params_before = state.params

            state, loss = train_step(state, (encoded_tokens, target_outs, logits))

            # verify that we're only training the embedding matrix
            changes = dict()
            for key, val in state.params.items():
                for comp, weight in val.items():
                    changes[key + comp] = (state.params[key][comp] - params_before[key][comp]).sum()
                    if key + comp != 'compressed_transformerw_emb':
                        assert changes[key + comp] == 0
                          

            
            
            tepoch.set_postfix({'Batch': idx, 'Train Loss': loss})
            logger.add_scalar('hf_loss', loss.item(), global_step=global_idx)
        
            tepoch.update(1)
            total_loss += loss

            global_idx += 1
            if (global_idx % 20) == 0:
                fig = show_emb(state.params, show=False)
                logger.add_figure('emb', fig, global_step=global_idx)

                output = state.apply_fn(state.params, encoded_tokens)
                pred_decoded = compressed_assembled_model.decode_output(output)
                acc = (np.array(pred_decoded) == np.array(decoded)).mean()
                logger.add_scalar('acc', acc, global_step=global_idx)

                compressed_outs = output.transformer_output.layer_outputs
                fig = show_images([x[0, :,:].T for x in compressed_outs])
                logger.add_figure('outs', fig, global_step=global_idx)

                loss, grads = jit_grads(state.params, (encoded_tokens, target_outs, logits))

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
        if (epoch % 10) == 0:
            fig = show_emb(state.params)
            

show_emb(state.params)

#%%
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']))
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']) * np.array(state.params['compressed_transformer']['w_emb']).T)

#%%



#%%
it = iter(dataset)

#%%

formatted_input, encoded_tokens, outs, targ_decoded, logits = next(it)

output = compressed_assembled_model.forward(state.params, encoded_tokens)
decoded = compressed_assembled_model.decode_output(output)
print(f"targ: {targ_decoded}, pred: {decoded}")
#assert (targ_decoded == decoded).all()


# %%

# Set the embedding to the identity to disable it
state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)

# %%



# state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)
# state.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), state.params['compressed_transformer']['w_emb'].shape) / 10

# Visualise residuals
(formatted_input, encoded_tokens, target_outs, decoded, logits) = next(iter(train_dataloader))
output = state.apply_fn(state.params, encoded_tokens)
compressed_outs = jnp.stack(output.transformer_output.layer_outputs, axis=1).squeeze()


logits = output.transformer_output.output @ res_to_out.matrix

plt.imshow(np.array(target_outs[0,:,:,:].reshape(-1, 25)))
plt.show()
plt.imshow(np.array(compressed_outs[0,:,:,:].reshape(-1, 25)))
plt.show()

# %%
