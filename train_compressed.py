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


# %%


prog_name = "sort"#"length"
program, vocab, input_seq = get_program(prog_name, 6)
vocab = set(list(input_seq))
# formatted_input = [COMPILER_BOS] + list(input_seq)
max_seq_len = len(input_seq)+1

# vocab = {1,2,3,4,5}
# max_seq_len = 5



assembled_model, compressed_assembled_model = compile_with_compressed(
    program, vocab, max_seq_len, compression=1.0)


# # print(f"Runnning {prog_name} with input {input_seq}")
# # pred = assembled_model.apply(formatted_input)
# # prog_out = pred.decoded
# # print(f"Program outputs: {prog_out}")


# # print(f"Runnning {prog_name} with input {input_seq}")
# # compressed_pred = compressed_assembled_model.apply(formatted_input)
# # prog_out = compressed_pred.decoded
# # print(f"Program outputs: {prog_out}")

# print(jax.tree_map(lambda x: jnp.sum(x).item(), assembled_model.params))
# print(jax.tree_map(lambda x: jnp.sum(x).item(), compressed_assembled_model.params))

# # #%%



# # %%

# encoded_tokens = assembled_model.encode_input(formatted_input)
# output = assembled_model.forward(assembled_model.params, encoded_tokens)
# decoded = assembled_model.decode_output(output)
# decoded


# #%%

# output = compressed_assembled_model.forward(compressed_assembled_model.params, encoded_tokens)
# decoded = compressed_assembled_model.decode_output(output)
# decoded


#%%

class TeacherDataset:
    def __init__(self, vocab, teacher, max_seq_len) -> None:
        self.vocab = vocab
        self.inputs = list(product(*[vocab]*max_seq_len))
        self.teacher = teacher
        self.teacher_call = jax.jit(teacher.forward)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        formatted_input = [COMPILER_BOS] + list(self.inputs[idx])
        encoded_tokens = self.teacher.encode_input(formatted_input)
        output = self.teacher_call(self.teacher.params, encoded_tokens)
        target_outs = jnp.stack(output.transformer_output.layer_outputs)
        decoded = self.teacher.decode_output(output)
        logits = output.transformer_output.output
        return formatted_input, np.array(encoded_tokens), np.array(target_outs), decoded, np.array(logits)

dataset = TeacherDataset(vocab, assembled_model, max_seq_len)
it = iter(dataset)
#%%

formatted_input, encoded_tokens, outs, decoded, logits = next(it)
pred = assembled_model.apply(formatted_input).decoded
assert (pred == decoded)
print(formatted_input, decoded, pred)
# %%
import torch
torch.cuda.is_available = lambda : False
from torch.utils.data import DataLoader
def make_collate_fn():
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(data):
            formatted_input = [d[0] for d in data]
            encoded_tokens = [np.array(d[1]) for d in data]
            target_outs = [np.array(d[2]) for d in data]
            decoded = [d[3] for d in data]
            logits = [d[4] for d in data]
            # inputs = pad_sequence(inputs, batch_first=True)
            # attention_masks = pad_sequence(attention_masks, batch_first=True)
            

            encoded_tokens = np.stack(encoded_tokens, axis=0).squeeze()
            target_outs = np.stack(target_outs)#.squeeze()
            logits = np.stack(logits)
            

            return formatted_input, np.array(encoded_tokens), np.array(target_outs), decoded, np.array(logits)
    return collate_fn

train_dataloader = DataLoader(dataset, batch_size=64, collate_fn=make_collate_fn())#, num_workers=8, prefetch_factor=18, shuffle=True)


#%%

it = iter(train_dataloader)
formatted_input, encoded_tokens, outs, decoded, logits = next(it)



#%%

LR = 1e-2


optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
    #optax.adam(learning_rate=LR) #lr_schedule)
    optax.sgd(learning_rate=LR) #lr_schedule)
    # optax.adamw(learning_rate=config.lr, weight_decay=config.weight_decay)
)

# Initialize training state
state = train_state.TrainState.create(
    apply_fn=jax.jit(compressed_assembled_model.forward), params=compressed_assembled_model.params, tx=optimizer)


def calculate_loss(params, batch):
    encoded_tokens, targets, logits = batch
    output = state.apply_fn(params, encoded_tokens)
    compressed_outs = jnp.stack(output.transformer_output.layer_outputs)
    loss = jnp.mean((targets - compressed_outs) ** 2)
    #loss += optax.softmax_cross_entropy(output.transformer_output.output, logits).mean()#* 100.0
    return loss

def train_step(state, batch):
    loss_fn = lambda params: calculate_loss(params, batch)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    w_emb = grads['compressed_transformer']['w_emb']
    grads = jax.tree_map(lambda x: jnp.zeros(x.shape), grads)
    grads['compressed_transformer']['w_emb'] = w_emb
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step = jax.jit(train_step)



#%%

state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)
state.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), state.params['compressed_transformer']['w_emb'].shape) / 10
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']))
plt.show()
#%%
for epoch in range(300):
    with tqdm(total=len(train_dataloader), unit='batch') as tepoch:
        total_loss = 0.0
        for idx, batch in enumerate(train_dataloader):

            (formatted_input, encoded_tokens, target_outs, decoded, v) = batch 

            params_before = state.params

            state, loss = train_step(state, (encoded_tokens, target_outs, logits))

            changes = dict()
            for key, val in state.params.items():
                for comp, weight in val.items():
                    changes[key + comp] = (state.params[key][comp] - params_before[key][comp]).sum()
                    if key + comp != 'compressed_transformerw_emb':
                        assert changes[key + comp] == 0
                            

            output = state.apply_fn(state.params, encoded_tokens)
            decoded = compressed_assembled_model.decode_output(output)
            tepoch.set_postfix({'Batch': idx, 'Train Loss': loss})
        
            tepoch.update(1)
            total_loss += loss
        plt.imshow(np.array(state.params['compressed_transformer']['w_emb']))
        plt.show()
        avg_loss = total_loss / len(train_dataloader)
        tepoch.set_postfix({'Batch': idx, 'Avg Loss': avg_loss})

#%%
plt.imshow(np.array(state.params['compressed_transformer']['w_emb']))

#%%

state.params['compressed_transformer']['w_emb'] = jnp.eye(*state.params['compressed_transformer']['w_emb'].shape)

#%%
it = iter(dataset)

#%%

formatted_input, encoded_tokens, outs, targ_decoded, logits = next(it)

output = compressed_assembled_model.forward(state.params, encoded_tokens)
decoded = compressed_assembled_model.decode_output(output)
print(f"targ: {targ_decoded}, pred: {decoded}")
#assert (targ_decoded == decoded).all()


# %%


# %%

plot_residuals_and_input(
    model=assembled_model,
    inputs=formatted_input,
    figsize=(10, 9)
)

# %%
# @title Plot layer outputs
plot_layer_outputs(
    model=assembled_model,
    inputs=formatted_input,
    figsize=(8, 9)
)
