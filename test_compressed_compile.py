# %%
import sys
sys.path.append('tracr/')
from utils.compile_with_compressed import compile_with_compressed, COMPILER_BOS
from utils.plot import *
import jax
from random import choice
import os
from argparse import Namespace
import torch
torch.cuda.is_available = lambda : False
from datetime import datetime
from tracr.rasp import rasp
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"



#  =================== init program and compile transformer programs ===========================
program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_op, ops_range = [None]*7

from data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
ops_range=(10, 15)
numeric_range=(5, 8)
vocab_size_range=(5, 8)
numeric_inputs_possible=True
max_seq_len = np.random.randint(4, 9)
CRAFT_TIMEOUT = 2# 0.2 + 0.00001 * max(ops_range) ** 4



n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
print(n_ops, vocab, TARGET_PROGRAM_LENGTH)

program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)

assembled_model, compressed_assembled_model, craft_model, rasp_model = compile_with_compressed(
    program, vocab, max_seq_len, compression=2,
    CRAFT_TIMEOUT=CRAFT_TIMEOUT)

# make the embedding identity so teacher == compressed
compressed_assembled_model.params['compressed_transformer']['w_emb'] = jnp.eye(*compressed_assembled_model.params['compressed_transformer']['w_emb'].shape)
compressed_assembled_model.params['compressed_transformer']['w_emb'] += jax.random.normal(jax.random.PRNGKey(0), compressed_assembled_model.params['compressed_transformer']['w_emb'].shape) / 10


ex_input = [choice(vocab) for i in range(max_seq_len-1)]



from tracr.craft import bases
indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


formatted_input = [COMPILER_BOS] + ex_input

_ONE_DIRECTION = 'one'
_BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if '_selector_width_attn_output' in basis.name][0]


from utils.craft_embeddings import embed_input
embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)


output_seq = craft_model.apply(embedded_input)



from tracr.compiler import nodes
output_space = bases.VectorSpaceWithBasis(rasp_model.sink[nodes.OUTPUT_BASIS])

outs = output_seq.project(output_space)



output = assembled_model.apply(formatted_input)

def decode_outs(output_seq, output_space):
    outs = output_seq.project(output_space)
    labels = outs.magnitudes.argmax(axis=1)
    return [output_space.basis[i].value for i in labels]

print(f"craft {decode_outs(output_seq, output_space)}")
print(f"JAX: {output.decoded}")



from utils.verbose_craft import plot_basis_dir
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
plot_basis_dir(axs, outs, "")

#%%



def attn_layer(x, Q, K, V, linear, heads):
    def project(x, params, heads):
        query_heads = x @ params['w'] + params['b']
        *leading_dims, _ = x.shape
        return query_heads.reshape((*leading_dims, heads, -1))
    query_heads = project(x, Q, heads) # [T', H, Q=K]
    key_heads = project(x, K, heads) # [T, H, Q=K]
    value_heads = project(x, V, heads) # [T, H, V]
    attn_logits = np.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(key_heads.shape[-1]).astype(x.dtype)
    #attn_weights = softmax(attn_logits)
    attn_weights = np.array(jax.nn.softmax(attn_logits))  # [H, T', T]
    attn = np.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    *leading_dims, sequence_length, _ = x.shape
    attn = np.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = attn @ linear['w'] + linear['b']
    return final_projection

def mlp_layer(x, fst, snd):
    f = x @ fst['w'] + fst['b']
    #f = np.max(f, 0)
    f = np.array(jax.nn.relu(f))
    s = f @ snd['w'] + snd['b']
    return s

from collections import defaultdict
def interp_params(params):
    new_params = defaultdict(lambda: dict())
    for key, val in params.items():
        try:
            model_name, layer, block, p = key.split('/')
            np_vals = dict([(k,np.array(v)) for k,v in val.items()])
            new_params['/'.join([model_name, layer, block])]['/'.join([block, p])] = np_vals
        except:
            pass
    return new_params

def execute(params, x, heads):
    outs = []
    resid = x
    for layer, blocks in params.items():
        block_1, p_1 = list(blocks.keys())[0].split('/')
        print(layer, blocks.keys())
        if block_1 == 'attn':
            Q, K, V, linear = blocks['attn/query'], blocks['attn/key'], blocks['attn/value'], blocks['attn/linear']
            res = attn_layer(resid, Q, K, V, linear, heads)
            print(f"attn_out {res.mean()}")
            outs.append(res)
            resid += res
        elif block_1 == 'mlp':
            fst, snd = blocks['mlp/linear_1'], blocks['mlp/linear_2']
            res = mlp_layer(resid, fst, snd)
            print(f"mlp_out {res.mean()}")
            outs.append(res)
            resid += res
        print(resid.mean())
        print(resid.shape)
    return outs

ps = interp_params(compressed_assembled_model.params)
encoded = assembled_model.encode_input(formatted_input)
x = assembled_model.forward_emb(assembled_model.params, encoded)
outs = execute(ps, x, assembled_model.model_config.num_heads)

jax_outs = assembled_model.forward_no_emb(assembled_model.params, x)


for i in range(len(outs)):
    plt.imshow(outs[i].squeeze())
    plt.show()
    plt.imshow(np.array(jax_outs.transformer_output.layer_outputs[i]).squeeze())
    plt.show()



#%%


def execute_compressed(params, x, heads, w_emb):
    outs = []
    compress = lambda x: x @ w_emb.T 
    decompress = lambda x: x @ w_emb
    resid = compress(x)
    for layer, blocks in params.items():
        block_1, p_1 = list(blocks.keys())[0].split('/')
        print(layer, blocks.keys())
        decompressed = decompress(resid)
        if block_1 == 'attn':
            Q, K, V, linear = blocks['attn/query'], blocks['attn/key'], blocks['attn/value'], blocks['attn/linear']
            res = attn_layer(decompressed, Q, K, V, linear, heads)
            print(f"attn_out {res.mean()}")
            outs.append(res)
            resid += compress(res)
        elif block_1 == 'mlp':
            fst, snd = blocks['mlp/linear_1'], blocks['mlp/linear_2']
            res = mlp_layer(decompressed, fst, snd)
            print(f"mlp_out {res.mean()}")
            outs.append(res)
            resid += compress(res)
        print(resid.mean())
        print(resid.shape)
    return outs
ps = interp_params(compressed_assembled_model.params)
encoded = compressed_assembled_model.encode_input(formatted_input)
x = compressed_assembled_model.forward_emb(compressed_assembled_model.params, encoded)
outs = execute_compressed(ps, x, compressed_assembled_model.model_config.num_heads, compressed_assembled_model.params['compressed_transformer']['w_emb'])
jax_outs = compressed_assembled_model.forward_no_emb(compressed_assembled_model.params, x)


for i in range(len(outs)):
    plt.imshow(outs[i].squeeze())
    plt.show()
    plt.imshow(np.array(jax_outs.transformer_output.layer_outputs[i]).squeeze())
    plt.show()

#%%


def attn_layer(x, Q, K, V, linear, heads, w_emb):

    def project(x, w, b, heads):
        query_heads = x @ w + b
        *leading_dims, _ = x.shape
        return query_heads.reshape((*leading_dims, heads, -1))
    query_heads = project(x, (w_emb @ Q['w']), Q['b'], heads) # [T', H, Q=K]
    key_heads = project(x, (w_emb @ K['w']), K['b'], heads) # [T, H, Q=K]
    value_heads = project(x, (w_emb @ V['w']) , V['b'], heads) # [T, H, V]
    attn_logits = np.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(key_heads.shape[-1]).astype(x.dtype)
    #attn_weights = softmax(attn_logits)
    attn_weights = np.array(jax.nn.softmax(attn_logits))  # [H, T', T]
    attn = np.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    *leading_dims, sequence_length, _ = x.shape
    attn = np.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]


    # Apply another projection to get the final embeddings.
    final_projection = attn @ (linear['w'] @ w_emb.T)  + linear['b'] @ w_emb.T
    #assert (linear['b'] == np.zeros_like(linear['b'])).all()

    uncompressed_for_comparison = (attn @ linear['w'])  + linear['b']

    return final_projection, uncompressed_for_comparison

def mlp_layer(x, fst, snd, w_emb):
    # compress = lambda x: x @ w_emb.T 
    # decompress = lambda x: x @ w_emb
    f = x @ (w_emb @ fst['w']) + fst['b']
    #f = np.max(f, 0)
    f = np.array(jax.nn.relu(f))
    s = (f @ snd['w'] @ w_emb.T + snd['b'] @ w_emb.T) 
    uncompressed_for_comparison = f @ snd['w'] + snd['b']

    return s, uncompressed_for_comparison

def execute(params, x, heads,w_emb):
    outs = []
    compress = lambda x: x @ w_emb.T 
    decompress = lambda x: x @ w_emb
    #resid = compress(x)
    resid = x
    print(resid.mean())
    print(resid.shape)
    for layer, blocks in params.items():
        block_1, p_1 = list(blocks.keys())[0].split('/')
        print(layer, blocks.keys())
        if block_1 == 'attn':
            Q, K, V, linear = blocks['attn/query'], blocks['attn/key'], blocks['attn/value'], blocks['attn/linear']
            res, uncomp = attn_layer(resid, Q, K, V, linear, heads, w_emb)
            print(f"attn_out {res.mean()}")
            outs.append(uncomp)
            resid += res
        elif block_1 == 'mlp':
            fst, snd = blocks['mlp/linear_1'], blocks['mlp/linear_2']
            res, uncomp = mlp_layer(resid, fst, snd,w_emb)
            # print(f"mlp_out {res.mean()}")
            outs.append(uncomp)
            resid += res
        print(f"resid: {resid.mean()}")
        print(resid.shape)
        
    return outs

ps = interp_params(compressed_assembled_model.params)

ps = interp_params(compressed_assembled_model.params)
encoded = compressed_assembled_model.encode_input(formatted_input)
x = compressed_assembled_model.forward_emb(compressed_assembled_model.params, encoded)
outs = execute(ps, x, compressed_assembled_model.model_config.num_heads,compressed_assembled_model.params['compressed_transformer']['w_emb'])

#%%
jax_outs = compressed_assembled_model.forward_no_emb(compressed_assembled_model.params, x)

#%%
w_emb = compressed_assembled_model.params['compressed_transformer']['w_emb']
decompress = lambda x: x @ w_emb

for i in range(len(outs)):
    plt.imshow(outs[i].squeeze())
    plt.show()
    plt.imshow(np.array(jax_outs.transformer_output.layer_outputs[i]).squeeze())
    plt.show()