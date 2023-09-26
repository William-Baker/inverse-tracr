#%%
from tracr.compiler.compiling import compile_rasp_to_model
# from utils.plot import *
import matplotlib.pyplot as plt
import numpy as np
import jax
from random import choice
import os
from tracr.rasp import rasp
jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"

#  =================== init program and compile transformer programs ===========================
program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_op, ops_range = [None]*7

from data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
ops_range=(10, 15)
numeric_range=(5, 8)
vocab_size_range=(5, 8)
numeric_inputs_possible=True
max_seq_len = np.random.randint(4, 9)



n_ops, vocab = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
print(n_ops, vocab)

program, actual_ops = build_program_of_length(vocab, numeric_range, n_ops-2, n_ops+2)




assembled_model, craft_model, rasp_model = compile_rasp_to_model(
    program, vocab, max_seq_len)



ex_input = [choice(vocab) for i in range(max_seq_len-1)]

print(ex_input)

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