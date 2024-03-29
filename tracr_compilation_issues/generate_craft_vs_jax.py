#%%

from inverse_tracr.utils.compiling_all import compile_rasp_to_model_returns_all
import matplotlib.pyplot as plt
import numpy as np
import jax
from random import choice
import os
from tracr.rasp import rasp
from tracr.craft import bases
from inverse_tracr.utils.craft_embeddings import embed_input
from tracr.compiler import nodes
from inverse_tracr.utils.verbose_craft import plot_basis_dir
from inverse_tracr.utils.time_sensitive import time_sensitive

jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"

#  =================== init program and compile transformer programs ===========================
program, vocab, max_seq_len, assembled_model, compressed_assembled_model, actual_op, ops_range = [None]*7

from inverse_tracr.data.dataset import choose_vocab_and_ops, build_program_of_length,program_craft_generator
ops_range=((5,6))
numeric_range=(3,4)
vocab_size_range=(3,4)
numeric_inputs_possible=False
max_seq_len = np.random.randint(3,5)



def gen_sample():
    def timed():
        n_ops, vocab = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
        print(n_ops, vocab)
        program, actual_ops = build_program_of_length(vocab, numeric_range, n_ops-2, n_ops+2)
        for x in actual_ops: print(x)
        assembled_model, rasp_model, craft_model = compile_rasp_to_model_returns_all(
            program, set(vocab), max_seq_len)
        return assembled_model, rasp_model, craft_model, program, vocab, actual_ops
    ret = None
    while ret is None:
        ret = time_sensitive(timed, timeout=5)
    return ret
    

assembled_model, rasp_model, craft_model, program, vocab, actual_ops = gen_sample()

ex_input = [choice(vocab) for i in range(max_seq_len-1)]
print(ex_input)


print(f"prog outs: {program(ex_input)}")

indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


formatted_input = [COMPILER_BOS] + ex_input

_ONE_DIRECTION = 'one'
_BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if (basis.value == 'compiler_bos')][0]



embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)


output_seq = craft_model.apply(embedded_input)



output_space = bases.VectorSpaceWithBasis(rasp_model.sink[nodes.OUTPUT_BASIS])

outs = output_seq.project(output_space)



output = assembled_model.apply(formatted_input)

def decode_outs(output_seq, output_space):
    outs = output_seq.project(output_space)
    labels = outs.magnitudes.argmax(axis=1)
    return [output_space.basis[i].value for i in labels]

print(f"craft {decode_outs(output_seq, output_space)}")
print(f"JAX: {output.decoded}")


fig, axs = plt.subplots(1, 1, figsize=(5, 5))
plot_basis_dir(axs, outs, "")

for x in actual_ops: print(x)

#%%