#%%
from inverse_tracr.utils.compiling_all import compile_rasp_to_model_returns_all
import matplotlib.pyplot as plt
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

# vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
# max_seq_len = 8
# program = so7
# se1 = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y)
# so2 = rasp.Aggregate(se1, rasp.indices)
# se2 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x < y)
# so3 = rasp.SelectorWidth(se2)
# se3 = rasp.Select(so3, so2, lambda x, y: x!=y)
# so6 = rasp.SequenceMap(lambda x,y: x-y, so3, so3)
# so7 = rasp.Aggregate(se3, so6)



vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
max_seq_len = 4 # maybe 5

v1 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x > y)
v2 = rasp.Map(lambda y: vocab[0] and y, rasp.indices)
v3 = rasp.SelectorWidth(v1)
v4 = rasp.Select(v2, v3, lambda x, y: x < y)
v5 = rasp.Aggregate(v4, v3)

program = v5


assembled_model, rasp_model, craft_model = compile_rasp_to_model_returns_all(
    program, set(vocab), max_seq_len)



ex_input = [choice(vocab) for i in range(max_seq_len-1)]
print(ex_input)

from tracr.craft import bases
indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


formatted_input = [COMPILER_BOS] + ex_input

_ONE_DIRECTION = 'one'
_BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if (basis.value == 'compiler_bos')][0]

#%%
from inverse_tracr.utils.craft_embeddings import embed_input


embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)

#%%

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



from inverse_tracr.utils.verbose_craft import plot_basis_dir
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
plot_basis_dir(axs, outs, "")

#%%