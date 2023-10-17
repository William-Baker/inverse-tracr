#%%

# Imports for compile_rasp_to_model_returns_all
from typing import Set
from tracr.compiler import assemble
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp




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


def compile_rasp_to_model_returns_all(
    program: rasp.SOp,
    vocab: Set[rasp.Value],
    max_seq_len: int,
    causal: bool = False,
    compiler_bos: str = COMPILER_BOS,
    compiler_pad: str = COMPILER_PAD,
    mlp_exactness: int = 100) -> assemble.AssembledTransformerModel:

  if compiler_bos in vocab:
    raise ValueError("Compiler BOS token must not be present in the vocab. "
                     f"Found '{compiler_bos}' in {vocab}")

  if compiler_pad in vocab:
    raise ValueError("Compiler PAD token must not be present in the vocab. "
                     f"Found '{compiler_pad}' in {vocab}")

  rasp_model = rasp_to_graph.extract_rasp_graph(program)
  graph, sources, sink = rasp_model.graph, rasp_model.sources, rasp_model.sink

  basis_inference.infer_bases(
      graph,
      sink,
      vocab,
      max_seq_len,
  )

  expr_to_craft_graph.add_craft_components_to_rasp_graph(
      graph,
      bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
      mlp_exactness=mlp_exactness,
  )

  craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

  return craft_model_to_transformer.craft_model_to_transformer(
      craft_model=craft_model,
      graph=graph,
      sink=sink,
      max_seq_len=max_seq_len,
      causal=causal,
      compiler_bos=compiler_bos,
      compiler_pad=compiler_pad,
  ), rasp_model, craft_model




def sum_of_inputs(x: rasp.SOp) -> rasp.SOp:
    x = rasp.numerical(x)
    before = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    means = rasp.Aggregate(before, rasp.tokens)  # returns sequence s_i = mean_{j<=i} input_j
    sums = rasp.SequenceMap(lambda x, y: x*y, means, rasp.indices+1)
    return sums


sums = sum_of_inputs(rasp.tokens)
# print(sums([3,2,1,1]))  # output of RASP program

# model = compile_rasp(sums)
# model = compiling.compile_rasp_to_model(sums, vocab={1,2,3}, max_seq_len=5, compiler_bos="BOS")
# print(model.apply(["BOS", 3,2,1,1]).decoded)  # different output when compiled

vocab={1,2,3}
max_seq_len=5
program = sums

assembled_model, rasp_model, craft_model = compile_rasp_to_model_returns_all(
    program, vocab, max_seq_len, compiler_bos=COMPILER_BOS)



#ex_input = [choice(list(vocab)) for i in range(max_seq_len-1)]
ex_input = [3,2,1,1]

print(ex_input)

from tracr.craft import bases
indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


formatted_input = [COMPILER_BOS] + ex_input

_ONE_DIRECTION = 'one'
_BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if (basis.value == 'compiler_bos')][0]

#%%
# from inverse_tracr.utils.craft_embeddings import embed_input
def embed_input(input_seq, input_space, _BOS_DIRECTION, _ONE_DIRECTION, BOS_VALUE='compiler_bos'):
  bos_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_BOS_DIRECTION, BOS_VALUE))
  one_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_ONE_DIRECTION))
  embedded_input = [bos_vec + one_vec]
  for i, val in enumerate(input_seq):
    i_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection(rasp.indices.label, i))
    val_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection(rasp.tokens.label, val))
    embedded_input.append(i_vec + val_vec + one_vec)
  return bases.VectorInBasis.stack(embedded_input)

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

# %%
