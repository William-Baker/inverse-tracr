#%%
TF_CPP_MIN_LOG_LEVEL=0
import jax
import sys
sys.path.append('tracr/')

from utils.plot import *

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')


from utils import compiling_all
from utils.verbose_craft import make_craft_model_verbose
from utils.craft_embeddings import embed_input

from tracr.rasp import rasp


def make_hist() -> rasp.SOp:
  """Returns the number of times each token occurs in the input.

   (As implemented in the RASP paper.)

  Example usage:
    hist = make_hist()
    hist("abac")
    >> [2, 1, 2, 1]
  """
  same_tok = rasp.Select(rasp.tokens, rasp.tokens,
                         rasp.Comparison.EQ).named("same_tok")
  return rasp.SelectorWidth(same_tok).named("hist")



def make_hist() -> rasp.SOp:
  """Returns the number of times each token occurs in the input.

   (As implemented in the RASP paper.)

  Example usage:
    hist = make_hist()
    hist("abac")
    >> [2, 1, 2, 1]
  """
  same_tok = rasp.Select(rasp.tokens, rasp.tokens,
                         rasp.Comparison.EQ).named("same_tok")
  same_tok3 = rasp.SelectorNot(same_tok).named("same_tok3")
  return rasp.SelectorWidth(same_tok3).named("hist")

#%%

TF_CPP_MIN_LOG_LEVEL=0

input_seq = "hello"
program = make_hist()
vocab = set(list(input_seq))
formatted_input = ["bos"] + list(input_seq)
max_seq_len=len(input_seq)+1

TF_CPP_MIN_LOG_LEVEL=0
assembled_model, rasp_model, craft_model = compiling_all.compile_rasp_to_model_returns_all(
      program=program,
      vocab=vocab,
      max_seq_len=max_seq_len,
      causal=False,
      compiler_bos="bos",
      compiler_pad="pad",
      mlp_exactness=100)

#%%

assembled_model.apply(formatted_input).decoded

# %%

plot_residuals_and_input(
  model=assembled_model,
  inputs=formatted_input,
  figsize=(10, 9)
)

# %%
#@title Plot layer outputs
plot_layer_outputs(
  model=assembled_model,
  inputs = formatted_input,
  figsize=(8, 9)
)

#%%
import networkx as nx
nx.draw(rasp_model.graph, with_labels=True)



#%%

make_craft_model_verbose(craft_model)


from tracr.craft import bases
indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


_BOS_DIRECTION = craft_model.residual_space.basis[7].name
_ONE_DIRECTION = craft_model.residual_space.basis[8].name

embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)

output_seq = craft_model.apply(embedded_input)



#%%


plt.imshow(output_seq.magnitudes)

#%%

from utils.verbose_craft import plot_basis_dir

fig, axs = plt.subplots(1, 1, figsize=(5, 3))
plot_basis_dir(axs, output_seq, f'POut + resid')
plt.show()


#%%





