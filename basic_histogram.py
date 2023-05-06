#%%

import jax
import sys
sys.path.append('tracr/')

from utils.plot import *

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')

from tracr.compiler import compiling
from tracr.compiler import lib
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


#%%

input = "hello"
program = make_hist()
vocab = set(list(input))
formatted_input = ["bos"] + list(input)

assembled_model, rasp_model, craft_model = compiling.compile_rasp_to_model_returns_all(
      program=program,
      vocab=vocab,
      max_seq_len=len(vocab)+1,
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

# %%

attn = craft_model.blocks[0]
mlp = craft_model.blocks[1]

#%%

attn_head = list(attn.heads())[0]
# %%

attn_head.w_ov.matrix
attn_head.w_qk.matrix
attn_head.w_ov_residual

#%%

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(7, 3.5))

fig.suptitle('layer X, head X')#, fontsize=8)

# Line plots
ax1.set_title('W_QK')
ax1.imshow(attn_head.w_qk.matrix)

ax2.set_title('W_OV')
ax2.imshow(attn_head.w_ov.matrix)

ax3.set_title("Resid Basis")
ax3.imshow(np.stack([x.magnitudes for x in attn_head.residual_space.basis_vectors()]))

plt.tight_layout()
plt.show()

# %%


