#%%

import jax
import sys
sys.path.append('tracr/')

from utils.plot import *

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')

from tracr.compiler import compiling
from utils import compiling_all
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

input_seq = "hello"
program = make_hist()
vocab = set(list(input_seq))
formatted_input = ["bos"] + list(input_seq)
max_seq_len=len(input_seq)+1

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

def format_basis_seq(seq):
  def shorten(s):
    s = str(s)
    return s if len(s) < 10 else s[:4] + '~' + s[-5:] 
      
  #return [f"{x.name} {x.value}" for x in seq]
  return [f"{shorten(x.name)} {shorten(x.value)}" for x in seq]

def format_basis(space):
  return format_basis_seq(space.basis)

def show_mlp(mlp):
  def imshow_linear(ax, layer):
    ax.imshow(layer.matrix)
    ax.set_xticks(np.arange(0, layer.matrix.shape[1], 1.0))
    ax.set_yticks(np.arange(0, layer.matrix.shape[0], 1.0))
    ax.set_yticklabels(format_basis(layer.input_space))
    ax.set_xticklabels(format_basis(layer.output_space), rotation=90)
    ax.tick_params(axis="both", direction="in", pad=5, labelsize=6)


  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7, 3.5))
  fig.suptitle('layer X, mlp X')#, fontsize=8)

  ax1.set_title('FST (Dense1)')
  imshow_linear(ax1, mlp.fst)

  ax2.set_title('SND (Dense2)')
  imshow_linear(ax2, mlp.snd)

  plt.tight_layout()
  plt.show()

show_mlp(mlp)
#%%



from tracr.craft import transformers
from tracr.craft import vectorspace_fns as vs_fns

def make_mlp_verbose(mlp):
  def mlp_apply(self, x: bases.VectorInBasis) -> bases.VectorInBasis:
    from tracr.craft.vectorspace_fns import project
    from tracr.craft.transformers import relu

    assert x in self.residual_space

    fig, axs = plt.subplots(1,8, figsize=(7, 3.5))

    axs[0].set_title('In')
    axs[0].imshow(x.magnitudes)
    axs[0].set_yticks(np.arange(0, x.magnitudes.shape[0], 1.0))
    axs[0].set_yticklabels(format_basis(self.fst.input_space))
    axs[0].set_xticks(np.arange(0, x.magnitudes.shape[1], 1.0))
    axs[0].set_xticklabels(format_basis_seq(x.basis_directions), rotation=90)
    axs[0].tick_params(axis="both", direction="in", pad=5, labelsize=6)

    x = project(self.residual_space, self.fst.input_space)(x)

    axs[1].set_title('PIn')
    axs[1].imshow(x.magnitudes)

    hidden = self.fst(x)

    axs[2].set_title('FST')
    axs[2].imshow(self.fst.matrix)


    axs[3].set_title('h')
    axs[3].imshow(hidden.magnitudes)
    axs[3].set_yticks(np.arange(0, self.fst.matrix.shape[1], 1.0))
    axs[3].set_yticklabels(format_basis(self.fst.output_space))

    hidden = relu(hidden)

    axs[4].set_title('hRelu')
    axs[4].imshow(hidden.magnitudes)

    out = self.snd(hidden)
    axs[5].set_title('SND')
    axs[5].imshow(self.snd.matrix)

    axs[6].set_title('out')
    axs[6].imshow(out.magnitudes)
    axs[6].set_yticks(np.arange(0, self.snd.matrix.shape[1], 1.0))
    axs[6].set_yticklabels(format_basis(self.snd.output_space))

    projected = project(self.snd.output_space, self.residual_space)(out)
    axs[7].set_title('Pout')
    axs[7].imshow(projected.magnitudes)
    axs[7].set_xticks(np.arange(0, projected.magnitudes.shape[1], 1.0))
    plt.tight_layout()
    plt.show()

    return projected

  from types import MethodType
  mlp.apply = MethodType( mlp_apply, mlp )

from tracr.craft import bases
def test_mlp(with_residual_stream, same_in_out):
  i = bases.VectorSpaceWithBasis.from_values("i", [1, 2])
  if same_in_out:
    o, rs = i, i
    expected_result = np.array([
        #o1 o2
        [1, 0],
        [0, 1],
    ])
  else:
    o = bases.VectorSpaceWithBasis.from_values("o", [1, 2])
    rs = bases.direct_sum(i, o)
    expected_result = np.array([
        #i1 i2 o1 o2
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
  h = bases.VectorSpaceWithBasis.from_values("p", [1, 2])

  seq = bases.VectorInBasis(
      i.basis,
      np.array([
          #i1  i2
          [1, -1],
          [-1, 1],
      ])).project(rs)
  print(format_basis(rs))
  print(format_basis(i))
  print(format_basis(h))
  print(format_basis(o))
  mlp = transformers.MLP(
      fst=vs_fns.Linear(i, h, np.eye(2)),
      snd=vs_fns.Linear(h, o, np.eye(2)),
      residual_space=rs if with_residual_stream else None,
  )
  show_mlp(mlp)
  
  # replace the apply method with our new meh
  make_mlp_verbose(mlp)


  res = mlp.apply(seq)
  res == bases.VectorInBasis(rs.basis, expected_result)


test_mlp(False, False)



# %%


make_mlp_verbose(craft_model.blocks[1])

#%%

#assembled_model.apply(formatted_input).decoded

#mlp.residual_space


seq = bases.VectorInBasis(
    craft_model.blocks[0].input_space.basis,
    formatted_input).project(mlp.residual_space)

craft_model.apply(formatted_input)
# %%

format_basis(craft_model.residual_space)


#%%

from tracr.compiler.rasp_to_craft_integration_test import _make_input_space, _embed_input, _embed_output
#input_space = _make_input_space(vocab, max_seq_len)
input_space = craft_model.residual_space
embedded_input = _embed_input(formatted_input, input_space=input_space)

#%%

_BOS_DIRECTION = input_space.basis[7].name
_ONE_DIRECTION = input_space.basis[8].name


def _make_input_space(vocab, max_seq_len):
  tokens_space = bases.VectorSpaceWithBasis.from_values("tokens", vocab)
  indices_space = bases.VectorSpaceWithBasis.from_values(
      "indices", range(max_seq_len))
  one_space = bases.VectorSpaceWithBasis.from_names([_ONE_DIRECTION])
  bos_space = bases.VectorSpaceWithBasis.from_names([_BOS_DIRECTION])
  input_space = bases.join_vector_spaces(tokens_space, indices_space, one_space,
                                         bos_space)

  return input_space


def _embed_input(input_seq, input_space):
  bos_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_BOS_DIRECTION))
  one_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_ONE_DIRECTION))
  embedded_input = [bos_vec + one_vec]
  for i, val in enumerate(input_seq):
    i_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("hist_1", i))
    val_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("tokens", val))
    embedded_input.append(i_vec + val_vec + one_vec)
  return bases.VectorInBasis.stack(embedded_input)




input_space = craft_model.residual_space
embedded_input = _embed_input(formatted_input, input_space=input_space)

output_seq = craft_model.apply(embedded_input)

#%%

def _embed_output(output_seq, output_space, categorical_output):
  embedded_output = []
  output_label = output_space.basis[0].name
  for x in output_seq:
    if x is None:
      out_vec = output_space.null_vector()
    elif categorical_output:
      out_vec = output_space.vector_from_basis_direction(
          bases.BasisDirection(output_label, x))
    else:
      out_vec = x * output_space.vector_from_basis_direction(
          output_space.basis[0])
    embedded_output.append(out_vec)
  return bases.VectorInBasis.stack(embedded_output)

# from tracr.craft.bases import BasisDirection, VectorSpaceWithBasis

# output_space = [BasisDirection(name='hist_1', value=0),
# BasisDirection(name='hist_1', value=1),
# BasisDirection(name='hist_1', value=2),
# BasisDirection(name='hist_1', value=3),
# BasisDirection(name='hist_1', value=4),
# BasisDirection(name='hist_1', value=5),
# BasisDirection(name='hist_1', value=6)]



# #_embed_output(output_seq, craft_model.residual_space, True)#bases.VectorInBasis.stack(output_seq.basis_directions), True)
# _embed_output(output_seq, VectorSpaceWithBasis(output_space), True)