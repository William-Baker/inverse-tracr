#%%

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


# def make_hist() -> rasp.SOp:
#   """Returns the number of times each token occurs in the input.

#    (As implemented in the RASP paper.)

#   Example usage:
#     hist = make_hist()
#     hist("abac")
#     >> [2, 1, 2, 1]
#   """
#   same_tok = rasp.Select(rasp.tokens, rasp.tokens,
#                          rasp.Comparison.EQ).named("same_tok")
#   return rasp.SelectorWidth(same_tok).named("hist")


import tracr.compiler.lib as lib

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
    input_seq = [3,2,3,5,2]
  elif program_name == "sort_unique":
    vocab = {1, 2, 3, 4, 5}
    program = lib.make_sort_unique(rasp.tokens, rasp.tokens)
    input_seq = [3,2,1,5,2]
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
    vocab = {1,2,3,4,5}
    program = rasp.Map(lambda x: x > 4, rasp.tokens)
    input_seq = [1,2]
  elif program_name == "map_test_b":
    vocab = {1,2,3,4,5}
    program = rasp.Map(lambda x: x < 1, rasp.Map(lambda x: x > 1, rasp.tokens))
    input_seq = [1,2]
  elif program_name == "map_test_c":
    vocab = {1,2,3,4,5}
    input_seq = [1,2]
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


#%%


# assembled_model, rasp_model, craft_model  = compiling_all.compile_rasp_to_model_returns_all(
#       program=program,
#       vocab=vocab,
#       max_seq_len=max_seq_len,
#       causal=False,
#       compiler_bos="bos",
#       compiler_pad="pad",
#       mlp_exactness=100)

#%%

from tracr.compiler import assemble
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp

COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"



prog_name = "map_test_c"
program, vocab, input_seq = get_program(prog_name, 6)
vocab = set(list(input_seq))
formatted_input = [COMPILER_BOS] + list(input_seq)
max_seq_len=len(input_seq)+1



program = program
vocab = vocab
max_seq_len = max_seq_len
causal = False
compiler_bos = COMPILER_BOS
compiler_pad = COMPILER_PAD
mlp_exactness = 100

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

craft_model_to_transformer.craft_model_to_transformer(
      craft_model=craft_model,
      graph=graph,
      sink=sink,
      max_seq_len=max_seq_len,
      causal=causal,
      compiler_bos=compiler_bos,
      compiler_pad=compiler_pad,
  )


from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft import transformers
from tracr.rasp import rasp
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft import vectorspace_fns
from tracr.transformer import model
from tracr.transformer import encoder
from tracr.compiler.assemble import _get_model_config_and_module_names, _make_embedding_modules, AssembledTransformerModel


"""Turn a craft model into a transformer model."""

# Add the compiler BOS token.
tokens_value_set = (
    graph.nodes[rasp.tokens.label][nodes.VALUE_SET].union(
        {compiler_bos, compiler_pad}))
tokens_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label,
                                                      tokens_value_set)

indices_space = bases.VectorSpaceWithBasis.from_values(
    rasp.indices.label, range(max_seq_len))

categorical_output = rasp.is_categorical(sink[nodes.EXPR])
output_space = bases.VectorSpaceWithBasis(sink[nodes.OUTPUT_BASIS])




def assemble_craft_model(
    craft_model: transformers.SeriesWithResiduals,
    tokens_space: bases.VectorSpaceWithBasis,
    indices_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    categorical_output: bool,
    causal: bool = False,
) -> AssembledTransformerModel:
  """Assembles the given components into a Haiku model with parameters.

  Args:
    craft_model: Model to assemble weights for.
    tokens_space: Vectorspace to embed the input tokens to.
    indices_space: Vectorspace to embed the indices to (position encodings).
    output_space: Vectorspace that the model will write outputs to that should
      be unembedded.
    categorical_output: Whether the output is categorical. If True, we take an
      argmax when unembedding.
    causal: Whether to output a causally-masked model.

  Returns:
    An AssembledTransformerModel that contains the model and parameters of the
      assembled transformer.
  """
  # TODO(b/255936413): Make embeddings only retain the tokens and indices that
  #   are actually used.
  # TODO(b/255936496): Think about enabling layer norm and reversing it somehow

  model_config, module_names = _get_model_config_and_module_names(craft_model)
  model_config.causal = causal

  residual_space = bases.join_vector_spaces(craft_model.residual_space,
                                            tokens_space, indices_space,
                                            output_space)
  residual_labels = [str(basis_dir) for basis_dir in residual_space.basis]

  # Build model with embedding and unembedding layers
  def get_compiled_model():
    transformer = model.Transformer(model_config)
    embed_modules = _make_embedding_modules(
        residual_space=residual_space,
        tokens_space=tokens_space,
        indices_space=indices_space,
        output_space=output_space)
    return model.CompiledTransformerModel(
        transformer=transformer,
        token_embed=embed_modules.token_embed,
        position_embed=embed_modules.pos_embed,
        unembed=embed_modules.unembed,
        use_unembed_argmax=categorical_output)

  @hk.without_apply_rng
  @hk.transform
  def forward(emb):
    compiled_model = get_compiled_model()
    return compiled_model(emb, use_dropout=False)

  params = forward.init(jax.random.PRNGKey(0), jnp.array([[1, 2, 3]]))
  params = {k: dict(v) for k, v in params.items()}

  for key in params:
    if "transformer" in key:
      for par in params[key]:
        params[key][par] = np.zeros_like(params[key][par])

  # Assemble attention and MLP weights.
  project = lambda space: vectorspace_fns.project(residual_space, space).matrix

  for module_name, module in zip(module_names, craft_model.blocks):
    if isinstance(module, transformers.MLP):
      hidden_size = module.fst.output_space.num_dims
      residual_to_fst_input = project(module.fst.input_space)
      snd_output_to_residual = project(module.snd.output_space).T
      params[f"{module_name}/linear_1"]["w"][:, :hidden_size] = (
          residual_to_fst_input @ module.fst.matrix)
      params[f"{module_name}/linear_2"]["w"][:hidden_size, :] = (
          module.snd.matrix @ snd_output_to_residual)
    else:  # Attention module
      query, key, value, linear = [], [], [], []
      for head in module.as_multi().heads():
        key_size = head.w_qk.matrix.shape[1]
        query_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        residual_to_query = project(head.w_qk.left_space)
        query_mat[:, :key_size] = residual_to_query @ head.w_qk.matrix
        query.append(query_mat)

        key_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        key_mat[:, :key_size] = project(head.w_qk.right_space)
        key.append(key_mat)

        value_size = head.w_ov.matrix.shape[1]
        value_mat = np.zeros((residual_space.num_dims, model_config.key_size))
        residual_to_ov_input = project(head.w_ov.input_space)
        value_mat[:, :value_size] = residual_to_ov_input @ head.w_ov.matrix
        value.append(value_mat)

        linear_mat = np.zeros((model_config.key_size, residual_space.num_dims))
        linear_mat[:value_size, :] = project(head.w_ov.output_space).T
        linear.append(linear_mat)

      # Fill up heads that are not used with zero weights
      for _ in range(model_config.num_heads - module.as_multi().num_heads):
        query.append(np.zeros_like(query[0]))
        key.append(np.zeros_like(key[0]))
        value.append(np.zeros_like(value[0]))
        linear.append(np.zeros_like(linear[0]))

      query = einops.rearrange(query,
                               "heads input output -> input (heads output)")
      key = einops.rearrange(key, "heads input output -> input (heads output)")
      value = einops.rearrange(value,
                               "heads input output -> input (heads output)")
      linear = einops.rearrange(linear,
                                "heads input output -> (heads input) output")

      params[f"{module_name}/query"]["w"][:, :] = query
      params[f"{module_name}/key"]["w"][:, :] = key
      params[f"{module_name}/value"]["w"][:, :] = value
      params[f"{module_name}/linear"]["w"][:, :] = linear

  params = jax.tree_util.tree_map(jnp.array, params)
  return AssembledTransformerModel(
      forward=forward.apply,
      get_compiled_model=get_compiled_model,
      params=params,
      model_config=model_config,
      residual_labels=residual_labels,
  )

assembled_model = assemble.assemble_craft_model(
    craft_model=craft_model,
    tokens_space=tokens_space,
    indices_space=indices_space,
    output_space=output_space,
    categorical_output=categorical_output,
    causal=causal,
)


assembled_model.input_encoder = encoder.CategoricalEncoder(
    basis=tokens_space.basis,
    enforce_bos=compiler_bos is not None,
    bos_token=compiler_bos,
    pad_token=compiler_pad,
    max_seq_len=max_seq_len + 1 if compiler_bos is not None else max_seq_len,
)

if categorical_output:
  assembled_model.output_encoder = encoder.CategoricalEncoder(
      basis=output_space.basis,
      enforce_bos=False,
      bos_token=None,
      pad_token=None)
else:
  assembled_model.output_encoder = encoder.NumericalEncoder()





print(f"Runnning {prog_name} with input {input_seq}")
prog_out = assembled_model.apply(formatted_input).decoded
print(f"Prgoram outputs: {prog_out}")




#%%

import haiku as hk

@hk.without_apply_rng
@hk.transform
def compiled_model(emb):
  compiled_model = assembled_model.get_compiled_model()
  return compiled_model(emb, use_dropout=False)

#%%

from tracr.transformer import compressed_model, compressed_model_test

# todo test the new model embedding size 

import haiku as hk
@hk.without_apply_rng
@hk.transform
def forward_superposition(emb, mask):
  return compressed_model.CompressedTransformer(assembled_model.model_config)(emb, mask).output



#%%

#forward_superposition.apply(formatted_input)

import jax.numpy as jnp

seq_len = 4
model_size = 16

emb = np.random.random((1, seq_len, model_size))
mask = np.ones((1, seq_len))
emb, mask = jnp.array(emb), jnp.array(mask)

rng = hk.PRNGSequence(1)
params = forward_superposition.init(next(rng), emb, mask)
activations = forward_superposition.apply(params, next(rng), emb, mask)


#%%

from tracr.transformer import model
import jax.numpy as jnp

embedding_size = 6
unembed_at_every_layer = False
@hk.transform
def forward(emb, mask):
  transformer = compressed_model.CompressedTransformer(
      assembled_model.model_config)
  return transformer(
      emb,
      mask,
      embedding_size=embedding_size,
      unembed_at_every_layer=unembed_at_every_layer,
  )

seq_len = 4
model_size = 16

emb = np.random.random((1, seq_len, model_size))
mask = np.ones((1, seq_len))
emb, mask = jnp.array(emb), jnp.array(mask)

rng = hk.PRNGSequence(1)
params = forward.init(next(rng), emb, mask)
activations = forward.apply(params, next(rng), emb, mask)




#%%

from tracr.craft import bases
indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)


_BOS_DIRECTION = 'map_1'#[basis.name for basis in craft_model.residual_space.basis if '_selector_width_attn_output' in basis.name][0]
_ONE_DIRECTION = 'one'


embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)



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


_BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if '_selector_width_attn_output' in basis.name][0]
_ONE_DIRECTION = 'one'


embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)

output_seq = craft_model.apply(embedded_input)

#%%

plt.imshow(output_seq.magnitudes)

#%%








