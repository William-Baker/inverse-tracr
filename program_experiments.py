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

# SOp is an n-length vector
# Selector is an nxn matrix

# Aggregate(sel: Selector, sop: Sop, default: Optional[Union[None, int, float, str, bool]]) -> Sop
# masks the SOp according to each row of the selector, then takes the average over the non-masked values. Giving a new value for each row, forming an SOp
#   e.g. aggregate(sel, [124]):
#     F T T -> 1 * 0 + 2 * 1 + 4 * 1 / 2 = 3
#     F F F -> 0 + 0 + 0 = 0
#     T F F -> 1 * 1 + 0 + 0 / 1 = 1
#     => [301]

# SelectorWidth(s: Selector) -> Sop - computes the average over rows of a matrix returning a vector

# SelectorOr(A: Selector, B: Selector) -> Selector  - computes elementwise OR over 2 matrices returning a matrix
# SelectorAnd(A: Selector, B: Selector) -> Selector  - computes elementwise AND over 2 matrices returning a matrix
# SelectorNot(A: Selector) -> Selector - computes elementwise NOT over a matrix returning a matrix





# Select(k: Sop, q:Sop, p: Predicate) -> Selector - applies a predicate operation over k_i and q_j, forming matrix S_ij

# Predicate - Comparisions:
#   rasp.Comparison.EQ
#   rasp.Comparison.FALSE
#   rasp.Comparison.TRUE
#   rasp.Comparison.GEQ
#   rasp.Comparison.GT
#   rasp.Comparison.LEQ
#   rasp.Comparison.LT
#   rasp.Comparison.NEQ

# Annotations
#   numerical(sop) - annotates that the vecotr is numerical
#   categorical(sop) - annotates that the vecotr is categorical

# SOp
#  > TokensType
#  > IndicesType
#  > LengthType

# Value = Union[None, int, float, str, bool]
# Map(fx: f(Value) ->Value, s: SOp, simplify: bool) -> SOp - calls fx on each element in s
#     > if simplify = True, Map(fx, Map(gx, X, False), True) = Map(fx(gx(..)), X, False) - nestled mappings will be simplified

# SequenceMap(fx: f(Value, Value) -> Value, A: SOp, B: SOp) -> SOp - calls fx on each element in A[i] and B[i]
# LinearSequenceMap(A: SOp, B: SOp, wa: float, wb: float) -> SOp - same as SequenceMap(lambda x y: wa * x + wb * y, A, B)

# fx can be anything since the function is applied over the input_value_set, and the results are memorised within the model

# Full(x) -> SOp - [x] * input_length - filled list with value x

# SOP operators
# [<, >, <=, >=, ==, !=] - other is Value -> Map(A, lambda x: x @ Value) operations
# [+, -, *, /, , &&, ||] - if other is a SOp -> SequenceMap(A, B, lambda), else if other is value Map(A, B, lambda)
# [! ] - if self is Value or SOp. Map(A, lambda x: not x)





def make_program() -> rasp.SOp:
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
def make_shuffle_dyck(pairs) -> rasp.SOp:
  """Returns 1 if a set of parentheses are balanced, 0 else.

   (As implemented in the RASP paper.)

  Example usage:
    shuffle_dyck2 = make_shuffle_dyck(pairs=["()", "{}"])
    shuffle_dyck2("({)}")
    >> [1, 1, 1, 1]
    shuffle_dyck2("(){)}")
    >> [0, 0, 0, 0, 0]

  Args:
    pairs: List of pairs of open and close tokens that each should be balanced.
  """
  from tracr.compiler.lib import make_pair_balance, length
  assert len(pairs) >= 1

  # Compute running balance of each type of parenthesis
  balances = []
  for pair in pairs:
    assert len(pair) == 2
    open_token, close_token = pair
    balance = make_pair_balance(
        rasp.tokens, open_token=open_token,
        close_token=close_token).named(f"balance_{pair}")
    balances.append(balance)

  # Check if balances where negative anywhere -> parentheses not balanced
  any_negative = balances[0] < 0
  for balance in balances[1:]:
    any_negative = any_negative | (balance < 0)

  # Convert to numerical SOp
  any_negative = rasp.numerical(rasp.Map(lambda x: x,
                                         any_negative)).named("any_negative")

  select_all = rasp.Select(rasp.indices, rasp.indices,
                           rasp.Comparison.TRUE).named("select_all")
  has_neg = rasp.numerical(rasp.Aggregate(select_all, any_negative,
                                          default=0)).named("has_neg")

  # Check if all balances are 0 at the end -> closed all parentheses
  all_zero = balances[0] == 0
  for balance in balances[1:]:
    all_zero = all_zero & (balance == 0)

  select_last = rasp.Select(rasp.indices, length - 1,
                            rasp.Comparison.EQ).named("select_last")
  last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

  not_has_neg = (~has_neg).named("not_has_neg")
  return (last_zero & not_has_neg).named("shuffle_dyck")

make_shuffle_dyck(pairs=["{}", "[]"])

#%%


input_seq = "hello"
program = make_program()
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





