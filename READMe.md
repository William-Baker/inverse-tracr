`~/go/bin/pprof -svg ~/inverse-tracr/memory.prof`


# MAP - MLP Layer
Maps from known inputs to all outputs from applying lambda to inputs
1. Input Space
2. Lambda

given categoric inputs cat->cat, cat->num:
 - For each direction in the input space, apply lambda, giving a vector  
 - Concat this vector with the weight matrix  
   -> yeilds a matrix with as many columns as input basis directions, and rows as output directions

```
def operation_fn(direction):
    if direction in input_space:
      output_direction = operation(direction)
      if output_direction in output_space:
        return output_space.vector_from_basis_direction(output_direction)
    return output_space.null_vector()

  first_layer = vectorspace_fns.Linear.from_action(input_space, output_space,
                                                   operation_fn)

  second_layer = vectorspace_fns.project(output_space, output_space)

  return transformers.MLP(first_layer, second_layer)
```

given numeric inputs num->cat, num->num
 - uses a descritisation layer if the inputs are numeric
 - this layer acts similarly to an SVM
 - The second layer then maps from the outputs of this SVM in the input domain to the output domain

```
bases.ensure_dims(input_space, num_dims=1, name="input_space")
  bases.ensure_dims(output_space, num_dims=1, name="output_space")
  bases.ensure_dims(one_space, num_dims=1, name="one_space")

  input_space = bases.join_vector_spaces(input_space, one_space)
  out_vec = output_space.vector_from_basis_direction(output_space.basis[0])

  discretising_layer = _get_discretising_layer(
      input_value_set=input_value_set,
      f=f,
      hidden_name=hidden_name,
      one_direction=one_space.basis[0],
      large_number=large_number)
  first_layer = vectorspace_fns.Linear.from_action(
      input_space, discretising_layer.hidden_space, discretising_layer.action)

  def second_layer_action(
      direction: bases.BasisDirection) -> bases.VectorInBasis:
    # output = sum(
    #     (hidden_k_0 - hidden_k_1) * (f(input[k]) - f(input[k-1]))
    #   for all k)
    if direction.name == f"{hidden_name}start":
      return discretising_layer.output_values[0] * out_vec
    k, i = direction.value
    # add hidden_k_0 and subtract hidden_k_1
    sign = {0: 1, 1: -1}[i]
    return sign * (discretising_layer.output_values[k] -
                   discretising_layer.output_values[k - 1]) * out_vec

  second_layer = vectorspace_fns.Linear.from_action(
      discretising_layer.hidden_space, output_space, second_layer_action)

  return transformers.MLP(first_layer, second_layer)
```


# Sequence Map
1. Input x
2. Input y
3. Lambda

Create positive +1 and negative -1 basis directions for x and y
Define the action for the first layer to map from the input space to the +x -x +y -y space
Define the action for the second layer to map from the previous (hidden space) 

```
def linear_sequence_map_numerical_mlp(
    input1_basis_direction: bases.BasisDirection,
    input2_basis_direction: bases.BasisDirection,
    output_basis_direction: bases.BasisDirection,
    input1_factor: float,
    input2_factor: float,
    hidden_name: bases.Name = "__hidden__",
) -> transformers.MLP:
  """Returns an MLP that encodes a linear function f(x, y) = a*x + b*y.

  Args:
    input1_basis_direction: Basis direction that encodes the input x.
    input2_basis_direction: Basis direction that encodes the input y.
    output_basis_direction: Basis direction to write the output to.
    input1_factor: Linear factor a for input x.
    input2_factor: Linear factor a for input y.
    hidden_name: Name for hidden dimensions.
  """
  input_space = bases.VectorSpaceWithBasis(
      [input1_basis_direction, input2_basis_direction])
  output_space = bases.VectorSpaceWithBasis([output_basis_direction])
  out_vec = output_space.vector_from_basis_direction(output_basis_direction)

  hidden_directions = [
      bases.BasisDirection(f"{hidden_name}x", 1),
      bases.BasisDirection(f"{hidden_name}x", -1),
      bases.BasisDirection(f"{hidden_name}y", 1),
      bases.BasisDirection(f"{hidden_name}y", -1)
  ]
  hidden_space = bases.VectorSpaceWithBasis(hidden_directions)
  x_pos_vec, x_neg_vec, y_pos_vec, y_neg_vec = (
      hidden_space.vector_from_basis_direction(d) for d in hidden_directions)

  def first_layer_action(
      direction: bases.BasisDirection) -> bases.VectorInBasis:
    output = hidden_space.null_vector()
    if direction == input1_basis_direction:
      output += x_pos_vec - x_neg_vec
    if direction == input2_basis_direction:
      output += y_pos_vec - y_neg_vec
    return output

  first_layer = vectorspace_fns.Linear.from_action(input_space, hidden_space,
                                                   first_layer_action)

  def second_layer_action(
      direction: bases.BasisDirection) -> bases.VectorInBasis:
    if direction.name == f"{hidden_name}x":
      return input1_factor * direction.value * out_vec
    if direction.name == f"{hidden_name}y":
      return input2_factor * direction.value * out_vec
    return output_space.null_vector()

  second_layer = vectorspace_fns.Linear.from_action(hidden_space, output_space,
                                                    second_layer_action)

  return transformers.MLP(first_layer, second_layer)
```


# Aggregate (Select)

```
def categorical_attn(
    query_space: bases.VectorSpaceWithBasis,
    key_space: bases.VectorSpaceWithBasis,
    value_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    bos_space: bases.VectorSpaceWithBasis,
    one_space: bases.VectorSpaceWithBasis,
    attn_fn: QueryKeyToAttnLogit,
    default_output: Optional[bases.VectorInBasis] = None,
    causal: bool = False,
    always_attend_to_bos: bool = False,
    use_bos_for_default_output: bool = True,
    softmax_coldness: float = 100.,
) -> transformers.AttentionHead:
  """Returns an attention head for categorical inputs.

  Assumes the existence of a beginning of sequence token and attends to it
  always with strength 0.5*softmax_coldness. This allows to implement an
  arbitrary default value for rows in the attention pattern that are all-zero.

  Attends to the BOS token if all other key-query pairs have zero attention.
  Hence, the first value in the value sequence will be the default output for
  such cases.

  Args:
    query_space: Vector space containing (categorical) query input.
    key_space: Vector space containing (categorical) key input.
    value_space: Vector space containing (numerical) value input.
    output_space: Vector space which will contain (numerical) output.
    bos_space: 1-d space used to identify the beginning of sequence token.
    one_space: 1-d space which contains 1 at every position.
    attn_fn: A selector function f(query, key) operating on the query/key basis
      directions that defines the attention pattern.
    default_output: Output to return if attention pattern is all zero.
    causal: If True, use masked attention.
    always_attend_to_bos: If True, always attend to the BOS token. If False,
      only attend to BOS when attending to nothing else.
    use_bos_for_default_output: If True, assume BOS is not in the value space
      and output a default value when attending to BOS. If False, assume BOS is
      in the value space, and map it to the output space like any other token.
    softmax_coldness: The inverse temperature of the softmax. Default value is
      high which makes the attention close to a hard maximum.
  """
  bases.ensure_dims(bos_space, num_dims=1, name="bos_space")
  bases.ensure_dims(one_space, num_dims=1, name="one_space")
  bos_direction = bos_space.basis[0]
  one_direction = one_space.basis[0]

  # Add bos direction to query, key, and value spaces in case it is missing
  query_space = bases.join_vector_spaces(query_space, bos_space, one_space)
  key_space = bases.join_vector_spaces(key_space, bos_space)
  value_space = bases.join_vector_spaces(value_space, bos_space)

  if always_attend_to_bos:
    value_basis = value_space.basis
  else:
    value_basis = [v for v in value_space.basis if v != bos_direction]
  assert len(value_basis) == output_space.num_dims
  value_to_output = dict(zip(value_basis, output_space.basis))

  if default_output is None:
    default_output = output_space.null_vector()
  assert default_output in output_space

  def qk_fun(query: bases.BasisDirection, key: bases.BasisDirection) -> float:

    # We want to enforce the following property on our attention patterns:
    # - if nothing else is attended to, attend to the BOS token.
    # - otherwise, don't attend to the BOS token.
    #
    # We assume that the BOS position always only contains the vector bos + one,
    # and that any other position has bos coefficient 0.
    #
    # We do this as follows:
    # Let Q and K be subspaces of V containing the query and key vectors,
    # both disjoint with the BOS space {bos} or the one space {one}.
    # Suppose we have an attn_fn which defines a bilinear W_QK: V x V -> ℝ,
    # s.t. W_QK(q, k) = 0 whenever either q or k are bos or one.
    #
    # Then define W_new: V x V -> ℝ st:
    # W_new(one, bos) = 0.5, otherwise 0.
    #
    # Now set W_QK' = W_QK + W_new.
    #
    # To evaluate the attention to the BOS position:
    # W_QK'(q, bos + one)
    # = W_QK'(q, bos) + W_QK'(q, one)
    # = W_QK(q, bos) + W_QK(q, one) + W_new(q, bos) + W_new(q, one)
    # = 0            + 0            + W_new(q, bos) + W_new(q, one)
    # = W_new(q, bos) + W_new(q, one)
    # = W_new(q' + one, bos) + W_new(q' + one, one)  where q = one + q'
    # = W_new(q', bos) + W_new(one, bos) + W_new(q', one) + W_new(one, one)
    # = 0              + 0.5             + 0              + 0
    # = 0.5
    #
    # To evaluate the attention to a non-BOS position:
    # W_QK'(0 * bos + q, 0 * bos + k)  # s.t. q ∈ Q+{one}, k ∈ K+{one}
    # = 0*W_QK'(bos, 0*bos + k) + W_QK'(q, 0*bos + k)
    # = W_QK'(q, 0*bos + k)
    # = 0*W_QK'(q, bos) + W_QK'(q, k)
    # = W_QK'(q, k)
    # = W_QK(q, k)    since W_QK' = W_QK on inputs not containing bos.
    # = W_QK(q', k')  since W_QK(x, y) = 0 whenever x or y are one.
    #
    # Since W_QK(q, k) takes values in 0, 1, a sufficiently high softmax
    # coldness will give us the desired property.                            QED
    #
    # The following implements this idea.
    # By replacing 0.5 with 1, we can instead enforce a different property: that
    # the BOS token is always attended to in addition to whatever else.

    if key == bos_direction and query == one_direction:
      c = 1. if always_attend_to_bos else 0.5
      return c * softmax_coldness
    elif {key, query}.intersection({one_direction, bos_direction}):
      return 0

    return softmax_coldness * attn_fn(query, key)

  w_qk = vectorspace_fns.ScalarBilinear.from_action(
      query_space,
      key_space,
      qk_fun,
  )

  def ov_fun(input_dir: bases.BasisDirection) -> bases.VectorInBasis:
    if use_bos_for_default_output and input_dir == bos_direction:
      return default_output
    return output_space.vector_from_basis_direction(value_to_output[input_dir])

  w_ov = vectorspace_fns.Linear.from_action(
      value_space,
      output_space,
      ov_fun,
  )

  return transformers.AttentionHead(w_qk, w_ov, causal=causal)
  ```






# Test results
Extra map
pred LAM_ADD vs LAM_MUL


{'snd': 13, 'w_qk': 7, 'w_ov': 7, 'fst': 3, '': 1}
56.6%
AND not OR
missed double select
missed double sel width
but spotted the ending aggregates

{'w_qk': 4, 'w_ov': 4, 'fst': 2, 'snd': 2}
84%
mixed up a aggregate->map for a sequence map aggregate
all lambdas wrong
Mul not and
ge not gt - VERY SIMILAR

{'fst': 4, 'snd': 4, 'w_qk': 1, 'w_ov': 1}
52%
missed a 3rd Map
otherwise perfect


{'fst': 4, 'snd': 4, 'w_qk': 1, 'w_ov': 1}
52%
missed 2 extra maps from 4
erroneous map after a sequence map


{'w_qk': 11, 'w_ov': 11, 'fst': 5, 'snd': 10}
54%
pred select sequence map instead of map select
missed double selector withd with same vars
select not agregate

{'w_qk': 4, 'w_ov': 4, 'fst': 2, 'snd': 4}
70%
seelct not map
miss pred double map
miss pred tripple agg


92%
57%
94%
56%
71%
70%
70%
72%
50%
95%
64%
90%
54%
85%
60%
70%
71%
80%
80%
64%
81%
69%
65%
74%
86%
68%
64%
70%
80%
76%
94%
82%
66%
82%
72%
78%
60%
49%
94%
58%
90%
84%
89%
80%
60%
68%
66%
71%
90%
