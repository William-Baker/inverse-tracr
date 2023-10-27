
from tracr.craft import bases
from tracr.rasp import rasp

def make_input_space(vocab, max_seq_len, _ONE_DIRECTION, _BOS_DIRECTION):
  tokens_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label, vocab)
  indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
  one_space = bases.VectorSpaceWithBasis.from_names([_ONE_DIRECTION])
  bos_space = bases.VectorSpaceWithBasis.from_names([_BOS_DIRECTION])
  input_space = bases.join_vector_spaces(tokens_space, indices_space, one_space,
                                         bos_space)

  return input_space


# def embed_input(input_seq, input_space, _BOS_DIRECTION, _ONE_DIRECTION):
#   bos_vec = input_space.vector_from_basis_direction(
#       bases.BasisDirection(_BOS_DIRECTION))
#   one_vec = input_space.vector_from_basis_direction(
#       bases.BasisDirection(_ONE_DIRECTION))
#   embedded_input = [bos_vec + one_vec]
#   for i, val in enumerate(input_seq):
#     i_vec = input_space.vector_from_basis_direction(
#         bases.BasisDirection(rasp.indices.label, i))
#     val_vec = input_space.vector_from_basis_direction(
#         bases.BasisDirection(rasp.tokens.label, val))
#     embedded_input.append(i_vec + val_vec + one_vec)
#   return bases.VectorInBasis.stack(embedded_input)

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


def embed_output(output_seq, output_space, categorical_output):
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