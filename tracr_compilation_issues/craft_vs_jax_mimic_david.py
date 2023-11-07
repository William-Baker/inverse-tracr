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
import itertools
import pandas as pd
from tracr.compiler import nodes
jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


_BOS_DIRECTION = "rasp_to_transformer_integration_test_BOS"
_ONE_DIRECTION = "rasp_to_craft_integration_test_ONE"
_COMPILER_PAD = "compiler_pad"

def make_input_space(vocab, max_seq_len):
  tokens_space = bases.VectorSpaceWithBasis.from_values("tokens", vocab)
  indices_space = bases.VectorSpaceWithBasis.from_values(
      "indices", range(max_seq_len))
  one_space = bases.VectorSpaceWithBasis.from_names([_ONE_DIRECTION])
  bos_space = bases.VectorSpaceWithBasis.from_names([_BOS_DIRECTION])
  input_space = bases.join_vector_spaces(tokens_space, indices_space, one_space,
                                         bos_space)

  return input_space

def embed_input(input_seq, input_space):
  bos_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_BOS_DIRECTION))
  one_vec = input_space.vector_from_basis_direction(
      bases.BasisDirection(_ONE_DIRECTION))
  embedded_input = [bos_vec + one_vec]
  for i, val in enumerate(input_seq):
    i_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("indices", i))
    val_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection("tokens", val))
    embedded_input.append(i_vec + val_vec + one_vec)
  return bases.VectorInBasis.stack(embedded_input)

def vocab_to_lang(vocab, max_seq_len):
    return sum([list(itertools.product(vocab, repeat=l-1)) for l in range(1, max_seq_len+1)], [])
    

def prog_a():
    # vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
    # max_seq_len = 4 # maybe 5
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        v1 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x > y)
        v2 = rasp.Map(lambda y: vocab[0] and y, rasp.indices)
        v3 = rasp.SelectorWidth(v1)
        v4 = rasp.Select(v2, v3, lambda x, y: x < y)
        v5 = rasp.Aggregate(v4, v3)
        return v5
    return rasp_prog(), vocab, max_seq_len, language

def prog_b():
    # vocab = ['t0', 't1', 't2', 't3', 't4', 't5', 't6']
    # max_seq_len = 8
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y)
        so2 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x < y)
        so3 = rasp.SelectorWidth(se2)
        se3 = rasp.Select(so3, so2, lambda x, y: x!=y)
        so6 = rasp.SequenceMap(lambda x,y: x-y, so3, so3)
        so7 = rasp.Aggregate(se3, so6)
        return so7
    return rasp_prog(), vocab, max_seq_len, language


def prog_c() -> rasp.SOp:
    """
    sum_of_inputs
    """
    vocab = [1,2,3]
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        x = rasp.numerical(rasp.tokens)
        before = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
        means = rasp.Aggregate(before, rasp.tokens)  # returns sequence s_i = mean_{j<=i} input_j
        sums = rasp.SequenceMap(lambda x, y: x*y, means, rasp.indices+1)
        return sums
    return rasp_prog(), vocab, max_seq_len, language

def prog_d():
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        so3 = rasp.SequenceMap(lambda x,y: x and y, rasp.indices, rasp.indices)
        se1 = rasp.Select(rasp.tokens, rasp.tokens, lambda x,y: x <= y)
        so4 = rasp.Aggregate(se1, so3)
        so5 = rasp.Map(lambda x: x and 4, so4)
        so9 = rasp.SequenceMap(lambda x,y: x-y, so5, so5) # makes the program pointless
        return so9
    return rasp_prog(), vocab, max_seq_len, language

def prog_e():
    #  Select'>, inputs=['tokens', 'tokens', <Comparison.GT: '>'>], output="<class 'Selector'> 1", lambda_name=None),
    #  SelectorWidth'>, inputs=["<class 'Selector'> 1"], output="<class 'SOp'> 5", lambda_name=None),
    #  Map'>, inputs=[functools.partial(<function <lambda> at 0x2affe42f6560>, 4), "<class 'SOp'> 5"], output="<class 'SOp'> 7", lambda_name='LAM_GE'),
    #  Select'>, inputs=["<class 'SOp'> 7", "<class 'SOp'> 7", <Comparison.TRUE: 'True'>], output="<class 'Selector'> 3", lambda_name=None),
    #  Aggregate'>, inputs=["<class 'Selector'> 3", "<class 'SOp'> 5"], output="<class 'SOp'> 9", lambda_name=None)
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GT)
        so5 = rasp.SelectorWidth(se1)
        so7 = rasp.Map(lambda x: x >= 2, so5)
        se3 = rasp.Select(so7, so7, rasp.Comparison.TRUE)
        so9 = rasp.Aggregate(se3, so5)
        return so9
    return rasp_prog(), vocab, max_seq_len, language

def prog_f():
    # Map'>, inputs=function 4), 'indices'], output="<class 'SOp'> 2", lambda_name='LAM_SUB')
    # Map'>, inputs=function 3), "<class 'SOp'> 2"], output="<class 'SOp'> 3", lambda_name='LAM_EQ')
    # Map'>, inputs=function 't2'), 'tokens'], output="<class 'SOp'> 1", lambda_name='LAM_LT')
    # Select'>, inputs=[SOp'> 1", "<class 'SOp'> 3", <Comparison.GT: '>'>], out=.Selector'> 1", lambda_name=None)
    # SelectorWidth'>, inputs=["Selector'> 1"], output="<class 'SOp'> 7", lambda_name=None)
    # SequenceMap'>, inputs=[<func "<class 'SOp'> 7", "<class 'SOp'> 7"], output="SOp'> 8", lambda_name='LAM_ADD')
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        so2 = rasp.Map(lambda x: x - 1, rasp.indices)
        so3 = rasp.Map(lambda x: x == 1, so2)
        so1 = rasp.Map(lambda x: x < "b", rasp.tokens)
        se1 = rasp.Select(so1, so3, rasp.Comparison.GT)
        so7 = rasp.SelectorWidth(se1)
        so8 = rasp.SequenceMap(lambda x,y: x + y, so7, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language

def prog_g():
    # Map'>, inputs=[f  4.28), "<class 'SOp'> 4"], output="<class 'SOp'> 5", lambda_name='LAM_SUB')
    # Select'>, inputs=['tokens', 'tokens', <Comparison.TRUE: 'True'>], output="<class 'Selector'> 1", lambda_name=None)
    # Aggregate'>, inputs=["<class 'Selector'> 1", 'indices'], output="<class 'SOp'> 4", lambda_name=None)
    # Select'>, inputs=["<class 'SOp'> 4", "<class 'SOp'> 4", <Comparison.FALSE: 'False'>], output="<class 'Selector'> 2", lambda_name=None)
    # Aggregate'>, inputs=["<class 'Selector'> 2", "<class 'SOp'> 5"], output="<class 'SOp'> 6", lambda_name=None)
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2af496ed0ee0>, 4), "<class 'SOp'> 6"], output="<class 'SOp'> 10", lambda_name='LAM_ADD')
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        so4 = rasp.Aggregate(se1, rasp.indices)
        so5 = rasp.Map(lambda x: x - 4.28, so4)
        se2 = rasp.Select(so4, so4, rasp.Comparison.FALSE)
        so6 = rasp.Aggregate(se2, so5)
        so10 = rasp.Map(lambda x: x + 4, so6)
        return so10
    return rasp_prog(), vocab, max_seq_len, language

def prog_h():
    #'Select'>, inputs=['tokens', 'tokens', <Comparison.GEQ: '>='>], output="<class 'Selector'> 1", lambda_name=None)
    #'SelectorWidth'>, inputs=["<class 'Selector'> 1"], output="<class 'SOp'> 6", lambda_name=None)
    #'Map'>, inputs=[functools.partial(<function <lambda> at 0x2b29b36ac9d0>, 4), "<class 'SOp'> 6"], output="<class 'SOp'> 7", lambda_name='LAM_AND')
    #'SequenceMap'>, inputs=[<function <lambda> at 0x2b29b36ac940>, "<class 'SOp'> 7", "<class 'SOp'> 6"], output="<class 'SOp'> 8", lambda_name='LAM_OR')
    #'SequenceMap'>, inputs=[<function <lambda> at 0x2b29b36ac940>, "<class 'SOp'> 8", "<class 'SOp'> 8"], output="<class 'SOp'> 9", lambda_name='LAM_OR'
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.GEQ)
        so6 = rasp.SelectorWidth(se1)
        so7 = rasp.Map(lambda x: x and 2, so6)
        so8 = rasp.SequenceMap(lambda x, y: x or y, so7, so6)
        so9 = rasp.SequenceMap(lambda x, y: x or y, so8, so8)
        return so9
    return rasp_prog(), vocab, max_seq_len, language

def prog_i():
    #  'Aggregate'>, inputs=["<class 'Selector'> 1", 'indices'], output="<class 'SOp'> 3", lambda_name=None)
    #  'Select'>, inputs=["<class 'SOp'> 4", "<class 'SOp'> 3", <Comparison.LT: '<'>], output="<class 'Selector'> 2", lambda_name=None)
    #  'SelectorWidth'>, inputs=["<class 'Selector'> 2"], output="<class 'SOp'> 7", lambda_name=None)
    #  'Select'>, inputs=['tokens', 'tokens', <Comparison.NEQ: '!='>], output="<class 'Selector'> 1", lambda_name=None)
    #  'SelectorWidth'>, inputs=["<class 'Selector'> 1"], output="<class 'SOp'> 4", lambda_name=None)
    #  'SequenceMap'>, inputs=[<function <lambda> at 0x2b2b8a960430>, "<class 'SOp'> 4", "<class 'SOp'> 7"], output="<class 'SOp'> 8", lambda_name='LAM_OR')
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
        so4 = rasp.SelectorWidth(se1)
        so3 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(so4, so3, rasp.Comparison.LT)
        so7 = rasp.SelectorWidth(se2)
        so8 = rasp.SequenceMap(lambda x,y: x or y, so4, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language

def prog_i_n():
    #  'Aggregate'>, inputs=["<class 'Selector'> 1", 'indices'], output="<class 'SOp'> 3", lambda_name=None)
    #  'Select'>, inputs=["<class 'SOp'> 4", "<class 'SOp'> 3", <Comparison.LT: '<'>], output="<class 'Selector'> 2", lambda_name=None)
    #  'SelectorWidth'>, inputs=["<class 'Selector'> 2"], output="<class 'SOp'> 7", lambda_name=None)
    #  'Select'>, inputs=['tokens', 'tokens', <Comparison.NEQ: '!='>], output="<class 'Selector'> 1", lambda_name=None)
    #  'SelectorWidth'>, inputs=["<class 'Selector'> 1"], output="<class 'SOp'> 4", lambda_name=None)
    #  'SequenceMap'>, inputs=[<function <lambda> at 0x2b2b8a960430>, "<class 'SOp'> 4", "<class 'SOp'> 7"], output="<class 'SOp'> 8", lambda_name='LAM_OR')
    vocab = [1, 2, 3]
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
        so4 = rasp.SelectorWidth(se1)
        so3 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(so4, so3, rasp.Comparison.LT)
        so7 = rasp.SelectorWidth(se2)
        so8 = rasp.SequenceMap(lambda x,y: x or y, so4, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language
   
def prog_j():
    #'Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8aa0c8b0>, 3), 'indices'], output="<class 'SOp'> 2", lambda_name='LAM_SUB')
    #'SequenceMap'>, inputs=[<function <lambda> at 0x2b2b8aa0c820>, "<class 'SOp'> 2", 'indices'], output="<class 'SOp'> 4", lambda_name='LAM_OR')
    #'Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8aa0c700>, 't2'), 'tokens'], output="<class 'SOp'> 1", lambda_name='LAM_GT')
    #'Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8aa0c670>, False), "<class 'SOp'> 1"], output="<class 'SOp'> 3", lambda_name='LAM_LT')
    #'Select'>, inputs=["<class 'SOp'> 3", "<class 'SOp'> 3", <Comparison.LEQ: '<='>], output="<class 'Selector'> 1", lambda_name=None)
    #'Aggregate'>, inputs=["<class 'Selector'> 1", "<class 'SOp'> 4"], output="<class 'SOp'> 6", lambda_name=None) 
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        so2 = rasp.Map(lambda x: x - 2, rasp.indices)
        so4 = rasp.SequenceMap(lambda x,y: x or y, so2, rasp.indices)
        so1 = rasp.Map(lambda x: x > 'b', rasp.tokens)
        so3 = rasp.Map(lambda x: x < False, so1)
        se1 = rasp.Select(so3, so3, rasp.Comparison.LEQ)
        so6 = rasp.Aggregate(se1, so4)
        return so6
    return rasp_prog(), vocab, max_seq_len, language        

def prog_k():
    # SelectorWidth'>, inputs=["<class Selector'> 3"], output="<class SOp'> 6", lambda_name=None)
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b29b30f8940>, 3), "<class SOp'> 1"], output="<class SOp'> 3", lambda_name='LAM_MUL')
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b29b30f88b0>, 4.11), 'indices'], output="<class SOp'> 1", lambda_name='LAM_ADD')
    # Select'>, inputs=["<class SOp'> 1", 'indices', <Comparison.NEQ: '!='>], output="<class Selector'> 3", lambda_name=None)
    # Aggregate'>, inputs=["<class Selector'> 3", "<class SOp'> 3"], output="<class SOp'> 5", lambda_name=None)
    # SequenceMap'>, inputs=[<function <lambda> at 0x2b29b30f88b0>, "<class SOp'> 5", "<class SOp'> 6"], output="<class SOp'> 8", lambda_name='LAM_ADD')
    vocab = [1,2, 3]
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        so1 = rasp.Map(lambda x: x + 4.11, rasp.tokens)
        se3 = rasp.Select(so1, rasp.indices, rasp.Comparison.NEQ)
        so6 = rasp.SelectorWidth(se3)
        so3 = rasp.Map(lambda x: x * 3, so1)
        so5 = rasp.Aggregate(se3, so3)
        so8 = rasp.SequenceMap(lambda x, y: x + y, so5, so6)
        return so8
    return rasp_prog(), vocab, max_seq_len, language        

def prog_l():
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8ab101f0>, True), "<class SOp'> 1"], output="<class SOp'> 3", lambda_name='LAM_IV')
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8ab100d0>, 't0'), 'tokens'], output="<class SOp'> 1", lambda_name='LAM_LT')
    # Select'>, inputs=["<class SOp'> 1", "<class SOp'> 3", <Comparison.EQ: '=='>], output="<class Selector'> 1", lambda_name=None)
    # SelectorWidth'>, inputs=["<class Selector'> 1"], output="<class SOp'> 10", lambda_name=None)
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8ab10040>, 4), "<class SOp'> 10"], output="<class SOp'> 11", lambda_name='LAM_OR')
    # Map'>, inputs=[functools.partial(<function <lambda> at 0x2b2b8aac7f40>, 3.29), "<class SOp'> 11"], output="<class SOp'> 13", lambda_name='LAM_MUL')
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        so1 = rasp.Map(lambda x: x < 'a', rasp.tokens)
        so3 = rasp.Map(lambda x: not x, so1)
        se1 = rasp.Select(so1, so3, rasp.Comparison.EQ)
        so10 = rasp.SelectorWidth(se1)
        so11 = rasp.Map(lambda x: x or 4, so10)
        so13 = rasp.Map(lambda x: x * 3.29, so11)
        return so13
    return rasp_prog(), vocab, max_seq_len, language

def prog_m():
    # key is none?
    #  Map'>, inputs=[functools.partial(<function <lambda> at 0x2b29b30f8a60>, 4), "<class SOp'> 1"], output="<class SOp'> 4", lambda_name='LAM_AND')
    #  Map'>, inputs=[functools.partial(<function <lambda> at 0x2b29b30f8af0>, 4.7), "<class SOp'> 4"], output="<class SOp'> 5", lambda_name='LAM_OR')
    #  SelectorWidth'>, inputs=["<class Selector'> 1"], output="<class SOp'> 1", lambda_name=None)
    #  Select'>, inputs=['tokens', 'tokens', <Comparison.NEQ: '!='>], output="<class Selector'> 1", lambda_name=None)
    #  Aggregate'>, inputs=["<class Selector'> 1", "<class SOp'> 1"], output="<class SOp'> 3", lambda_name=None)
    #  Select'>, inputs=["<class SOp'> 3", "<class SOp'> 5", <Comparison.FALSE: 'False'>], output="<class Selector'> 2", lambda_name=None)
    #  SelectorWidth'>, inputs=["<class Selector'> 2"], output="<class SOp'> 7", lambda_name=None)
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)
    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
        so1 = rasp.SelectorWidth(se1)
        so4 = rasp.Map(lambda x: x and 4, so1)
        so5 = rasp.Map(lambda x: x or 4.7, so4)
        so3 = rasp.Aggregate(se1, so1)
        se2 = rasp.Select(so3, so5, rasp.Comparison.FALSE)
        so7 = rasp.SelectorWidth(se2)
        return so7
    
    return rasp_prog(), vocab, max_seq_len, language
        
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
    # vocab = {"a", "b", "c", "d"}
    # program = lib.make_hist()
    # input_seq = "abccd"
    vocab = {"h", "e", "l", "o"}
    program = lib.make_hist()
    input_seq = "hello"
  elif program_name == "sort_freq":
    vocab = {"a", "b", "c", "d"}
    program = lib.make_sort_freq(max_seq_len=max_seq_len)
    input_seq = "abcaba"
  elif program_name == "pair_balance":
    vocab = {"(", ")"}
    program = lib.make_pair_balance(
        sop=rasp.tokens, open_token="(", close_token=")")
    input_seq = "(()()"
  else:
    raise NotImplementedError(f"Program {program_name} not implemented.")
  language = vocab_to_lang(vocab, max_seq_len)
  return program, vocab, max_seq_len, language


def compile_rasp_to_model_returns_all(
    program: rasp.SOp,
    vocab: Set[rasp.Value],
    max_seq_len: int,
    causal: bool = False,
    mlp_exactness: int = 100) -> assemble.AssembledTransformerModel:

    if _BOS_DIRECTION in vocab:
        raise ValueError("Compiler BOS token must not be present in the vocab. "
                        f"Found '{_BOS_DIRECTION}' in {vocab}")

    if _COMPILER_PAD in vocab:
        raise ValueError("Compiler PAD token must not be present in the vocab. "
                        f"Found '{_COMPILER_PAD}' in {vocab}")

    #   rasp_model = rasp_to_graph.extract_rasp_graph(program)
    #   graph, sources, sink = rasp_model.graph, rasp_model.sources, rasp_model.sink

    #   basis_inference.infer_bases(
    #       graph,
    #       sink,
    #       vocab,
    #       max_seq_len,
    #   )

    #   expr_to_craft_graph.add_craft_components_to_rasp_graph(
    #       graph,
    #       bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
    #       mlp_exactness=mlp_exactness,
    #   )

    #   craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)
    categorical_output = rasp.is_categorical(program)
    rasp_model = rasp_to_graph.extract_rasp_graph(program)
    basis_inference.infer_bases(
        rasp_model.graph,
        rasp_model.sink,
        vocab,
        max_seq_len=max_seq_len,
    )
    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        rasp_model.graph,
        bos_dir=bases.BasisDirection(_BOS_DIRECTION),
        one_dir=bases.BasisDirection(_ONE_DIRECTION),
    )
    craft_model = craft_graph_to_model.craft_graph_to_model(rasp_model.graph,
                                                        rasp_model.sources)
    input_space = make_input_space(vocab, max_seq_len)
    output_space = bases.VectorSpaceWithBasis(
        rasp_model.sink[nodes.OUTPUT_BASIS])
    if not categorical_output:
            assert len(output_space.basis) == 1

    jax_model = craft_model_to_transformer.craft_model_to_transformer(
        craft_model=craft_model,
        graph=rasp_model.graph,
        sink=rasp_model.sink,
        max_seq_len=max_seq_len,
        causal=causal,
        compiler_bos=_BOS_DIRECTION,
        compiler_pad=_COMPILER_PAD,
    )
    
    return jax_model, rasp_model, craft_model, input_space, output_space

def test_program(rasp_prog, vocab, max_seq_len, language, prog_name: str):
    jax_model, rasp_model, craft_model, input_space, output_space = compile_rasp_to_model_returns_all(
        rasp_prog, set(vocab), max_seq_len)
    
    # _ONE_DIRECTION = 'one'
    # _BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if (basis.value == 'compiler_bos')][0]
    
    df_rows = []
    
    for inp in language:
        # CRAFT forward pass
        test_input_vector = embed_input(list(inp), input_space)
        output_seq = craft_model.apply(test_input_vector).project(output_space)
        
        output_space = bases.VectorSpaceWithBasis(rasp_model.sink[nodes.OUTPUT_BASIS])

        def decode_outs(output_seq, output_space):
            outs = output_seq.project(output_space) # sparse outs
            labels = outs.magnitudes.argmax(axis=1)
            return [output_space.basis[i].value for i in labels]

        craft_outputs = decode_outs(output_seq, output_space)
        
        
        
        formatted_input = [_BOS_DIRECTION] + list(inp)
        
        # Jax forward pass
        output = jax_model.apply(formatted_input)
        jax_output = output.decoded
        
        
        # rasp forward pass
        try:
            rasp_out = rasp_prog(list(inp))
        except Exception as E:
            rasp_out = f"RASP_FAILED:{E}"
        
        
        
        
        # embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)
        # output_seq = craft_model.apply(embedded_input)

        
        
        
        

        df_rows.append(dict(prog_name=prog_name, inp=inp, 
                            rasp=rasp_out, 
                            craft=craft_outputs, 
                            jax=jax_output))

        from inverse_tracr.utils.verbose_craft import plot_basis_dir
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        outs = output_seq.project(output_space) # sparse outs
        plot_basis_dir(axs, outs, "")
    return pd.DataFrame(df_rows)


progs = {
    "prog_a": prog_a,
    # "prog_b": prog_b, # compiler error jax conversion qk matrix
    "prog_c": prog_c, # calls aggregate on numeric tokens - bos throws exception
    "prog_d": prog_d,
    "prog_e": prog_e,
    "prog_f": prog_f,
    "prog_g": prog_g,
    "prog_h": prog_h,
    "prog_i": prog_i, # query is None!
    "prog_i_n": prog_i_n,
    # "prog_j": prog_j,  # compiler error jax conversion qk matrix
    "prog_k": prog_k, # calls aggregate on numeric tokens - bos throws exception
    "prog_m": prog_m,
    
}

rasp_prog, vocab, max_seq_len, language = get_program("sort_unique", 4)

from functools import partial
ex_progs = ["length", "frac_prevs", "dyck-2", "dyck-3", "sort", "sort_unique", "hist", "sort_freq", "pair_balance"]
for ex in ex_progs:
    progs[ex] = partial(get_program, ex, 4)


# master_df = []
# for prog_name, prog in progs.items():
#     print(prog_name)
#     rasp_prog, vocab, max_seq_len, language = prog()
#     df = test_program(rasp_prog, vocab, max_seq_len, language, prog_name)
#     master_df.append(df)
# master_df = pd.concat(master_df)
# master_df.to_csv('craft_vs_jax_v2.csv')


#%%

rasp_prog, vocab, max_seq_len, language = progs['prog_e']()

inp = list('aab')


jax_model, rasp_model, craft_model, input_space, output_space = compile_rasp_to_model_returns_all(
        rasp_prog, set(vocab), max_seq_len)


from inverse_tracr.utils.verbose_craft import make_craft_model_verbose

make_craft_model_verbose(craft_model)

test_input_vector = embed_input(list(inp), input_space)
output_seq = craft_model.apply(test_input_vector).project(output_space)

output_space = bases.VectorSpaceWithBasis(rasp_model.sink[nodes.OUTPUT_BASIS])

def decode_outs(output_seq, output_space):
    outs = output_seq.project(output_space) # sparse outs
    labels = outs.magnitudes.argmax(axis=1)
    return [output_space.basis[i].value for i in labels]

craft_outputs = decode_outs(output_seq, output_space)

        

rasp_out = rasp_prog(list(inp))
# %%


[0.333333, 0.666667, 0.      , 0.      , 0.      , 0.      ],
[0.333333, 0.666667, 0.      , 0.      , 0.      , 0.      ],
[0.333333, 0.666667, 0.      , 0.      , 0.      , 0.      ]])