# %%

import tracr.compiler.lib as lib
from functools import partial
from typing import Set
from tracr.compiler import assemble
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp


import jax
import os
from tracr.rasp import rasp
import itertools
import pandas as pd
from tracr.compiler import nodes
jax.config.update('jax_platform_name', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"
COMPILER_ONE = "one"


def _make_input_space(vocab, max_seq_len):
    tokens_space = bases.VectorSpaceWithBasis.from_values("tokens", vocab)
    indices_space = bases.VectorSpaceWithBasis.from_values(
        "indices", range(max_seq_len))
    one_space = bases.VectorSpaceWithBasis.from_names([COMPILER_ONE])
    bos_space = bases.VectorSpaceWithBasis.from_names([COMPILER_BOS])
    input_space = bases.join_vector_spaces(tokens_space, indices_space, one_space,
                                           bos_space)

    return input_space


def _embed_input(input_seq, input_space):
    bos_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection(COMPILER_BOS))
    one_vec = input_space.vector_from_basis_direction(
        bases.BasisDirection(COMPILER_ONE))
    embedded_input = [bos_vec + one_vec]
    for i, val in enumerate(input_seq):
        i_vec = input_space.vector_from_basis_direction(
            bases.BasisDirection("indices", i))
        val_vec = input_space.vector_from_basis_direction(
            bases.BasisDirection("tokens", val))
        embedded_input.append(i_vec + val_vec + one_vec)
    return bases.VectorInBasis.stack(embedded_input)


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
        #bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
        bos_dir=bases.BasisDirection(compiler_bos),
        one_dir=bases.BasisDirection(compiler_pad),
        mlp_exactness=mlp_exactness,
    )

    craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)
    

    input_space = _make_input_space(vocab, max_seq_len)
    output_space = bases.VectorSpaceWithBasis(
        rasp_model.sink[nodes.OUTPUT_BASIS])

    return craft_model_to_transformer.craft_model_to_transformer(
        craft_model=craft_model,
        graph=graph,
        sink=sink,
        max_seq_len=max_seq_len,
        causal=causal,
        compiler_bos=compiler_bos,
        compiler_pad=compiler_pad,
        ), rasp_model, craft_model, input_space, output_space


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


def vocab_to_lang(vocab, max_seq_len):
    return sum([list(itertools.product(vocab, repeat=l-1)) for l in range(1, max_seq_len+1)], [])


def prog_a():
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
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        se1 = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y)
        so2 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x < y)
        so3 = rasp.SelectorWidth(se2)
        se3 = rasp.Select(so3, so2, lambda x, y: x != y)
        so6 = rasp.SequenceMap(lambda x, y: x-y, so3, so3)
        so7 = rasp.Aggregate(se3, so6)
        return so7
    return rasp_prog(), vocab, max_seq_len, language


def prog_c() -> rasp.SOp:
    vocab = [1, 2, 3]
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        x = rasp.numerical(rasp.tokens)
        before = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
        # returns sequence s_i = mean_{j<=i} input_j
        means = rasp.Aggregate(before, rasp.tokens)
        sums = rasp.SequenceMap(lambda x, y: x*y, means, rasp.indices+1)
        return sums
    return rasp_prog(), vocab, max_seq_len, language


def prog_d():
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        so3 = rasp.SequenceMap(lambda x, y: x and y,
                               rasp.indices, rasp.indices)
        se1 = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x <= y)
        so4 = rasp.Aggregate(se1, so3)
        so5 = rasp.Map(lambda x: x and 4, so4)
        # makes the program pointless
        so9 = rasp.SequenceMap(lambda x, y: x-y, so5, so5)
        return so9
    return rasp_prog(), vocab, max_seq_len, language


def prog_e():
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
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        so2 = rasp.Map(lambda x: x - 1, rasp.indices)
        so3 = rasp.Map(lambda x: x == 1, so2)
        so1 = rasp.Map(lambda x: x < "b", rasp.tokens)
        se1 = rasp.Select(so1, so3, rasp.Comparison.GT)
        so7 = rasp.SelectorWidth(se1)
        so8 = rasp.SequenceMap(lambda x, y: x + y, so7, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language


def prog_g():
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
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
        so4 = rasp.SelectorWidth(se1)
        so3 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(so4, so3, rasp.Comparison.LT)
        so7 = rasp.SelectorWidth(se2)
        so8 = rasp.SequenceMap(lambda x, y: x or y, so4, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language


def prog_i_n():
    vocab = [1, 2, 3]
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        se1 = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.NEQ)
        so4 = rasp.SelectorWidth(se1)
        so3 = rasp.Aggregate(se1, rasp.indices)
        se2 = rasp.Select(so4, so3, rasp.Comparison.LT)
        so7 = rasp.SelectorWidth(se2)
        so8 = rasp.SequenceMap(lambda x, y: x or y, so4, so7)
        return so8
    return rasp_prog(), vocab, max_seq_len, language


def prog_j():
    vocab = ['a', 'b', 'c']
    max_seq_len = 3+1
    language = vocab_to_lang(vocab, max_seq_len)

    def rasp_prog():
        so2 = rasp.Map(lambda x: x - 2, rasp.indices)
        so4 = rasp.SequenceMap(lambda x, y: x or y, so2, rasp.indices)
        so1 = rasp.Map(lambda x: x > 'b', rasp.tokens)
        so3 = rasp.Map(lambda x: x < False, so1)
        se1 = rasp.Select(so3, so3, rasp.Comparison.LEQ)
        so6 = rasp.Aggregate(se1, so4)
        return so6
    return rasp_prog(), vocab, max_seq_len, language


def prog_k():
    vocab = [1, 2, 3]
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
        input_seq = [3, 2, 3, 5, 2]
    elif program_name == "sort_unique":
        vocab = {1, 2, 3, 4, 5}
        program = lib.make_sort_unique(rasp.tokens, rasp.tokens)
        input_seq = [3, 2, 1, 5, 2]
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


def test_program(rasp_prog, vocab, max_seq_len, language, prog_name: str):
    assembled_model, rasp_model, craft_model, input_space, output_space = compile_rasp_to_model_returns_all(
        rasp_prog, set(vocab), max_seq_len, compiler_bos=COMPILER_BOS, compiler_pad=COMPILER_PAD)

    # indices_space = bases.VectorSpaceWithBasis.from_values(
    #     rasp.indices.label, range(max_seq_len))
    # input_space = bases.join_vector_spaces(indices_space, craft_model.residual_space)

    # _BOS_DIRECTION = [basis.name for basis in craft_model.residual_space.basis if (basis.value == 'compiler_bos')][0]

    df_rows = []

    for inp in language:
        formatted_input = [COMPILER_BOS] + list(inp)

        # Jax forward pass
        output = assembled_model.apply(formatted_input)
        jax_output = output.decoded

        # rasp forward pass
        try:
            rasp_out = rasp_prog(list(inp))
        except Exception as E:
            rasp_out = f"RASP_FAILED:{E}"

        # CRAFT forward pass
        # embedded_input = embed_input(formatted_input, input_space=input_space, _BOS_DIRECTION=_BOS_DIRECTION, _ONE_DIRECTION=_ONE_DIRECTION)
        embedded_input = _embed_input([formatted_input], input_space)
        output_seq = craft_model.apply(embedded_input)

        output_space = bases.VectorSpaceWithBasis(
            rasp_model.sink[nodes.OUTPUT_BASIS])

        def decode_outs(output_seq, output_space):
            outs = output_seq.project(output_space)  # sparse outs
            labels = outs.magnitudes.argmax(axis=1)
            return [output_space.basis[i].value for i in labels]

        craft_outputs = decode_outs(output_seq, output_space)

        df_rows.append(dict(prog_name=prog_name, inp=inp,
                            rasp=rasp_out,
                            craft=craft_outputs,
                            jax=jax_output))

        # from inverse_tracr.utils.verbose_craft import plot_basis_dir
        # fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        # plot_basis_dir(axs, outs, "")
    return pd.DataFrame(df_rows)


progs = {
    "prog_a": prog_a,
    "prog_b": prog_b,  # compiler error jax conversion qk matrix
    "prog_c": prog_c,
    "prog_d": prog_d,
    "prog_e": prog_e,
    "prog_f": prog_f,
    "prog_g": prog_g,
    "prog_h": prog_h,
    "prog_i": prog_i,  # query is None!
    "prog_i_n": prog_i_n,
    "prog_j": prog_j,  # compiler error jax conversion qk matrix
    "prog_k": prog_k,  # issue with aggregation types causes RASP to fail
    "prog_m": prog_m,

}

rasp_prog, vocab, max_seq_len, language = get_program("sort_unique", 4)

ex_progs = ["length", "frac_prevs", "dyck-2", "dyck-3",
            "sort", "sort_unique", "hist", "sort_freq", "pair_balance"]
for ex in ex_progs:
    progs[ex] = partial(get_program, ex, 4)


master_df = []
for prog_name, prog in progs.items():
    print(prog_name)
    rasp_prog, vocab, max_seq_len, language = prog()
    df = test_program(rasp_prog, vocab, max_seq_len, language, prog_name)
    master_df.append(df)
master_df = pd.concat(master_df)
master_df.to_csv('craft_vs_jax_wov.csv')
# %%
