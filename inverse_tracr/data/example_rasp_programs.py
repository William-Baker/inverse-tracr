
import sys
sys.path.append('tracr/')
import tracr.compiler.lib as lib
from tracr.rasp import rasp

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
        vocab = {1, 2, 3, 4, 5}
        program = rasp.Map(lambda x: x > 4, rasp.tokens)
        input_seq = [1, 2]
    elif program_name == "map_test_b":
        vocab = {1, 2, 3, 4, 5}
        program = rasp.Map(lambda x: x < 1, rasp.Map(
            lambda x: x > 1, rasp.tokens))
        input_seq = [1, 2]
    elif program_name == "map_test_c":
        vocab = {1, 2, 3, 4, 5}
        input_seq = [1, 2]

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

