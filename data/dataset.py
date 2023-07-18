import sys
sys.path.append('tracr/')

from typing import Union, TypeVar, Sequence, Callable, Optional
from random import choice, randint, choices
from functools import partial
from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp
from tracr.craft.transformers import MultiAttentionHead, MLP
from dataclasses import dataclass
from data.canonical_ordering import sort_program
from data.rasp_operators import *
import numpy as np
from data.sigterm import guard_timeout, TimeoutException
import inspect
from collections import defaultdict
from enum import Enum
import tracr.compiler.lib as lib
from tracr.rasp import rasp
class Cat(Enum):
    numeric = 1
    categoric = 2
    boolean = 3


class Scope:
    def __init__(self, vocabulary: Sequence[Union[str, int, bool]], numeric_range) -> None:    
        self.scope = dict()
        self.names = set()
        self.types = set()
        self.counter = defaultdict(lambda: 0)
        self.names_by_type = defaultdict(lambda: [])
        self.names_by_type_and_cat = defaultdict(lambda: defaultdict(lambda: []))
        self.type_cat = dict()
        self.sampling_weights = [1,1]
        if type(vocabulary[0]) == str:
            token_type = Cat.categoric
        else:
            token_type = Cat.numeric
        self.__add__(SOp, "tokens", token_type, weight=1)
        # self.__add__(SOp, "indices", Cat.numeric, weight=1)
        self.sampling_weights = self.sampling_weights[2:]
        self.numeric_range = numeric_range
        self.vocabulary = vocabulary

    def add(self, t: type, cat: Cat, weight: Optional[int] = None) -> None:
        self.counter[t] += 1
        return self.__add__(t, str(t) + ' ' + str(self.counter[t]), cat, weight)
    
    def __add__(self, t: type, name: str, cat: Cat, weight: Optional[int] = None) -> None:
        if name in self.names:
            raise ValueError(f'{name} is already present in the set')
        self.scope[name] = t
        self.type_cat[name] = cat
        self.names.add(name)
        self.types.add(t)
        # preferentially sample most recent ops
        if len(self.sampling_weights) > 1:
            new_weight = (self.sampling_weights[-1] * 2 - self.sampling_weights[-2] + 1)
        else:
            new_weight = self.sampling_weights[-1] + 1

        sample_weight = new_weight if weight is None else weight
        self.sampling_weights.append(sample_weight)
        self.names_by_type[t].append((name, sample_weight))
        self.names_by_type_and_cat[t][cat].append((name, sample_weight))
        
        return name
    
    def get_cat(self, name: str):
        return self.type_cat[name]
    
    def weighted_sample(lst):
        weights = [x[1] for x in lst]
        sample = choices(lst, weights, k=1)
        return sample[0][0]

    def pick_var(self, y: type): 
        return Scope.weighted_sample(self.names_by_type[y])
    
    def pick_var_cat(self, y: type, cat: Cat):
        return Scope.weighted_sample(self.names_by_type_and_cat[y][cat])
    
    def var_exists(self, desired_type: type):
        return len(self.names_by_type[desired_type]) > 0
    
    def var_exists_cat(self, desired_type: type, desired_cat: Cat):
        return len(self.names_by_type_and_cat[desired_type][desired_cat]) > 0
    
    def matching_var_exists(self, src, desired_type: type):
        """returns True if a variable with the desired type is in scope with category matching the source"""
        return self.var_exists_cat(desired_type, self.get_cat(src))
    
    def gen_const(self, target_cat: Cat):
        if target_cat == Cat.numeric:
            return randint(*self.numeric_range)
        elif target_cat == Cat.categoric:
            return choice(self.vocabulary)
        else:
            raise NotImplementedError()

@dataclass
class Operation:
    operator: type
    inputs: Sequence[Union[str, Callable, rasp.Predicate ]]
    output: str
    lambda_name: Optional[str] = None
    


# =================================== Program Sampling ====================================


def sample_function(scope: Scope, ops, df=RASP_OPS):
    if scope.var_exists(Selector):
        sampled = df.sample(weights=df.weight).iloc[0]
    else:
        sampled = RASP_OPS_NO_SELECTOR.sample(weights=RASP_OPS_NO_SELECTOR.weight).iloc[0]


    if sampled.cls == rasp.Map:
        # [lambda1, SOp],
        
        return_type = SOp
        

        if randint(0,1) == 0: # Boolean operators
            return_cat = Cat.boolean
            
            s1 = scope.pick_var(SOp)
            f1, lambda_name = choice(UNI_LAMBDAS)
            
            obj_cat = scope.get_cat(s1)
            y = None
            if s1 == "tokens":
                y = scope.gen_const(Cat.categoric)
            elif obj_cat == Cat.numeric:
                y = scope.gen_const(Cat.numeric)
            elif obj_cat == Cat.categoric:
                y = scope.gen_const(Cat.categoric)
            elif obj_cat == Cat.boolean:
                y = bool(randint(0,1))
            else:
                raise NotImplementedError()
            
            func = partial(f1, y)

        else: # linear operators
            return_cat = Cat.numeric
            s1 = scope.pick_var_cat(SOp, return_cat)
            # TODO we can guarentee that the operand is non-zero, so lets make it possible to do division
            f1, bad_vals, lambda_name = choice(SEQUENCE_LAMBDAS)# + [(lambda x, y: x / y,  [0])]) 
            if  randint(0,2) <= 1: # 2/3 of the time generate an int
                s2 = scope.gen_const(Cat.numeric)
            else: # occasionally generate a float - may have a different impl
                s2 = float(scope.gen_const(Cat.numeric)) + randint(0,100)/100

            while s2 in bad_vals: # prevent bad values, such as div 0
                if isinstance(s2, float):
                    s2 += 0.1
                else:
                    s2 += 1

            func = partial(f1, s2)

        # Allocate a variable to hold the return value
        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [func, s1], allocated_name, lambda_name=lambda_name)
        ops.append(op)

    elif sampled.cls == rasp.SequenceMap: # must have double the weight of Map
        # [SOp, SOpNumericValue, lambda2],
        # todo chance of const full selector/sop
        s1 = scope.pick_var_cat(SOp, Cat.numeric)
        s1_cat = scope.get_cat(s1)
        if scope.var_exists_cat(SOp, s1_cat): # s2 wil be an SOp
            s2 = scope.pick_var_cat(SOp, s1_cat)
        else: 
            print("dead end")
            return
            
        f1, bad, lambda_name = choice(SEQUENCE_LAMBDAS)

        return_type = SOp
        return_cat = s1_cat

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [f1, s1, s2], allocated_name, lambda_name)
        ops.append(op)

    elif sampled.cls == rasp.Select:
        # [SOp, SOp, rasp.Predicate],    
        # todo chance of const full selector/sop
        s1 = scope.pick_var(SOp)
        s2 = scope.pick_var_cat(SOp, scope.get_cat(s1))
        pred = choice(PREDICATES)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2, pred], allocated_name)
        ops.append(op)

    elif sampled.cls == rasp.Aggregate:
        # [Selector, SOp],
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)
        s2 = scope.pick_var_cat(SOp, Cat.numeric)

        return_type = SOp
        return_cat = Cat.numeric

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2], allocated_name)
        ops.append(op)


    elif sampled.cls == rasp.SelectorWidth:
        # [Selector],                     
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)

        return_type = SOp
        return_cat = Cat.numeric

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1], allocated_name)
        ops.append(op)


    elif (sampled.cls == rasp.SelectorOr) or (sampled.cls == rasp.SelectorAnd):
        # [Selector, Selector],           
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)
        s2 = scope.pick_var(Selector)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2], allocated_name)
        ops.append(op)

    elif sampled.cls == rasp.SelectorNot:
        s1 = scope.pick_var(Selector)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1], allocated_name)
        ops.append(op)

    else:
        raise NotImplementedError()


def generate_ops(max_ops: int, vocab: Sequence, numeric_range: tuple):
    scope = Scope(vocab, numeric_range)
    ops = []

    sample_function(scope, ops, RASP_OPS )
    scope.__add__(SOp, "indices", Cat.numeric, weight=1) 
    for i in range(0, max_ops-2):
        sample_function(scope, ops, RASP_OPS )
    sample_function(scope, ops, RASP_OPS_RETURNS_SOP)

    return ops
import networkx as nx
def compile_program(ops):
    op_names = dict([(op.output, idx+2) for idx, op in enumerate(ops)] + [('tokens', 0), ('indices', 1)])

    G = nx.DiGraph()
    for idx, op in enumerate(ops):
        idx += 2
        for inp in op.inputs:
            if isinstance(inp, str):
                G.add_edge(op_names[inp], idx)
    if 1 in G:
        G.remove_node(op_names['indices']) # we want the longest path to start from the input tokens
    longest_path = nx.dag_longest_path(G)
    terminal_node = longest_path[-1]


    @dataclass
    class Program:
        ops: Sequence[Operation]
        
        def __post_init__(self):
            self.named_ops = dict((op.output, op) for op in self.ops)
            
    actual_ops = []
    def populate_params(op: Operation, prog: Program):
        actual_ops.append(op)
        params = []
        for inp in op.inputs:
            if isinstance(inp, str):
                if inp == 'tokens':
                    params.append(rasp.tokens)
                elif inp == 'indices':
                    params.append(rasp.indices)
                else:
                    child = populate_params(prog.named_ops[inp], prog)
                    params.append(child)
            elif isinstance(inp, Callable):
                params.append(inp)
            elif isinstance(inp, Union[float, int]):
                params.append(inp)
        ret = op.operator(*params)
        named_ret = ret.named(op.output)
        return named_ret


    program = populate_params(ops[terminal_node-2], Program(ops))

    # discard duplicates
    seen = []
    actual_ops = list(filter(lambda x: seen.append(x.output) is None if x.output not in seen else False, actual_ops))
    actual_ops = actual_ops[::-1] # reverse the traversal
    #print(f"Program Length: {len(actual_ops)}")


    return program, actual_ops

def compile_program_into_craft_model(program, vocab, max_seq_len: int):

    COMPILER_BOS = "compiler_bos"
    COMPILER_PAD = "compiler_pad"



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

    return craft_model


def gen_vocab(vocab_size: int, prefix='t', numeric=False):
    if not numeric:
        return [prefix+str(x) for x in range(vocab_size)]
    else:
        return list(range(vocab_size))


def build_program_of_length(n_ops, vocab, numeric_range: tuple, TARGET_PROGRAM_LENGTH):
    program_length = 0
    program, actual_ops = None, None
    while program_length < TARGET_PROGRAM_LENGTH:
        ops = generate_ops(n_ops, vocab, numeric_range)
        program, actual_ops = compile_program(ops)
        program_length = len(actual_ops)
    return program, actual_ops

def choose_vocab_and_ops(ops_range: tuple, vocab_size_range: tuple, numeric_inputs_possible: bool):
    n_ops = randint(*ops_range)
    vocab_size = randint(*vocab_size_range)
    TARGET_PROGRAM_LENGTH = n_ops // 2
    numeric_inputs = choice([True, False]) if numeric_inputs_possible else False
    vocab = gen_vocab(vocab_size, prefix='t', numeric=numeric_inputs)
    return n_ops, vocab, TARGET_PROGRAM_LENGTH

def program_generator(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool):
    n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
    program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
    return program, actual_ops


def program_craft_generator(ops_range: tuple, vocab_size_range: tuple, numeric_range: tuple, numeric_inputs_possible: bool):
    n_ops, vocab, TARGET_PROGRAM_LENGTH = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=vocab_size_range, numeric_inputs_possible=numeric_inputs_possible)
    program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)
    craft_model = compile_program_into_craft_model(program, vocab, max(numeric_range))
    return craft_model, actual_ops







def traverse_prog(prog, lambdas = []):
    """lambdas: provide the names of lambdas in order of program depth, from:
         LAM_LT LAM_LE LAM_GT LAM_GE LAM_NE LAM_EQ LAM_IV LAM_ADD LAM_MUL LAM_SUB LAM_AND LAM_OR"""
    def check_lam_name(lam_name):
        possible_names = set(x[-1] for x in UNI_LAMBDAS) | set(x[-1] for x in  SEQUENCE_LAMBDAS)
        if lam_name in possible_names:
            return lam_name
        else:
            raise(Exception(f"Lambda name {lam_name} was not in the list of possible names: {possible_names}"))
    lambdas = [check_lam_name(lam) for lam in reversed(lambdas)]

    object_names = dict()
    type_assingments = defaultdict(lambda: 0)
    def get_name(obj):
        if isinstance(obj, rasp.TokensType):
            return "tokens"
        elif isinstance(obj, rasp.IndicesType):
            return "indices"
        elif id(obj) not in object_names:
            t = type(obj)
            type_assingments[t] += 1
            object_names[id(obj)] = f"{t.__name__} {type_assingments[t]}"
        return object_names[id(obj)]
    
    def rasp_expr_to_op(expr: rasp.RASPExpr):
        if isinstance(expr, rasp.Select):
            sop1 = get_name(expr.keys)
            sop2 = get_name(expr.queries)
            pred = expr.predicate
            return Operation(type(expr), [sop1, sop2, pred], get_name(expr))
        elif isinstance(expr,  rasp.SelectorWidth):
            sel = get_name(expr.selector)
            return Operation(type(expr), [sel], get_name(expr))
        elif isinstance(expr, rasp.Aggregate):
            sel = get_name(expr.selector)
            sop = get_name(expr.sop)
            return Operation(type(expr), [sel, sop], get_name(expr))
        elif isinstance(expr, rasp.Map):
            sop = get_name(expr.inner)
            lam = expr.f
            if lambdas:
                lam_name = lambdas.pop(0)
            else:
                print("Please input the name of the lambda for:")
                print(expr)
                print(sop)
                print(lam)
                
                print(inspect.getsourcelines(lam))
                lam_name = check_lam_name(input())
            return Operation(type(expr), [sop, lam], get_name(expr), lambda_name=lam_name)
        elif isinstance(expr, rasp.SequenceMap):
            sop1 = get_name(expr.fst)
            sop2 = get_name(expr.snd)
            lam = expr.f
            if lambdas:
                lam_name = lambdas.pop(0)
            else:
                print("Please input the name of the lambda for:")
                print(expr)
                print(sop1)
                print(sop2)
                print(lam)
                
                print(inspect.getsourcelines(lam))
                lam_name = check_lam_name(input())
            return Operation(type(expr), [sop1, sop2, lam], get_name(expr), lambda_name=lam_name)
        else:
            print(f"No translation exists for operator: {expr}")

    actual_ops = [rasp_expr_to_op(prog)]


    current_node = prog
    unvisited = list(zip(prog.children, [id(x) for x in prog.children]))

    def reduce_univisted(unvisited):
        repeats = set()
        out = []
        for x, id in unvisited:
            if id not in repeats:
                repeats.add(id)
                out.append((x, id))
        return out
    unvisited = reduce_univisted(unvisited)


    while unvisited:
        current_node = unvisited.pop(0)[0]
        if isinstance(current_node, rasp.TokensType) or isinstance(current_node, rasp.IndicesType):
            continue # terminal input node
        else:
            actual_ops.append(rasp_expr_to_op(current_node))
            unvisited += list(zip(current_node.children, [id(x) for x in current_node.children]))
            unvisited = reduce_univisted(unvisited)
    return actual_ops

def encode_rasp_program(program, PROG_LEN, lambdas=[], numeric_vars: bool = False):
    actual_ops = traverse_prog(program, lambdas)
    vocab = gen_vocab(PROG_LEN, prefix='t', numeric=numeric_vars)
    craft_model = compile_program_into_craft_model(program, vocab, PROG_LEN)

    encoded_ops = encode_ops(actual_ops)
    encoded_model = encode_craft_model(craft_model)
    return encoded_model, encoded_ops


example_program_dataset = [
    # Program generating lambda,      lambda names, numeric_inputs?
    (lambda: lib.make_length(), [], "length", False),
    (lambda: lib.make_hist(), [], "histogram", False), 
    (lambda: lib.make_frac_prevs((rasp.tokens == "x")), ['LAM_EQ'], "frac_prev", False),
    
    # Requires Linear Sequence Map
    # (lambda: lib.make_shuffle_dyck(pairs=["()", "{}"]), ['LAM_LT'], "2_shuffle_dyck"),
    # (lambda: lib.make_shuffle_dyck(pairs=["()", "{}", "[]"]), ['LAM_LT'], "3_shuffle_dyck"),
    
    # Expects numeric inputs
    (lambda: lib.make_sort(
            rasp.tokens, rasp.tokens, max_seq_len=4, min_key=1), ['LAM_MUL'], 'sort_4', True),
    (lambda: lib.make_sort(
            rasp.tokens, rasp.tokens, max_seq_len=8, min_key=1), ['LAM_MUL'], 'sort_8', True),

    (lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens), [], 'sort_unique', False),
    
    # ???
    (lambda: lib.make_sort_freq(max_seq_len=3), ['LAM_MUL', 'LAM_MUL'], 'sort_freq 3', True),
    (lambda: lib.make_sort_freq(max_seq_len=7), ['LAM_MUL', 'LAM_MUL'], 'sort_freq 7', True),

    # Requires Linear Sequence Map
    # (lambda: lib.make_pair_balance(
    #         sop=rasp.tokens, open_token="(", close_token=")"), [], 'pair_balance'),
    (lambda: rasp.Map(lambda x: x == "t4", rasp.tokens), ['LAM_GT'], 'map_eq_t4', False),
    (lambda: rasp.Map(lambda x: x > 2, rasp.tokens), ['LAM_GT'], 'map_gt_2', True),
    (lambda: rasp.Map(lambda x: x > 1, rasp.Map(lambda x: (2*x + 1) < 4, rasp.tokens)), ['LAM_EQ', 'LAM_GT'], 'map_map_num', True),        
    (lambda: rasp.Map(lambda x: x > 1, rasp.Map(lambda x: x == "t1", rasp.tokens)), ['LAM_EQ', 'LAM_GT'], 'map_map_char', False),        
]

