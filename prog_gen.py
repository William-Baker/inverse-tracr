#%%

import jax
import sys
sys.path.append('tracr/')

from utils.plot import *
from typing import Union, TypeVar, Sequence, Callable
from random import choice, randint
from functools import partial

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')


from tracr.rasp import rasp
from dataclasses import dataclass
import pandas as pd



SOp = rasp.SOp
Selector = rasp.Selector


NumericValue = Union[int, float]
SOpNumericValue = Union[SOp, int, float]
Value = Union[None, int, float, str, bool]
lambda1 = TypeVar("lambda1")
lambda2 = TypeVar("lambda2")


df = [
    # Class name,         input types,                 output type, weight
    [ rasp.Map,          [lambda1, SOp],                  SOp     , 4],
    [ rasp.Select,       [SOp, SOp, rasp.Predicate],           Selector, 3],
    [ rasp.SequenceMap,  [SOp, SOpNumericValue, lambda2], SOp,      2],
    [ rasp.Aggregate,    [Selector, SOp],                 SOp,      2],
    [ rasp.SelectorWidth,[Selector],                      SOp,      2],
    # [ rasp.SelectorOr,   [Selector, Selector],            Selector, 1],  # These arent implemented correclty in RASP
    # [ rasp.SelectorAnd,  [Selector, Selector],            Selector, 1], 
    # [ rasp.SelectorNot,  [Selector, Selector],            Selector, 1], 
]

df = pd.DataFrame(df, columns = ['cls', 'inp', 'out', 'weight'])
df_no_selector = pd.DataFrame(df[df.inp.apply(lambda x: Selector not in x)])
df_returns_sop = pd.DataFrame(df[df.out == SOp])



UNI_LAMBDAS = [
    lambda x, y: x < y,  
    lambda x, y: x <= y,  
    lambda x, y: x > y,  
    lambda x, y: x >= y,  
    lambda x, y: x != y,
    lambda x, y: x == y, 
    lambda x, y: not x
]

SEQUNCE_LAMBDAS = [
    (lambda x, y: x + y,  []),
    (lambda x, y: x * y,  []),
    (lambda x, y: x - y,  []),
    #(lambda x, y: x / y,  [0]),
    (lambda x, y: x and y,[]),
    (lambda x, y: x or y, []),
]


PREDICATES = [
    rasp.Comparison.EQ,
    rasp.Comparison.FALSE,
    rasp.Comparison.TRUE,
    rasp.Comparison.GEQ,
    rasp.Comparison.GT,
    rasp.Comparison.LEQ,
    rasp.Comparison.LT,
    rasp.Comparison.NEQ,
]




from collections import defaultdict
from enum import Enum
class Cat(Enum):
    numeric = 1
    categoric = 2
    boolean = 3


class Scope:
    def __init__(self, vocabulary: Sequence[Union[str, int, bool]], max_seq_length) -> None:    
        self.scope = dict()
        self.names = set()
        self.types = set()
        self.counter = defaultdict(lambda: 0)
        self.names_by_type = defaultdict(lambda: [])
        self.names_by_type_and_cat = defaultdict(lambda: defaultdict(lambda: []))
        self.type_cat = dict()
        if type(vocabulary[0]) == str:
            token_type = Cat.categoric
        else:
            token_type = Cat.numeric
        self.__add__(SOp, "tokens", token_type)
        self.__add__(SOp, "indices", Cat.numeric)
        self.max_seq_length = max_seq_length
        self.vocabulary = vocabulary

    def add(self, t: type, cat: Cat) -> None:
        self.counter[t] += 1
        return self.__add__(t, str(t) + ' ' + str(self.counter[t]), cat)
    
    def __add__(self, t: type, name: str, cat: Cat) -> None:
        if name in self.names:
            raise ValueError(f'{name} is already present in the set')
        self.scope[name] = t
        self.type_cat[name] = cat
        self.names.add(name)
        self.types.add(t)
        self.names_by_type[t].append(name)
        self.names_by_type_and_cat[t][cat].append(name)
        return name
    
    def get_cat(self, name: str):
        return self.type_cat[name]
    
    def pick_var(self, y: type): 
        return choice(self.names_by_type[y])
    
    def pick_var_cat(self, y: type, cat: Cat):
        return choice(self.names_by_type_and_cat[y][cat])
    
    def var_exists(self, desired_type: type):
        return len(self.names_by_type[desired_type]) > 0
    
    def var_exists_cat(self, desired_type: type, desired_cat: Cat):
        return len(self.names_by_type_and_cat[desired_type][desired_cat]) > 0
    
    def matching_var_exists(self, src, desired_type: type):
        """returns True if a variable with the desired type is in scope with category matching the source"""
        return self.var_exists_cat(desired_type, self.get_cat(src))
    
    def gen_const(self, target_cat: Cat):
        if target_cat == Cat.numeric:
            return randint(0, self.max_seq_length)
        elif target_cat == Cat.categoric:
            return choice(self.vocabulary)
        else:
            raise NotImplementedError()

@dataclass
class Operation:
    operator: type
    inputs: Sequence[Union[str, Callable, rasp.Predicate ]]
    output: str

        




vocab = list('abcd')
max_seq_len = 6
scope = Scope(vocab, max_seq_len)
ops = []



def sample_function(scope: Scope, df=df):
    if scope.var_exists(Selector):
        sampled = df.sample(weights=df.weight).iloc[0]
    else:
        sampled = df_no_selector.sample(weights=df_no_selector.weight).iloc[0]


    if sampled.cls == rasp.Map:
        # [lambda1, SOp],
        
        return_type = SOp
        

        if randint(0,1) == 0: # Boolean operators
            return_cat = Cat.boolean
            
            s1 = scope.pick_var(SOp)
            f1 = choice(UNI_LAMBDAS)
            
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
            f1, bad_vals = choice(SEQUNCE_LAMBDAS)# + [(lambda x, y: x / y,  [0])]) 
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
        op = Operation(sampled.cls, [func, s1], allocated_name)
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
            
        f1, bad = choice(SEQUNCE_LAMBDAS)

        return_type = SOp
        return_cat = s1_cat

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [f1, s1, s2], allocated_name)
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




for i in range(0, 10):
    sample_function(scope, df)

sample_function(scope, df_returns_sop)


@dataclass
class Program:
    ops: Sequence[Operation]
    
    def __post_init__(self):
        self.named_ops = dict((op.output, op) for op in self.ops)
        

def populate_params(op: Operation, prog: Program):
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


program = populate_params(ops[-1], Program(ops))


from utils import compiling_all

prog_name = "sort_unique"
#program, vocab, input_seq = get_program(prog_name, 6)


# assembled_model, rasp_model, craft_model  = compiling_all.compile_rasp_to_model_returns_all(
#       program=program,
#       vocab=vocab,
#       max_seq_len=max_seq_len,
#       causal=False,
#       compiler_bos="bos",
#       compiler_pad="pad",
#       mlp_exactness=100)




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
#%%


craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

#%%
import networkx as nx
nx.draw(rasp_model.graph, with_labels=True)
print(program)

#%%
