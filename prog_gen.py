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
    [ rasp.Map,          [SOp, lambda1],                  SOp     , 3],
    [ rasp.Select,       [SOp, rasp.Predicate],           Selector, 3],
    [ rasp.SequenceMap,  [SOp, SOpNumericValue, lambda2], SOp,      1],
    [ rasp.Aggregate,    [Selector, SOp],                 SOp,      2],
    [ rasp.SelectorWidth,[Selector],                      SOp,      2],
    [ rasp.SelectorOr,   [Selector, Selector],            Selector, 1], 
    [ rasp.SelectorAnd,  [Selector, Selector],            Selector, 1], 
    [ rasp.SelectorNot,  [Selector, Selector],            Selector, 1], 
]

df = pd.DataFrame(df, columns = ['cls', 'inp', 'out', 'weight'])
df_no_selector = pd.DataFrame(df[df.inp.apply(lambda x: Selector in x)])


# must ensure all basic comparison lambdas are present


# These function can be combined with a sequence map operator
# sequence_lambdas = [
#     [ "lambda",            "imp",                "input_types",                      "output_type"]
#     [lambda x, y: x + y,   rasp.SOp.__add__,     [SOpNumericValue, SOpNumericValue], SOp,],
#     [lambda x, y: x * y,   rasp.SOp.__mul__,     [SOpNumericValue, SOpNumericValue], SOp,],
#     [lambda x, y: x - y,   rasp.SOp.__sub__,     [SOpNumericValue, SOpNumericValue], SOp,],
#     [lambda x, y: x / y,   rasp.SOp.__truediv__, [SOpNumericValue, SOpNumericValue], SOp,],
#     [lambda x, y: x and y, rasp.SOp.__and__,     [SOpNumericValue, SOpNumericValue], SOp,],
#     [lambda x, y: x or y,  rasp.SOp.__or__,      [SOpNumericValue, SOpNumericValue], SOp,],
# ]

uni_lambdas = [
    lambda x, y: x < y,  
    lambda x, y: x <= y,  
    lambda x, y: x > y,  
    lambda x, y: x >= y,  
    lambda x, y: x != y,
    lambda x, y: x == y, 
    lambda x, y: not x
]

sequence_lambdas = [
    lambda x, y: x + y,  
    lambda x, y: x * y,  
    lambda x, y: x - y,  
    lambda x, y: x / y,  
    lambda x, y: x and y,
    lambda x, y: x or y, 
]


predicates = [
    rasp.Comparison.EQ,
    rasp.Comparison.FALSE,
    rasp.Comparison.TRUE,
    rasp.Comparison.GEQ,
    rasp.Comparison.GT,
    rasp.Comparison.LEQ,
    rasp.Comparison.LT,
    rasp.Comparison.NEQ,
]





#%%

# rasp.SOp.__lt__
# rasp.SOp.__le__
# rasp.SOp.__eq__
# rasp.SOp.__ne__
# rasp.SOp.__add__


from collections import defaultdict
from enum import Enum
class Cat(Enum):
    numeric = 1
    categoric = 2
    boolean = 3


class Scope:
    def __init__(self, token_type: Cat) -> None:    
        self.scope = dict()
        self.names = set()
        self.types = set()
        self.counter = defaultdict(lambda: 0)
        self.name_by_type = defaultdict(lambda: [])
        self.type_cat = dict()
        self.add(SOp, "tokens", token_type)
        self.add(SOp, "indices", Cat.numeric)

    def add(self, t: type, cat: Cat) -> None:
        self.counter[t] += 1
        return self.add(t, str(t) + ' ' + str(self.counter[t]), cat)
    
    def add(self, t: type, name: str, cat: Cat) -> None:
        if name in self.names:
            raise ValueError(f'{name} is already present in the set')
        self.scope[name] = t
        self.type_cat[name] = cat
        self.names.add(name)
        self.types.add(t)
        self.name_by_type[t].append(name)
        return name
    
    def get_cat(self, name: str):
        return self.type_cat[name]
    
    def pick_var(self, y: type):
        return choice(self.name_by_type[y])
    
@dataclass
class Operation:
    operator: type
    inputs: Sequence[Union[str, Callable, rasp.Predicate ]]
    output: str
    out_type: Cat
        





scope = Scope({"tokens": SOp, "indices": SOp}, Cat.categoric)




def sample_function(scope: Scope, input_space_values, input_cat: Cat, max_seq_length: int, df=df):
    sampled = df.sample(weights=df.weight)

    return_type = None
    for input_type in sampled.inp:
        if input_type == SOp:
            pass
        if input_type == SOpNumericValue:
            pass
        if input_type == Selector:
            pass
        if input_type == lambda1:
            chosen = choice(uni_lambdas)
            return_type = Cat.boolean
            obj_cat = scope.get_cat(inputs[0])
            if inputs[0] == "tokens":
                y = choice(input_space_values)
            if inputs[0] == "indices":
                y = randint(0, max_seq_length)
            elif obj_cat == Cat.numeric:
                y = randint(0, max_seq_length)
            elif obj_cat == Cat.categoric:
                y = choice(input_space_values)
            func = partial(chosen, y)
            inputs.append(func)
            
        if input_type == lambda2:
            inputs.append(choice(sequence_lambdas))
        if input_type == rasp.Predicate:
            inputs.append(choice(predicates))
            return_type = Cat.boolean

    
    if sampled.cls == rasp.Map:
        # [SOp, lambda1],
        return_cat = Cat.boolean
        return_type = SOp

        s1 = scope.pick_var(SOp)
        f1 = choice(uni_lambdas)
        
        obj_cat = scope.get_cat(s1)
        if s1 == "tokens":
            y = choice(input_space_values)
        if s1 == "indices":
            y = randint(0, max_seq_length)
        elif obj_cat == Cat.numeric:
            y = randint(0, max_seq_length)
        elif obj_cat == Cat.categoric:
            y = choice(input_space_values)
        func = partial(f1, y)

        # Allocate a variable to hold the return value
        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, func], allocated_name)
              

    elif sampled.cls == rasp.Select:
        # [SOp, rasp.Predicate],          
    elif sampled.cls == rasp.SequenceMap:
        # [SOp, SOpNumericValue, lambda2],
    elif sampled.cls == rasp.Aggregate:
        # [Selector, SOp],                
    elif sampled.cls == rasp.SelectorWidth:
        # [Selector],                     
    elif sampled.cls == rasp.SelectorOr:
        # [Selector, Selector],           
    elif sampled.cls == rasp.SelectorAnd:
        # [Selector, Selector],           
    elif sampled.cls == rasp.SelectorNot:
        # [Selector, Selector],           