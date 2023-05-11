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
    [ rasp.Map,          [SOp, lambda1],                  SOp     , 2],
    [ rasp.Select,       [SOp, SOp, rasp.Predicate],           Selector, 3],
    [ rasp.SequenceMap,  [SOp, SOpNumericValue, lambda2], SOp,      4],
    [ rasp.Aggregate,    [Selector, SOp],                 SOp,      2],
    [ rasp.SelectorWidth,[Selector],                      SOp,      2],
    [ rasp.SelectorOr,   [Selector, Selector],            Selector, 1], 
    [ rasp.SelectorAnd,  [Selector, Selector],            Selector, 1], 
    [ rasp.SelectorNot,  [Selector, Selector],            Selector, 1], 
]

df = pd.DataFrame(df, columns = ['cls', 'inp', 'out', 'weight'])
df_no_selector = pd.DataFrame(df[df.inp.apply(lambda x: Selector not in x)])


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
    lambda x, y: x + y,  
    lambda x, y: x * y,  
    lambda x, y: x - y,  
    lambda x, y: x / y,  
    lambda x, y: x and y,
    lambda x, y: x or y, 
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

# TODO add a small chance of generating a const SOp / Selector
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
    
    def pick_var(self, y: type): # TODO add a small chance of generating a const SOp / Selector
        return choice(self.names_by_type[y])
    
    def pick_var_cat(self, y: type, cat: Cat): # TODO add a small chance of generating a const SOp / Selector
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

        





scope = Scope(list('abcd'), 6)
ops = []



def sample_function(scope: Scope, df=df):
    if scope.var_exists(Selector):
        sampled = df.sample(weights=df.weight).iloc[0]
    else:
        sampled = df_no_selector.sample(weights=df_no_selector.weight).iloc[0]

    
    if sampled.cls == rasp.Map:
        # [SOp, lambda1],
        return_cat = Cat.boolean
        return_type = SOp

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

        # Allocate a variable to hold the return value
        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, func], allocated_name)
        ops.append(op)

    elif sampled.cls == rasp.SequenceMap: # must have double the weight of Map
        # TODO this might only accept numeric s1 inputs, but I guess we''ll find out
        # [SOp, SOpNumericValue, lambda2],
        # todo chance of const full selector/sop
        s1 = scope.pick_var(SOp)
        s1_cat = scope.get_cat(s1)
        if scope.var_exists_cat(SOp, s1_cat) and randint(0,1) == 0: # s2 wil be an SOp
            s2 = scope.pick_var_cat(SOp, s1_cat)
        else: # s2 will be const
            if  randint(0,2) <= 1: # 2/3 of the time generate an int
                s2 = scope.gen_const(Cat.numeric)
            else: # occasionally generate a float - may have a different impl
                s2 = float(scope.gen_const(Cat.numeric)) + randint(0,100)/100 
        f1 = choice(SEQUNCE_LAMBDAS)

        return_type = SOp
        return_cat = s1_cat

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2, f1], allocated_name)
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


    elif (sampled.cls == rasp.SelectorOr) or (sampled.cls == rasp.SelectorAnd) or (sampled.cls == rasp.SelectorNot):
        # [Selector, Selector],           
        # todo chance of const full selector/sop
        s1 = scope.pick_var(Selector)
        s2 = scope.pick_var(Selector)

        return_type = Selector
        return_cat = Cat.boolean

        allocated_name = scope.add(return_type, return_cat)
        op = Operation(sampled.cls, [s1, s2], allocated_name)
        ops.append(op)






for i in range(0, 10):
    sample_function(scope, df)

#%%