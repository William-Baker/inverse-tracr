from tracr.rasp import rasp
from tracr.rasp import rasp
import pandas as pd
from typing import Union, TypeVar






SOp = rasp.SOp
Selector = rasp.Selector


NumericValue = Union[int, float]
SOpNumericValue = Union[SOp, int, float]
Value = Union[None, int, float, str, bool]
lambda1 = TypeVar("lambda1")
lambda2 = TypeVar("lambda2")
NO_PARAM = 'NA'


RASP_OPS = [
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

RASP_OPS = pd.DataFrame(RASP_OPS, columns = ['cls', 'inp', 'out', 'weight'])
RASP_OPS['cls_name'] = RASP_OPS.cls.apply(lambda x: x.__name__)
RASP_OPS_NO_SELECTOR = pd.DataFrame(RASP_OPS[RASP_OPS.inp.apply(lambda x: Selector not in x)])
RASP_OPS_RETURNS_SOP = pd.DataFrame(RASP_OPS[RASP_OPS.out == SOp])



UNI_LAMBDAS = [
    (lambda x, y: x < y,  'LAM_LT'), 
    (lambda x, y: x <= y, 'LAM_LE'),       
    (lambda x, y: x > y,  'LAM_GT'),      
    (lambda x, y: x >= y, 'LAM_GE'),       
    (lambda x, y: x != y, 'LAM_NE'),     
    (lambda x, y: x == y, 'LAM_EQ'),      
    (lambda x, y: not x,  'LAM_IV'),   
]

SEQUENCE_LAMBDAS = [
    (lambda x, y: x + y,  [] , 'LAM_ADD'),
    (lambda x, y: x * y,  [] , 'LAM_MUL'),
    (lambda x, y: x - y,  [] , 'LAM_SUB'),
    #(lambda x, y: x / y,  [0], 'LAM_DIV'),
    (lambda x, y: x and y,[] , 'LAM_AND'),
    (lambda x, y: x or y, [] , 'LAM_OR'),
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

NAMED_PREDICATES = dict([
    (rasp.Comparison.EQ,    'PRED_EQ'),
    (rasp.Comparison.FALSE, 'PRED_FALSE'),
    (rasp.Comparison.TRUE,  'PRED_TRUE'),
    (rasp.Comparison.GEQ,   'PRED_GEQ'),
    (rasp.Comparison.GT,    'PRED_GT'),
    (rasp.Comparison.LEQ,   'PRED_LEQ'),
    (rasp.Comparison.LT,    'PRED_LT'),
    (rasp.Comparison.NEQ,   'PRED_NEQ'),
])