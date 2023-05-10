#%%

import jax
import sys
sys.path.append('tracr/')

from utils.plot import *

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')


from tracr.rasp import rasp
#import pandas as pd

SOp = rasp.SOp
Selector = rasp.Selector

df = [
    # Class name, input types, output type
    [  rasp.SelectorOr, ] # we may be able to infer input and output types
]

# must ensure all basic comparison lambdas are present

sequence_lambdas = [
    [lambda x, y: x + y,   rasp.SOp.__add__,     ],
    [lambda x, y: x * y,   rasp.SOp.__mul__,     ],
    [lambda x, y: x - y,   rasp.SOp.__sub__,     ],
    [lambda x, y: x / y,   rasp.SOp.__truediv__, ],
    [lambda x, y: x and y, rasp.SOp.__and__,     ],
    [lambda x, y: x or y,  rasp.SOp.__or__,      ],
]

#%%

# rasp.SOp.__lt__
# rasp.SOp.__le__
# rasp.SOp.__eq__
# rasp.SOp.__ne__
# rasp.SOp.__add__
