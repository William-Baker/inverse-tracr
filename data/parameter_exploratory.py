#%%
from data.dataloader_streams import StreamReader
import numpy as np
pth = '.data/iTracr_dataset/'
dataset = StreamReader(pth)

it = iter(dataset)
x,y = next(it)

#%%

layer_names = []
layer_components = []
it = iter(dataset)
for _ in range(len(dataset)):
    x,y = next(it)
    if x.shape == ():
        print("empty sample")
        print(y)
        print(type(x))
        print(x.item()['MLP'])
        print("x:")
        print(x)
        continue
    layer_names += [list(i.keys())[0] for i in x]
    for j in x:
        layer_components += sum([list(i.keys()) for i in j.values()], [])

#%%

import pandas as pd
print(pd.Series(layer_names).value_counts())
print(pd.Series(layer_components).value_counts())




# %%
from data.program_dataloader import TorchProgramDataset
pdata = TorchProgramDataset(15)


# %%

# HEAD    116212
# MLP      57219
# Name: count, dtype: int64
# w_qk    116212
# w_ov    116212
# fst      57219
# snd      57219

# Compiles to an empty set of parameters
# PROGRAM_START
# Map LAM_GT indices NA v1
# Map LAM_GT v1 NA v2
# Map LAM_GE v2 NA v3
# Map LAM_LT v3 NA v4
# Map LAM_GE v4 NA v5
# Map LAM_LE v5 NA v6
# PROGRAM_END


#%%

it = iter(dataset)


#%%
x,y = next(it)
print(pdata.decode_pred(y.reshape(1, -1, 5), 0))
print([list(i.keys())[0] for i in x])


# %%


#   h m?    Select indices indices PRED_LEQ v1
#   h m    SelectorWidth v1 NA NA v2
#   h      Aggregate v1 indices NA v4
#     m    Map LAM_ADD v4 NA v5
#     m    SequenceMap LAM_ADD v2 v2 v3
#   h m?   Select v5 v5 PRED_LT v6
#          Aggregate v6 v3 NA v7