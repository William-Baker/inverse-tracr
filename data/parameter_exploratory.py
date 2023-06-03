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
    x = x.reshape(-1)
    y = y.reshape(-1)
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

# HEAD    383036
# MLP     293807

#%%

# inputs will be formatted as
# | Timestep type | Parameter block | Architecture |
# 
# Timestep type: {PAD, w_qk, w_ov, fst, snd}
# Parameter block upto 512 floats 
# Architecture concatenated onehot encoded layer types e.g. Head, MLP, head = [100110<PAD>]
#  (padded upto the program length x2)

TIMESTEP_ENCODER = dict(zip(['PAD', 'w_qk', 'w_ov', 'fst', 'snd'], range(5)))
PARAMETER_BLOCK_SZ = 512
ARCHITECTURE_ENCODER = {'PAD': 0, 'HEAD': 1, 'MLP': 2}

TIMESTEP_DECODER = dict(zip(TIMESTEP_ENCODER.values(), TIMESTEP_ENCODER.keys()))
ARCHITECTURE_DECODER = dict(zip(ARCHITECTURE_ENCODER.values(), ARCHITECTURE_ENCODER.keys()))

from typing import Sequence
def encode_architecture(layer_types: Sequence[str], max_prog_length: int):
    encoding = np.zeros((max_prog_length, len(ARCHITECTURE_ENCODER)))
    for layer_no, layer_name in enumerate(layer_types):
        encoding[layer_no, ARCHITECTURE_ENCODER[layer_name]] = 1
    return encoding.reshape(-1)

def decoder_architecture(encoding: np.ndarray):
    encoding = encoding.reshape(-1, len(ARCHITECTURE_DECODER))
    layer_types = [ARCHITECTURE_DECODER[idx] for idx in encoding.argmax(axis=1) if idx != 0]
    return layer_types

    

def block_params(input_sequence: np.ndarray):
    # dict of dict with values being the parameters
    blocks = []
    block_names = []
    for layer in input_sequence:
        for layer_name, layer_components in layer.items():
            for component_name, component_params in layer_components.items():
                flat_params = component_params.reshape(-1)
                windows = np.arange(start=0, stop=flat_params.shape[0], step=PARAMETER_BLOCK_SZ)
                for start in windows:
                    end = start + PARAMETER_BLOCK_SZ
                    if end < flat_params.shape[0]:
                        blocks.append(flat_params[start:end])
                    else:
                        pad_wanting = flat_params[start:]
                        padded = np.concatenate([pad_wanting, np.zeros(PARAMETER_BLOCK_SZ - pad_wanting.shape[0])])
                        blocks.append(padded)
                    block_names.append(component_name)
    return (block_names, blocks)

names, blocks = block_params(x)




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