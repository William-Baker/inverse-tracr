#%%
import torch
import pandas as pd
import jax.numpy as jnp
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import numpy as np

START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

class ProgramDataset(torch.utils.data.Dataset):
    def __init__(self, program_file='program_dataset.pkl'):
        pass
    def __len__(self):
        return 1000
    def __getitem__(self, index):
        return np.ones((20, 219)), np.ones((20, 7))
    

    
from torch.utils.data import DataLoader

dataset = ProgramDataset()
train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2)#, pin_memory=True)

#%%

it = iter(train_dataloader)
#%%

next(it)
# %%

# x,y = next(it)

# print(dataset.decode_pred(x, 0))

# # %%
