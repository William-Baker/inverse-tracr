#%%
from datasetv4 import craft_dataset
gen, OP_VOCAB, VAR_VOCAB = craft_dataset()
g = gen()

# while True:
weights, program = next(g)

#%%

# test maximum time required to generate a program
from datasetv4 import craft_dataset
import time
import numpy as np
from tqdm import tqdm
gen, OP_VOCAB, VAR_VOCAB = craft_dataset(ops_range=(10,10))
g = gen()
weights, program = next(g)

times = []
for i in tqdm(range(50)):
	start = time.time()
	weights, program = next(g)
	end = time.time()
	times.append(end - start)
print(f"max time: {max(times)}")
print(f"mean time: {np.mean(times)}")
print(f"median time: {np.median(times)}")


#%%

import torch 
import numpy as np

gen, OP_NAME_VOCAB, VAR_VOCAB = craft_dataset(ops_range=(10,10))
OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
op_encoder = dict(zip(OP_NAME_VOCAB, [i for i in range(OP_NAME_VOCAB_SIZE)]))
var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))

def encode_program(program, op_encoder, var_encoder):
    encoded = np.zeros((5, len(program)), np.int32)
    for t, instruction in enumerate(program):
        # Loop through each operation which cotains list of {'op': 'SelectorWidth', 'p1': 'v1', 'p2': 'NA', 'p3': 'NA', 'r': 'v2'}
        encoded[0, t] = op_encoder[instruction['op']]
        encoded[1, t] = var_encoder[instruction['p1']]
        encoded[2, t] = var_encoder[instruction['p2']]
        encoded[3, t] = var_encoder[instruction['p3']]
        encoded[4, t] = var_encoder[instruction['r']]
    return encoded

def encoded_program_to_onehot(encoded, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE):
    one_hot = np.zeros((OP_NAME_VOCAB_SIZE + 4 * VAR_VOCAB_SIZE, encoded.shape[-1]))
    segment_sizes = [OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]
    for t in range(encoded.shape[-1]):
        ptr = 0
        # Loop through each operation which cotains list of 5 integer id's for each token
        for i in range(len(segment_sizes)):
            id = encoded[i, t]
            #print(f"ID: {id}, x: {ptr + id}, y: {t}")
            one_hot[ptr + id, t] = 1
            ptr += segment_sizes[i]
    return one_hot



def decoder_generator():
    _program_gen = gen()
    while True:
        program = next(_program_gen)
        encoded_program = encode_program(program, op_encoder, var_encoder)
        onehot_program = encoded_program_to_onehot(encoded_program, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)
        yield onehot_program, encoded_program

def encoded_data_gen():
    g = gen()
    while True:
        x, y = next(g)
        x, y = x, encode_program(y, op_encoder, var_encoder)
        yield x, y
        


data_iterator = encoded_data_gen()
#%%
iter_dataset = torch.data.IterDataset(data_iterator)
#%%
from torch.data
torch.data.Dataset(data_iterator, batch=32, num_workers=8, prefetch_factor=2, pin_memory=True)

#%%



#%%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_iterator):
        self.data_iterator = data_iterator

    def __len__(self):
        'Denotes the total number of samples'
        return 1000

    def __getitem__(self, index):
        return next(self.data_iterator)
    
from torch.utils.data import DataLoader

dataset = Dataset(data_iterator)
train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2, pin_memory=True)

#%%

it = iter(train_dataloader)
next(it)