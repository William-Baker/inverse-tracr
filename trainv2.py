#%%
import torch
from datasetv3 import craft_dataset, program_dataset
import jax.numpy as jnp
import jax.nn as nn
import numpy as np

gen, OP_NAME_VOCAB, VAR_VOCAB = program_dataset(ops_range=(30,30))
OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
op_encoder = dict(zip(OP_NAME_VOCAB, [i for i in range(OP_NAME_VOCAB_SIZE)]))
var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))

#%%

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

#%%

g = gen()
program = next(g)

encoded_program = encode_program(program, op_encoder, var_encoder)
onehot_program = encoded_program_to_onehot(encoded_program, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)



#%%

def decoder_generator():
    _program_gen = gen()
    while True:
        program = next(_program_gen)
        encoded_program = encode_program(program, op_encoder, var_encoder)
        onehot_program = encoded_program_to_onehot(encoded_program, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)
        yield onehot_program, encoded_program


#%%
import time
g = decoder_generator()
times = []
for i in range(100):
	start = time.time()
	weights, program = next(g)
	end = time.time()
	times.append(end - start)
print(f"max time: {max(times)}")
print(f"mean time: {np.mean(times)}")
print(f"median time: {np.median(times)}")
#%%

# perf test




#%%

# def encoded_data_gen():
#     g = gen()
#     while True:
#         x, y = next(g)
#         x, y = x, encode_program(y, op_encoder, var_encoder)
        


# data_iterator = encoded_data_gen()
# iter_dataset = torch.data.IterDataset(data_iterator)
# torch.data.Dataset(iter_dataset, batch=32, num_workers=8, prefetch_factor=2, pin_memory=True)




# def loss(pred, targ):
#     pass


# import optax
# optax.cross