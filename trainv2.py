import torch
from datasetv2 import craft_dataset
from sklearn.preprocessing import OrdinalEncoder

gen, OP_NAME_VOCAB, VAR_VOCAB = craft_dataset(ops_range=(30,30))
OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
op_encoder = dict(OP_NAME_VOCAB, [i for i in range(VOC)])





iter_dataset = torch.data.IterDataset(data_iterator)
torch.data.Dataset(iter_dataset, batch=32, num_workers=8, prefetch_factor=2, pin_memory=True)


import jax.numpy as jnp

def loss(pred, targ):
    pass
    