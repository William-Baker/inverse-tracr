import torch

#%%

from datasetv2 import craft_dataset
from sklearn.preprocessing import OrdinalEncoder

gen, OP_NAME_VOCAB, VAR_VOCAB = craft_dataset(ops_range=(30,30))

op_encoder = OrdinalEncoder()#categories=[OP_NAME_VOCAB]), OrdinalEncoder() #categories=OP_NAME_VOCAB), OneHotEncoder(categories=VAR_VOCAB)

#%%
import numpy as np
np.diag(OP_NAME_VOCAB)
op_encoder.fit(np.array(OP_NAME_VOCAB).reshape(-1,1))
#%%
import numpy as np
op_encoder.transform(np.array(OP_NAME_VOCAB[3]).reshape(-1,1))
#%%

data_iterator = gen()




iter_dataset = torch.data.IterDataset(data_iterator)
torch.data.Dataset(iter_dataset, batch=32, num_workers=8, prefetch_factor=2, pin_memory=True)


import jax.numpy as jnp
OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
def loss(pred, targ):
    pass
    