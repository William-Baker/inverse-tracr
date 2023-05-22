#%%
import torch
import pandas as pd
import jax.numpy as jnp
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import numpy as np


START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

from dataset import program_dataset

class TorchProgramDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.gen, OP_VOCAB, VAR_VOCAB = program_dataset((30,30))
        self.it = iter(self.gen())
        self.prog_len = 30
        OP_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_VOCAB), len(VAR_VOCAB)
        
        self.OP_VOCAB_SIZE = OP_VOCAB_SIZE
        self.VAR_VOCAB_SIZE = VAR_VOCAB_SIZE

        self.op_encoder = dict(zip(OP_VOCAB, [i for i in range(OP_VOCAB_SIZE)]))
        self.op_encoder[START_TOKEN] = self.OP_VOCAB_SIZE # Add a token for the start of the program
        self.OP_VOCAB_SIZE += 1
        self.op_encoder[END_TOKEN] = self.OP_VOCAB_SIZE # Add a token for the end of the program
        self.OP_VOCAB_SIZE += 1
        
        self.var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))
        self.op_decoder = dict(zip(self.op_encoder.values(), self.op_encoder.keys()))
        self.var_decoder = dict(zip(self.var_encoder.values(), self.var_encoder.keys()))

        self.segment_sizes = [self.OP_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.VAR_VOCAB_SIZE]

    def encode_program(program, op_encoder, var_encoder):
        encoded = np.zeros((len(program)+1, 5), np.int32)
        encoded[0, 0] = op_encoder[START_TOKEN]
        for t, instruction in enumerate(program):
            # Loop through each operation which cotains list of {'op': 'SelectorWidth', 'p1': 'v1', 'p2': 'NA', 'p3': 'NA', 'r': 'v2'}
            encoded[t+1, 0] = op_encoder[instruction['op']]
            encoded[t+1, 1] = var_encoder[instruction['p1']]
            encoded[t+1, 2] = var_encoder[instruction['p2']]
            encoded[t+1, 3] = var_encoder[instruction['p3']]
            encoded[t+1, 4] = var_encoder[instruction['r']]
        encoded[-1, 0] = op_encoder[END_TOKEN]
        return encoded
    def encoded_program_to_onehot(encoded, OP_VOCAB_SIZE, VAR_VOCAB_SIZE, segment_sizes):
        one_hot = np.zeros((encoded.shape[0], OP_VOCAB_SIZE + 4 * VAR_VOCAB_SIZE))
        for t in range(encoded.shape[0]):
            ptr = 0
            # Loop through each operation which cotains list of 5 integer id's for each token
            for i in range(len(segment_sizes)):
                id = encoded[t, i]
                one_hot[t, ptr + id] = 1
                ptr += segment_sizes[i]
        return one_hot

    def collate_fn(prog_len, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        ammount_to_pad = prog_len + 2 - targets.shape[1]
        targets = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(targets) # pad the target to the max possible length for the problem
        #return jnp.array(inputs), jnp.array(targets)
        return inputs, targets

    def __len__(self):
        'Denotes the total number of samples'
        return 10000

    def __getitem__(self, index):
        y = next(self.it)
        y = TorchProgramDataset.encode_program(y, self.op_encoder, self.var_encoder)
        x = TorchProgramDataset.encoded_program_to_onehot(y, self.OP_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.segment_sizes)
        return x,y
    
    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], self.OP_VOCAB_SIZE))
        logits = np.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax()
                ptr += seg_size
        return classes

    def decode_pred(self, y, batch_index: int):
        pred = y[batch_index, :, :]

        if pred.shape[-1] > 5: # compute the argmax in each segment
            pred = self.logit_classes_np(pred)

        translated = str()
        for t in range(pred.shape[0]):
            if pred[t, :].sum().item() == 0:
                translated += "<PAD>\n"
                continue
            op = self.op_decoder[pred[t, 0].item()]
            translated += op
            if op not in [START_TOKEN, END_TOKEN]:
                for i in range(1,5):
                    translated += " " + self.var_decoder[pred[t, i].item()]
            translated += "\n"
        return translated
    
from torch.utils.data import DataLoader

dataset = TorchProgramDataset()
train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2, collate_fn=partial(TorchProgramDataset.collate_fn, dataset.prog_len))#, pin_memory=True)



it = iter(train_dataloader)


x,y = next(it)
# %%

x,y = next(it)

print(dataset.decode_pred(x, 0))

print(dataset.decode_pred(y, 0))

# %%

# dataset.logit_classes_np(x[0, :, :])


#%%
import time
start = time.time()
for i in range(10):
    x,y = next(it)
end = time.time()
print(end - start)
