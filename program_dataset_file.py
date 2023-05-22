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
        self.df = pd.read_pickle(program_file)
        meta = pd.read_pickle('.data/program_meta.pkl')
        OP_VOCAB = meta['OP_VOCAB']
        VAR_VOCAB = meta['VAR_VOCAB']
        self.prog_len = meta['PROG_LEN']
        OP_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_VOCAB), len(VAR_VOCAB)
        self.segment_sizes = [OP_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]
        self.OP_VOCAB_SIZE = OP_VOCAB_SIZE
        self.VAR_VOCAB_SIZE = VAR_VOCAB_SIZE
        
        self.var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))
        self.op_encoder = dict(zip(OP_VOCAB, [i for i in range(OP_VOCAB_SIZE)]))
        self.op_decoder = dict(zip(self.op_encoder.values(), self.op_encoder.keys()))
        self.var_decoder = dict(zip(self.var_encoder.values(), self.var_encoder.keys()))

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
        return jnp.array(inputs), jnp.array(targets)

    def __len__(self):
        'Denotes the total number of samples'
        return self.df.shape[0]

    def __getitem__(self, index):
        y = self.df.y.iloc[index]
        x = ProgramDataset.encoded_program_to_onehot(y, self.OP_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.segment_sizes)
        return x,y
    
    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], 5))
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

dataset = ProgramDataset(program_file='.data/program_dataset1.pkl')
train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2, collate_fn=partial(ProgramDataset.collate_fn, dataset.prog_len))#, pin_memory=True)

# #%%

it = iter(train_dataloader)
#%%

next(it)
# # %%

# x,y = next(it)

# print(dataset.decode_pred(x, 0))

# %%
