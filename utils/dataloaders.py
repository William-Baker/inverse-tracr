
import torch
from torch.nn.utils.rnn import pad_sequence
from data.dataset import program_dataset
import numpy as np

START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

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

def encoded_program_to_onehot(encoded, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE):
    one_hot = np.zeros((encoded.shape[0], OP_NAME_VOCAB_SIZE + 4 * VAR_VOCAB_SIZE))
    segment_sizes = [OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]
    for t in range(encoded.shape[0]):
        ptr = 0
        # Loop through each operation which cotains list of 5 integer id's for each token
        for i in range(len(segment_sizes)):
            id = encoded[t, i]
            #print(f"ID: {id}, x: {ptr + id}, y: {t}")
            one_hot[t, ptr + id] = 1
            ptr += segment_sizes[i]
    return one_hot





def decoder_generator(_program_gen, op_encoder, var_encoder, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE):
    while True:
        program = next(_program_gen)
        encoded_program = encode_program(program, op_encoder, var_encoder)
        onehot_program = encoded_program_to_onehot(encoded_program, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)
        yield onehot_program, encoded_program

from functools import partial
from jax import numpy as jnp


class ProgramDataset(torch.utils.data.Dataset):
    def __init__(self, prog_len, sample_count=1000):
        self.prog_len = prog_len
        self.sample_count = sample_count
        gen, OP_VOCAB, VAR_VOCAB = program_dataset(ops_range=(prog_len,prog_len))
        OP_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_VOCAB), len(VAR_VOCAB)
        op_encoder = dict(zip(OP_VOCAB, [i for i in range(OP_VOCAB_SIZE)]))
        
        op_encoder[START_TOKEN] = OP_VOCAB_SIZE # Add a token for the start of the program
        OP_VOCAB_SIZE += 1
        op_encoder[END_TOKEN] = OP_VOCAB_SIZE # Add a token for the end of the program
        OP_VOCAB_SIZE += 1
        
        var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))
        self.data_iterator = decoder_generator(gen(), op_encoder, var_encoder, OP_VOCAB_SIZE, VAR_VOCAB_SIZE)

        self.OP_NAME_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.op_encoder, self.var_encoder = \
                    OP_VOCAB_SIZE, VAR_VOCAB_SIZE, op_encoder, var_encoder
        
        self.op_decoder = dict(zip(op_encoder.values(), op_encoder.keys()))
        self.var_decoder = dict(zip(var_encoder.values(), var_encoder.keys()))

        self.segment_sizes = [OP_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]

    def __len__(self):
        'Denotes the total number of samples'
        return self.sample_count

    def __getitem__(self, index):
        return next(self.data_iterator)
    
    def collate_fn(self, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        ammount_to_pad = self.prog_len + 2 - targets.shape[1]
        targets = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(targets) # pad the target to the max possible length for the problem
        return jnp.array(inputs), jnp.array(targets)

    def get_collate_fn(self):
        return partial(ProgramDataset.collate_fn, self)

    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], 5))
        logits = np.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax()
                ptr += seg_size
        return classes

    def logit_classes_jnp(self, logits):
        classes = jnp.zeros((logits.shape[0], 5))
        logits = jnp.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax().item()
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

import pandas as pd

class ProgramDatasetFromFile(torch.utils.Dataset):
    def __init__(self, program_file='program_dataset.pkl'):
        self.programs = pd.read_pickle(program_file)
        meta = pd.read_pickle('program_meta.pkl')
        OP_VOCAB = meta['OP_VOCAB']
        VAR_VOCAB = meta['VAR_VOCAB']
        self.prog_len = meta['PROG_LEN']

        OP_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_VOCAB), len(VAR_VOCAB)
        op_encoder = dict(zip(OP_VOCAB, [i for i in range(OP_VOCAB_SIZE)]))
        
        op_encoder[START_TOKEN] = OP_VOCAB_SIZE # Add a token for the start of the program
        OP_VOCAB_SIZE += 1
        op_encoder[END_TOKEN] = OP_VOCAB_SIZE # Add a token for the end of the program
        OP_VOCAB_SIZE += 1
        
        var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))

        self.OP_NAME_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.op_encoder, self.var_encoder = \
                    OP_VOCAB_SIZE, VAR_VOCAB_SIZE, op_encoder, var_encoder
        
        self.op_decoder = dict(zip(op_encoder.values(), op_encoder.keys()))
        self.var_decoder = dict(zip(var_encoder.values(), var_encoder.keys()))

        self.segment_sizes = [OP_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE, VAR_VOCAB_SIZE]

    def __len__(self):
        'Denotes the total number of samples'
        return self.programs.shape[0]

    def __getitem__(self, index):
        entry = self.programs.iloc[index]
        return (entry.x, entry.y)
    
    def collate_fn(self, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        ammount_to_pad = self.prog_len + 2 - targets.shape[1]
        targets = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(targets) # pad the target to the max possible length for the problem
        return jnp.array(inputs), jnp.array(targets)

    def get_collate_fn(self):
        return partial(ProgramDataset.collate_fn, self)

    def logit_classes_np(self, logits):
        classes = np.zeros((logits.shape[0], 5))
        logits = np.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax()
                ptr += seg_size
        return classes

    def logit_classes_jnp(self, logits):
        classes = jnp.zeros((logits.shape[0], 5))
        logits = jnp.array(logits)
        for t in range(logits.shape[0]):
            ptr = 0
            for i, seg_size in enumerate(self.segment_sizes):
                classes[t, i] = logits[t, ptr:ptr + seg_size].argmax().item()
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

