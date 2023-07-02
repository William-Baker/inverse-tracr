#%%
import torch
import pandas as pd
import jax.numpy as jnp
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import numpy as np
from typing import Union
from data.parameter_encoder import encode_sample


START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

from data.dataset import craft_dataset, program_craft_generator_bounded, program_craft_generator_unbounded

class TorchParameterProgramDataset(torch.utils.data.Dataset):
    def __init__(self, prog_len: int, no_samples = 10000, generator_backend=Union['bounded', 'unbounded'], bounded_timeout_multiplier=1, vocab_size_range=(6,6), numeric_range=(6,6), numeric_inputs_possible: bool = False):
        self.vocab_size_range, self.numeric_range, self.numeric_inputs_possible = vocab_size_range, numeric_range, numeric_inputs_possible
        self.no_samples = no_samples
        func = program_craft_generator_unbounded
        if generator_backend == 'bounded':
            func = program_craft_generator_bounded
        self.prog_len = prog_len
        self.gen, OP_VOCAB, VAR_VOCAB = craft_dataset((prog_len,prog_len), func=func, timeout_multiplier=bounded_timeout_multiplier.as_integer_ratio,
                                            vocab_size_range=self.vocab_size_range, numeric_range=self.numeric_range, numeric_inputs_possible=self.numeric_inputs_possible)
        self.it = iter(self.gen())
        
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
        self.encoders = [self.op_encoder, self.var_encoder, self.var_encoder, self.var_encoder, self.var_encoder]

    def encode_program(program, op_encoder, var_encoder):
        encoded = np.zeros((len(program)+2, 5), np.int32)
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

    def collate_fn(PROG_LEN, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        loss_masks = [torch.tensor(d[2], device='cpu') for d in data]
        attention_masks = [torch.tensor(d[3], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        attention_masks = pad_sequence(attention_masks, batch_first=True)
        #targets = pad_sequence(targets, batch_first=True)

        targets = torch.stack(targets)
        loss_masks = torch.stack(loss_masks)
        
        # pad the inputs to be at least as long as the max output program length
        ammount_to_pad = PROG_LEN + 2 - inputs.shape[1]
        if ammount_to_pad > 0:
            inputs = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(inputs) # pad the inputs to the same length as the target at least
            attention_masks = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(attention_masks)
        return np.array(inputs), np.array(targets), np.array(loss_masks), np.array(attention_masks[:,:,0])
    
    def collate_fn_w_posid(PROG_LEN, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        loss_masks = [torch.tensor(d[2], device='cpu') for d in data]
        attention_masks = [torch.tensor(d[3], device='cpu') for d in data]
        pos_ids = [torch.tensor(d[4], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        attention_masks = pad_sequence(attention_masks, batch_first=True)
        pos_ids = pad_sequence(pos_ids, batch_first=True)
        

        targets = torch.stack(targets)
        loss_masks = torch.stack(loss_masks)
        
        # pad the inputs to be at least as long as the max output program length
        ammount_to_pad = PROG_LEN + 2 - inputs.shape[1]
        if ammount_to_pad > 0:
            inputs = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(inputs) # pad the inputs to the same length as the target at least
            attention_masks = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(attention_masks)
            pos_ids = torch.nn.ConstantPad1d((0, ammount_to_pad), 0)(pos_ids)
        return np.array(inputs), np.array(targets), np.array(loss_masks), np.array(attention_masks[:,:,0]), np.array(pos_ids)

    def __len__(self):
        'Denotes the total number of samples'
        return self.no_samples

    def __getitem__(self, index):
        x, y = next(self.it)
        y = TorchParameterProgramDataset.encode_program(y, self.op_encoder, self.var_encoder)
        # x = TorchProgramDataset.encoded_program_to_onehot(y, self.OP_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.segment_sizes)
        
        # ammount_to_pad = self.prog_len + 2 - y.shape[0]
        # y = np.concatenate((y,np.zeros((ammount_to_pad, y.shape[1]))), axis=0)
        # y = y.astype(int)
        # print(y.shape)
        # assert (y.shape[0] == self.prog_len + 2) and (y.shape[1] == 5)

        return x,y

    def post_process_step(max_prog_len, x, y):
        sample_prog_length = y.shape[0]
        ammount_to_pad = max_prog_len + 2 - y.shape[0]
        padding = np.zeros((ammount_to_pad, y.shape[1]))
        y = np.concatenate((y,padding), axis=0)
        y = y.astype(int)

        loss_mask = np.ones((sample_prog_length))#, y.shape[1]))
        loss_mask = np.concatenate((loss_mask,padding[:, 0]), axis=0)

        enc_x, y = encode_sample(x, y, max_prog_len=max_prog_len)
        attention_mask = np.ones(enc_x.shape[0])

        return enc_x,y,loss_mask, attention_mask
    
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
            # if pred[t].sum().item() == 0: # skip padding
            #     continue
            op = self.op_decoder[pred[t, 0].item()]
            translated += op
            for i in range(1,5):
                translated += " " + self.var_decoder[pred[t, i].item()]
            translated += "\n"
        return translated





if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TorchParameterProgramDataset(15, generator_backend='bounded', bounded_timeout_multiplier=10)

    it = iter(dataset)
    x,y = next(it)

    #train_dataloader = DataLoader(dataset, batch_size=2, num_workers=8, prefetch_factor=2, collate_fn=partial(TorchParameterProgramDataset.collate_fn, dataset.prog_len))#, pin_memory=True)
    # train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1, 
    #                             prefetch_factor=2, )
                                #collate_fn=partial(TorchParameterProgramDataset.collate_fn, dataset.prog_len))#, pin_memory=True)





    # x,y = next(it)

    print(dataset.decode_pred(y.reshape(1, -1, 5), 0))

    # print(dataset.decode_pred(y, 0))


    # dataset.logit_classes_np(x[0, :, :])


    import time
    print("timing")
    start = time.time()
    for i in range(10):
        x,y = next(it)
    end = time.time()
    print(end - start)

# %%
