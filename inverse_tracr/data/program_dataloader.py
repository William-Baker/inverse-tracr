#%%
import time
import torch
import pandas as pd
import jax.numpy as jnp
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import numpy as np
from random import shuffle
from torch.utils.data import DataLoader
from inverse_tracr.data.dataset import example_program_dataset
from inverse_tracr.data.dataloaders import ProgramEncoder

START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'


class TorchProgramDataset(torch.utils.data.Dataset):
    def __init__(self, prog_len, no_samples = 10000, shuffled_inputs=True):
        self.shuffled_inputs = shuffled_inputs
        self.no_samples = no_samples
        self.prog_len = prog_len
        self.prog_enc = ProgramEncoder(self.prog_len)


    def collate_fn(prog_len, data):
        inputs = [torch.tensor(d[0], device='cpu') for d in data]
        targets = [torch.tensor(d[1], device='cpu') for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        ammount_to_pad = prog_len + 2 - targets.shape[1]
        targets = torch.nn.ConstantPad2d((0, 0, 0, ammount_to_pad), 0)(targets) # pad the target to the max possible length for the problem
        return np.array(inputs), np.array(targets)

    def __len__(self):
        'Denotes the total number of samples'
        return self.no_samples

    def __getitem__(self, index):
        prog = next(self.it)
        y = self.prog_enc.tokenise_program()
        if not self.shuffled_inputs:
            x = self.prog_enc.tokens_to_onehot(y) # reuse the encoding if we're not shuffling
        else:
            shuffle(prog)
            self.prog_enc.program_to_onehot(prog)
        return x,y
    


    def logit_classes_np(self, logits):
        return self.prog_enc.logit_classes_np(logits)

    def logit_classes_jnp(self, logits):
        return self.prog_enc.logit_classes_jnp(logits)
    
    def decode_pred(self, y, batch_index: int):
        pred = y[batch_index, :, :]

        if pred.shape[-1] > 5: # compute the argmax in each segment
            pred = self.logit_classes_np(pred)

        translated = self.prog_enc.decode_pred(pred)
        return translated

if __name__ == "__main__":

    dataset = TorchProgramDataset(30)
    train_dataloader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=2, collate_fn=partial(TorchProgramDataset.collate_fn, dataset.prog_len))#, pin_memory=True)



    it = iter(train_dataloader)


    x,y = next(it)
    # %%

    x,y = next(it)

    print(dataset.decode_pred(x, 0))

    print(dataset.decode_pred(y, 0))



    #%%
    pred = y - np.random.randint(0,2, size=y.shape)
    pred = np.maximum(pred, 0)#np.zeros(y.shape))

    #img = plot_orginal_heatmaps(y, pred, dataset)
    # %%

    # dataset.logit_classes_np(x[0, :, :])


    #%%
    start = time.time()
    for i in range(10):
        x,y = next(it)
    end = time.time()
    print(end - start)
