#%%
import time
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Union
from torch.utils.data import DataLoader

from inverse_tracr.data.parameter_encoder import encode_sample
from inverse_tracr.data.encoded_dataloaders import craft_dataset, program_craft_generator_bounded, program_craft_generator_unbounded
from inverse_tracr.data.dataloaders import ProgramEncoder
from inverse_tracr.data.program_dataloader import TorchProgramDataset


START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

class TorchParameterProgramDataset(TorchProgramDataset):
    def __init__(self, min_prog_len:int, max_prog_len: int, generator_backend=Union['bounded', 'unbounded'], bounded_timeout_multiplier=1, vocab_size_range=(6,6), numeric_range=(6,6), numeric_inputs_possible: bool = False):
        self.vocab_size_range, self.numeric_range, self.numeric_inputs_possible = vocab_size_range, numeric_range, numeric_inputs_possible
        func = program_craft_generator_unbounded
        if generator_backend == 'bounded':
            func = program_craft_generator_bounded
        self.prog_len = max_prog_len
        self.gen, OP_VOCAB, VAR_VOCAB = craft_dataset((min_prog_len,max_prog_len), func=func, timeout_multiplier=bounded_timeout_multiplier.as_integer_ratio,
                                            vocab_size_range=self.vocab_size_range, numeric_range=self.numeric_range, numeric_inputs_possible=self.numeric_inputs_possible)

        def lam():
            while True: 
                yield self.gen()                        
        self.it = iter(lam())

        self.prog_enc = ProgramEncoder(self.prog_len)
    
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


    def __getitem__(self, index):
        x, y = next(self.it)
        y = self.prog_enc.tokenise_program(y)
        return x,y

    def post_process_step(max_prog_len, x, y, TIMESTEPS, ARCH_LABELS):
        sample_prog_length = y.shape[0]
        ammount_to_pad = max_prog_len + 2 - y.shape[0]
        padding = np.zeros((ammount_to_pad, y.shape[1]))
        y = np.concatenate((y,padding), axis=0)
        y = y.astype(int)

        loss_mask = np.ones((sample_prog_length))#, y.shape[1]))
        loss_mask = np.concatenate((loss_mask,padding[:, 0]), axis=0)

        enc_x, y = encode_sample(x, y, max_prog_len=max_prog_len, TIMESTEPS=TIMESTEPS, ARCH_LABELS=ARCH_LABELS)
        attention_mask = np.ones(enc_x.shape[0])

        return enc_x,y,loss_mask, attention_mask

    def tokens_to_onehot(self, encoded):
        return self.prog_enc.tokens_to_onehot(encoded)
    
    def get_segment_sizes(self):
        return self.prog_enc.segment_sizes





if __name__ == "__main__":

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


    print("timing")
    start = time.time()
    for i in range(10):
        x,y = next(it)
    end = time.time()
    print(end - start)

# %%
