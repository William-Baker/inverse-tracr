
import torch
from torch.nn.utils.rnn import pad_sequence
from datasetv4 import program_dataset
import numpy as np

def encode_program(program, op_encoder, var_encoder):
    encoded = np.zeros((len(program)+1, 5), np.int32)
    encoded[0, 0] = op_encoder['PROGRAM_START']
    for t, instruction in enumerate(program):
        # Loop through each operation which cotains list of {'op': 'SelectorWidth', 'p1': 'v1', 'p2': 'NA', 'p3': 'NA', 'r': 'v2'}
        encoded[t+1, 0] = op_encoder[instruction['op']]
        encoded[t+1, 1] = var_encoder[instruction['p1']]
        encoded[t+1, 2] = var_encoder[instruction['p2']]
        encoded[t+1, 3] = var_encoder[instruction['p3']]
        encoded[t+1, 4] = var_encoder[instruction['r']]
    encoded[-1, 0] = op_encoder['PROGRAM_START']
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


class ProgramDataset(torch.utils.data.Dataset):
    def __init__(self, prog_len):
        gen, OP_NAME_VOCAB, VAR_VOCAB = program_dataset(ops_range=(prog_len,prog_len))
        OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE = len(OP_NAME_VOCAB), len(VAR_VOCAB)
        op_encoder = dict(zip(OP_NAME_VOCAB, [i for i in range(OP_NAME_VOCAB_SIZE)]))
        
        op_encoder['PROGRAM_START'] = OP_NAME_VOCAB_SIZE # Add a token for the start of the program
        OP_NAME_VOCAB_SIZE += 1
        op_encoder['PROGRAM_END'] = OP_NAME_VOCAB_SIZE # Add a token for the end of the program
        OP_NAME_VOCAB_SIZE += 1
        
        var_encoder = dict(zip(VAR_VOCAB, [i for i in range(VAR_VOCAB_SIZE)]))
        self.data_iterator = decoder_generator(gen(), op_encoder, var_encoder, OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE)

        self.OP_NAME_VOCAB_SIZE, self.VAR_VOCAB_SIZE, self.op_encoder, self.var_encoder = \
                    OP_NAME_VOCAB_SIZE, VAR_VOCAB_SIZE, op_encoder, var_encoder

    def __len__(self):
        'Denotes the total number of samples'
        return 1000

    def __getitem__(self, index):
        return next(self.data_iterator)
    
    def collate_fn(data):
        inputs = [torch.tensor(d[0]) for d in data]
        targets = [torch.tensor(d[1]) for d in data]
        inputs = pad_sequence(inputs, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        return inputs, targets

    
