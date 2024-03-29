#%%

from typing import Sequence
import numpy as np
import pandas as pd

from inverse_tracr.data.dataloader_streams import StreamReader




# inputs will be formatted as
# | Timestep type | Terminal Block Flag | Parameter block | Architecture |
# 
# Timestep type: {PAD, w_qk, w_ov, fst, snd}
# Terminal block flag: indicates if this is the last block for the given layer, 
#    important to differentiate between two sequential params of the same time
# Parameter block upto 512 floats 
# Architecture concatenated onehot encoded layer types e.g. Head, MLP, head = [100110<PAD>]
#  (padded upto the program length x2)

PARAMETER_BLOCK_SZ = 512

CRAFT_TIMESTEPS = ['PAD', 'w_qk', 'w_ov', 'fst', 'snd', 'PROGRAM_START', 'PROGRAM_END']
JAX_TIMESTEPS = ['PAD', 'key', 'value', 'query', 'linear', 'linear_1', 'linear_2', 'PROGRAM_START', 'PROGRAM_END']

CRAFT_ARCH = ['PAD', 'HEAD', 'MLP']
JAX_ARCH = ['PAD', 'MHA', 'MLP']

def encode_architecture(layer_types: Sequence[str], max_prog_length: int, ARCH_LABELS: Sequence[str]):
    ARCHITECTURE_ENCODER = dict(zip(ARCH_LABELS, range(len(ARCH_LABELS))))
    encoding = np.zeros((max_prog_length*5, len(ARCHITECTURE_ENCODER)))
    for layer_no, layer_name in enumerate(layer_types):
        encoding[layer_no, ARCHITECTURE_ENCODER[layer_name]] = 1
    return encoding.reshape(-1)

def decoder_architecture(encoding: np.ndarray, ARCH_LABELS: Sequence[str]):
    ARCHITECTURE_DECODER = dict(zip(range(len(ARCH_LABELS)), ARCH_LABELS))
    encoding = encoding.reshape(-1, len(ARCHITECTURE_DECODER))
    layer_types = [ARCHITECTURE_DECODER[idx] for idx in encoding.argmax(axis=1) if idx != 0]
    return layer_types



def block_params(input_sequence: np.ndarray):
    # dict of dict with values being the parameters
    blocks = []
    block_names = []
    terminal_block_flags = []
    for layer in input_sequence:
        for layer_name, layer_components in layer.items():
            for component_name, component_params in layer_components.items():
                flat_params = component_params.reshape(-1)
                windows = np.arange(start=0, stop=flat_params.shape[0], step=PARAMETER_BLOCK_SZ)
                for start in windows:
                    end = start + PARAMETER_BLOCK_SZ
                    if end < flat_params.shape[0]:
                        blocks.append(flat_params[start:end])
                    else:
                        pad_wanting = flat_params[start:]
                        padded = np.concatenate([pad_wanting, np.zeros(PARAMETER_BLOCK_SZ - pad_wanting.shape[0])])
                        blocks.append(padded)
                    block_names.append(component_name)
                    terminal_block_flags.append(True if end >= flat_params.shape[0] else False)
    return (block_names, blocks, terminal_block_flags)

def get_onehot_timestep_encoder(TIMESTEPS: Sequence[str]):
    return dict(zip(TIMESTEPS, list(np.identity(len(TIMESTEPS)))))


def encode_sample(x, y, max_prog_len: int, TIMESTEPS: Sequence[str], ARCH_LABELS: Sequence[str]):
    ONEHOT_TIMESTEP_ENCODER = get_onehot_timestep_encoder(TIMESTEPS)
    if type(x) == np.array:
        x = x.reshape(-1)
    layer_names = [list(i.keys())[0] for i in x]
    architecture = encode_architecture(layer_names, max_prog_len, ARCH_LABELS=ARCH_LABELS)
    block_names, blocks, terminal_block_flags = block_params(x)
    timesteps = []
    for block_name, block, terminal_block_flag in zip(block_names, blocks, terminal_block_flags):
        encoded_block_name = ONEHOT_TIMESTEP_ENCODER[block_name]
        timestep = np.concatenate([encoded_block_name, np.array([int(terminal_block_flag)]), block, architecture])
        timesteps.append(timestep)
    enc_x = np.stack(timesteps)
    return enc_x, y


def decode_sample(x, TIMESTEPS: Sequence[str], max_prog_len=None, PARAMETER_BLOCK_SZ=None):
    TIMESTEP_DECODER = dict(zip(range(len(TIMESTEPS)), TIMESTEPS))
    if max_prog_len:
        block_type, terminal, block, architecture = np.argmax(x[:, :len(TIMESTEPS)], axis=1), x[:, len(TIMESTEPS)], x[:, len(TIMESTEPS)+1: -max_prog_len], x[:, -max_prog_len:]
    else:
        block_type, terminal, block, architecture = np.argmax(x[:, :len(TIMESTEPS)], axis=1), x[:, len(TIMESTEPS)], x[:, len(TIMESTEPS)+1: len(TIMESTEPS)+1+PARAMETER_BLOCK_SZ], x[:, len(TIMESTEPS)+1+PARAMETER_BLOCK_SZ:]
    
    NUM_BLOCKS = block_type.shape[0]

    parameters = []
    block_names = []
    acc = [] # store block here until we reach the terminal
    for layer in range(NUM_BLOCKS):
        acc.append(block[layer, :])
        if terminal[layer] == 1:
            parameters.append(np.concatenate(acc))
            block_names.append(TIMESTEP_DECODER[block_type[layer]])
            acc = []
    
    model = []
    layer_types = decoder_architecture(architecture[0, :])

    for layer in layer_types:
        d = dict([(block_names.pop(0), parameters.pop(0)), (block_names.pop(0), parameters.pop(0))])
        model.append({layer: d})
    return model

def decode_timesteps(x, TIMESTEPS: Sequence[str], batch=0):
    ONEHOT_TIMESTEP_ENCODER = get_onehot_timestep_encoder(TIMESTEPS)
    TIMESTEP_TOKEN_SIZE = list(ONEHOT_TIMESTEP_ENCODER.values())[0].shape[0]
    this_batch = x[batch, :, :]
    s = []
    terminals = []
    for timestep in range(this_batch.shape[0]):
        index = np.array(this_batch[timestep, : TIMESTEP_TOKEN_SIZE]).argmax()
        #print(list(ONEHOT_TIMESTEP_ENCODER.keys())[index])
        s += [list(ONEHOT_TIMESTEP_ENCODER.keys())[index]]
        terminals += [bool(np.array(this_batch[timestep, TIMESTEP_TOKEN_SIZE]).item())]
    return s, terminals

def test_flat_match(src, B):
    for a, b in zip(src, B):
        for key in a.keys():
            a_child = a[key]
            b_child = b[key]
            for child_key in a_child.keys():
                ref = a_child[child_key].reshape(-1)
                param_match = (ref == b_child[child_key].reshape(-1)[:ref.shape[0]]).all()
                if not param_match:
                    print("not matching in layer {a} param {key}")
                    return False
    return True




# class ParameterEncoderWrapper:
#     def __init__(self, src, max_prog_len: int) -> None:
#         self.src = src
#         self.max_prog_len = max_prog_len
#     def __len__(self):
#         return len(self.src)
#     def __getitem__(self, idx):
#         x,y = self.src.__getitem__(idx)
#         enc_x, y = encode_sample(x, y, max_prog_len=self.max_prog_len)
#         return enc_x, y







# Compiles to an empty set of parameters
# PROGRAM_START
# Map LAM_GT indices NA v1
# Map LAM_GT v1 NA v2
# Map LAM_GE v2 NA v3
# Map LAM_LT v3 NA v4
# Map LAM_GE v4 NA v5
# Map LAM_LE v5 NA v6
# PROGRAM_END


#%%

if __name__ == "__main__":
    pth = '.data/iTracr_dataset/'
    dataset = StreamReader(pth)

    it = iter(dataset)
    x,y = next(it)


    layer_names = []
    layer_components = []
    it = iter(dataset)
    for _ in range(1000):
        x,y = next(it)
        x = x.reshape(-1)
        if x.shape == ():
            print("empty sample")
            print(y)
            print(type(x))
            print(x.item()['MLP'])
            print("x:")
            print(x)
            continue
        layer_names += [list(i.keys())[0] for i in x]
        for j in x:
            layer_components += sum([list(i.keys()) for i in j.values()], [])

    print("Dataset balance: ")
    print(pd.Series(layer_names).value_counts())
    # print(pd.Series(layer_components).value_counts())

    # HEAD    383036
    # MLP     293807


    # Test encoder-decoder
    it = iter(dataset)
    for _ in range(len(dataset)):
        x,y = next(it)
        x = x.reshape(-1)
        
        enc_x, y = encode_sample(x, y, max_prog_len=15)
        dec_x = decode_sample(enc_x, PARAMETER_BLOCK_SZ=PARAMETER_BLOCK_SZ)
        matching = test_flat_match(x, dec_x)
        if not matching:
            print("TEST - MATCHING PARAM ENCODER - FAILED")
            print(y)




# %%


#   h m?    Select indices indices PRED_LEQ v1
#   h m    SelectorWidth v1 NA NA v2
#   h      Aggregate v1 indices NA v4
#     m    Map LAM_ADD v4 NA v5
#     m    SequenceMap LAM_ADD v2 v2 v3
#   h m?   Select v5 v5 PRED_LT v6
#          Aggregate v6 v3 NA v7
