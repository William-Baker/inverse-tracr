#%%

# from zipfile import ZipFile, ZIP_DEFLATED
# import cloudpickle

# zip = ZipFile(file='cp_dataset_train_w.zip', mode='r', compression=ZIP_DEFLATED, compresslevel=9)
# out = ZipFile(file='cp_dataset_train_w_out.zip', mode='w', compression=ZIP_DEFLATED, compresslevel=9)
# files = sorted(zip.namelist())

# new_params = None
# for idx in range(len(files)):
#     x = zip.read(files[idx])
#     try:
#         x,y = cloudpickle.loads(x)
#         new_params = []
#         for line in x:
#             if 'key' in list(line.values())[0]:
#                 new_params.append({'MHA': list(line.values())[0]})
#             else:
#                 new_params.append({'MLP': list(line.values())[0]})
#         b = cloudpickle.dumps((new_params,y))
#         out.writestr(files[idx], b)
#     except:
#         pass
# zip.close()
# out.close()


#%%

# from zipfile import ZipFile, ZIP_DEFLATED
# import cloudpickle
# from tqdm import tqdm
# from jax import tree_map
# import numpy as np

# zip = ZipFile(file='cp_dataset_train_w.zip', mode='r', compression=ZIP_DEFLATED, compresslevel=9)
# out = ZipFile(file='cp_dataset_train_w_out.zip', mode='a', compression=ZIP_DEFLATED, compresslevel=9)
# in_files = set(sorted(zip.namelist()))
# out_files = set(sorted(out.namelist()))
# files = list(in_files - out_files)

# new_params = None
# for idx in tqdm(range(len(files))):
#     try:
#         x = zip.read(files[idx])
#         x,y = cloudpickle.loads(x)
#         new_params = tree_map(lambda x: np.array(x), x)
#         b = cloudpickle.dumps((new_params,y))
#         out.writestr(files[idx], b)
#     except:
#         pass
# zip.close()
# out.close()

#%%

# from zipfile import ZipFile, ZIP_DEFLATED
# import cloudpickle

# zip = ZipFile(file='cp_dataset_train_w.zip', mode='r', compression=ZIP_DEFLATED, compresslevel=9)
# out = ZipFile(file='cp_dataset_train_w_out.zip', mode='w', compression=ZIP_DEFLATED, compresslevel=9)
# files = sorted(zip.namelist())

# new_params = None
# for idx in range(len(files)):
#     x = zip.read(files[idx])
#     try:
#         x,y = cloudpickle.loads(x)
#         new_params = []
#         for line in x:
#             if list(line.keys())[0] == 'MHA':
#                 new_params.append({'MHA': list(line.values())[0]})
#             if 'key' in list(line.values())[0]:
#                 new_params.append({'MHA': list(line.values())[0]})
#             else:
#                 new_params.append({'MLP': list(line.values())[0]})
#         b = cloudpickle.dumps((new_params,y))
#         out.writestr(files[idx], b)
#     except:
#         pass
# zip.close()
# out.close()


#%% ================= make file names sequential ======================================

from zipfile import ZipFile, ZIP_DEFLATED
import cloudpickle
from tqdm import tqdm
from jax import tree_map
import numpy as np
from io import BytesIO

inp = ZipFile(file='.data/iTracr_dataset_v2_train.zip', mode='r', compression=ZIP_DEFLATED, compresslevel=9)
out = ZipFile(file='.data/iTracr_dataset_v2_train_sequential_pickle.zip', mode='a', compression=ZIP_DEFLATED, compresslevel=9)
in_files = sorted(inp.namelist())
out_files = [str(x).zfill(8) + '.pkl' for x in range(len(in_files))]

new_params = None
for in_file, out_file in tqdm(list(zip(in_files, out_files))):
    try:
        x = inp.read(in_file)
        loaded = np.load(BytesIO(x), allow_pickle=True)
        x,y = loaded['x'].squeeze(), loaded['y'].squeeze()
        assert len(x.shape) > 0
        b = cloudpickle.dumps((x,y))
        out.writestr(out_file, b)
    except:
        pass
inp.close()
out.close()





#%%

# import numpy as np


# PARAMETER_BLOCK_SZ = 512

# from typing import Sequence
# def encode_architecture(layer_types: Sequence[str], max_prog_length: int, ARCH_LABELS: Sequence[str]):
#     ARCHITECTURE_ENCODER = dict(zip(ARCH_LABELS, range(len(ARCH_LABELS))))
#     encoding = np.zeros((max_prog_length*3, len(ARCHITECTURE_ENCODER)))
#     for layer_no, layer_name in enumerate(layer_types):
#         encoding[layer_no, ARCHITECTURE_ENCODER[layer_name]] = 1
#     return encoding.reshape(-1)

# def decoder_architecture(encoding: np.ndarray, ARCH_LABELS: Sequence[str]):
#     ARCHITECTURE_DECODER = dict(zip(range(len(ARCH_LABELS)), ARCH_LABELS))
#     encoding = encoding.reshape(-1, len(ARCHITECTURE_DECODER))
#     layer_types = [ARCHITECTURE_DECODER[idx] for idx in encoding.argmax(axis=1) if idx != 0]
#     return layer_types

# def block_params(input_sequence: np.ndarray):
#     # dict of dict with values being the parameters
#     blocks = []
#     block_names = []
#     terminal_block_flags = []
#     for layer in input_sequence:
#         for layer_name, layer_components in layer.items():
#             for component_name, component_params in layer_components.items():
#                 flat_params = component_params.reshape(-1)
#                 windows = np.arange(start=0, stop=flat_params.shape[0], step=PARAMETER_BLOCK_SZ)
#                 for start in windows:
#                     end = start + PARAMETER_BLOCK_SZ
#                     if end < flat_params.shape[0]:
#                         blocks.append(flat_params[start:end])
#                     else:
#                         pad_wanting = flat_params[start:]
#                         padded = np.concatenate([pad_wanting, np.zeros(PARAMETER_BLOCK_SZ - pad_wanting.shape[0])])
#                         blocks.append(padded)
#                     block_names.append(component_name)
#                     terminal_block_flags.append(True if end >= flat_params.shape[0] else False)
#     return (block_names, blocks, terminal_block_flags)


# from data.dataloader_streams import ZipPickleStreamReader

# df = ZipPickleStreamReader('cp_dataset_train_w.zip')



# max_prog_len = 15
# it = iter(df)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # next(it)
# # x,y = next(it)

# #%%
# from random import randint
# x,y = df.__getitem__(randint(0, len(df)))

# for i in range(0, len(df)):
#     x,y = df.__getitem__(i)
#     sample_prog_length = y.shape[0]
#     ammount_to_pad = max_prog_len + 2 - y.shape[0]
#     padding = np.zeros((ammount_to_pad, y.shape[1]))
#     y = np.concatenate((y,padding), axis=0)
#     y = y.astype(int)

#     loss_mask = np.ones((sample_prog_length))#, y.shape[1]))
#     loss_mask = np.concatenate((loss_mask,padding[:, 0]), axis=0)



#     TIMESTEPS = ['PAD', 'key', 'value', 'query', 'linear', 'linear_1', 'linear_2', 'PROGRAM_START', 'PROGRAM_END']
#     ONEHOT_TIMESTEP_ENCODER = dict(zip(TIMESTEPS, list(np.identity(len(TIMESTEPS)))))
#     # encode sample x,y prog
#     layer_names = [list(i.keys())[0] for i in x]
#     print(layer_names)
#     architecture = encode_architecture(layer_names, max_prog_len, ['PAD', 'MHA', 'MLP'])
#     block_names, blocks, terminal_block_flags = block_params(x)
#     timesteps = []
#     for block_name, block, terminal_block_flag in zip(block_names, blocks, terminal_block_flags):
#         encoded_block_name = ONEHOT_TIMESTEP_ENCODER[block_name]
#         timestep = np.concatenate([encoded_block_name, np.array([int(terminal_block_flag)]), block, architecture])
#         timesteps.append(timestep)
#     enc_x = np.stack(timesteps)
#     #return enc_x, y
# #%%

    
