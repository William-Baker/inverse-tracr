#%%
from inverse_tracr.data.parallelzipfile import ParallelZipFile as ZipFile
import cloudpickle
from random import randint

class ZipStreamReader:
    def __init__(self, dir:str) -> None:
        self.zip = ZipFile(file=dir, mode='r')
        self.files = sorted(self.zip.namelist())
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        x = self.zip.read(self.files[idx])
        # loaded = np.load(BytesIO(x), allow_pickle=True)
        x,y = cloudpickle.loads(x)
        return x, y

# df = ZipStreamReader('cp_dataset_train_w.zip')
#df = ZipStreamReader('.data/iTracr_dataset_v2_train.zip')
#df = ZipStreamReader('.data/iTracr_standard_20M.zip')
df = ZipStreamReader('.data/deduplicated-v6.zip')
#%%
import numpy as np
#df = ZipStreamReader('.data/dltest.zip')
#df = ZipStreamReader('fixed.zip')
print(len(df))
# for i in range(200):
#     next(it)
sizes = []
params = []
for i in range(500):
    try:
        idx = randint(0, len(df))
        x,y = df.__getitem__(idx)

        # from data.dataloaders import ProgramEncoder
        # print(x.keys())
        # prog_enc = ProgramEncoder(15)
        # print(prog_enc.decode_pred(y))
        params.append(np.sum([np.prod(list(v.values())[0].shape) for v in  [list(i.values())[0] for i in x]]))
        sizes.append(y.shape[0])
        if (i % 100) == 0:
            import pandas as pd
            print(pd.Series(sizes).value_counts())
    except:
        pass

#%%

print(np.min(params))
print(np.max(params))
print(np.mean(params))
    
#df.zip.close()

#%%

# from data.parallelzipfile import ParallelZipFile as ZipFile
# import cloudpickle

# class ZipStreamReader:
#     def __init__(self, dir:str) -> None:
#         self.zip = ZipFile(file=dir, mode='r')
#         self.files = sorted(self.zip.namelist())
#     def __len__(self):
#         return len(self.files)
#     def __getitem__(self, idx):
#         x = self.zip.read(self.files[idx])
#         # loaded = np.load(BytesIO(x), allow_pickle=True)
#         x,y = cloudpickle.loads(x)
#         return x, y

# #df = ZipStreamReader('cp_dataset_train_all.zip')

# df = ZipStreamReader('.data/dltest.zip')
# #df = ZipStreamReader('.data/fixed.zip')
# print(len(df))

#%%

# from os import  listdir
# import os
# from tqdm import tqdm
# dirs = listdir(".data/iTracr_dataset_v2_train")
# print(len(dirs))
# for i in tqdm(dirs):
#     os.remove(".data/iTracr_dataset_v2_train/" + i)

#%%
