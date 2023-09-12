#%%
from data.parallelzipfile import ParallelZipFile as ZipFile
import cloudpickle

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
df = ZipStreamReader('.data/iTracr_dataset_v2_train.zip')

#df = ZipStreamReader('.data/dltest.zip')
#df = ZipStreamReader('fixed.zip')
print(len(df))
it = iter(df)
# for i in range(200):
#     next(it)
sizes = []
for i in range(5000):
    # next(it)
    # next(it)
    # next(it)
    x,y = next(it)

    # print(len(df))
    # from data.dataloaders import ProgramEncoder
    # print(x.keys())
    # prog_enc = ProgramEncoder(15)
    # print(prog_enc.decode_pred(y))
    
    sizes.append(y.shape[0])

import pandas as pd
print(pd.Series(sizes).value_counts())
    
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
from data.dataset import *
from random import choices
from tqdm import tqdm
rejections = 0

def build_program_of_length(vocab, numeric_range: tuple, MIN_PROG_LENGTH: int, MAX_PROG_LENGTH: int):
    global rejections
    program_length = 0
    program, actual_ops = None, None
    n_ops = int(1.8 * MAX_PROG_LENGTH)
    while not (MIN_PROG_LENGTH < program_length <= MAX_PROG_LENGTH):
        rejections += 1
        ops = generate_ops(n_ops, vocab, numeric_range)
        program, actual_ops = compile_program(ops)
        program_length = len(actual_ops)
    rejections -= 1
    return program, actual_ops

def choose_vocab_and_ops(ops_range: tuple, vocab_size_range: tuple, numeric_inputs_possible: bool, small_v_large_bias=1):
    """_
        small_v_large_bias (int, optional): must be >0, 0.5 would make large values half as likley
        2 would make large values twice as likely, 1 is uniform. Defaults to 1.
    """
    possible_ops = np.arange(ops_range[0], ops_range[1]+1, dtype=int)
    weights = np.linspace(1, small_v_large_bias, num=possible_ops.shape[0])
    n_ops = choices(possible_ops, weights, k=1)[0]
    # n_ops = np.random.choice(possible_ops, 1, weights)
    # n_ops = randint(*ops_range)
    vocab_size = randint(*vocab_size_range)
    numeric_inputs = choice([True, False]) if numeric_inputs_possible else False
    vocab = gen_vocab(vocab_size, prefix='t', numeric=numeric_inputs)
    return n_ops, vocab

n_ops, vocab = choose_vocab_and_ops(ops_range=(4, 15), vocab_size_range=(4, 15), numeric_inputs_possible=True, small_v_large_bias=2)


# program, actual_ops = build_program_of_length(n_ops, vocab, numeric_range, TARGET_PROGRAM_LENGTH)

l = []
for i in tqdm(range(100)):
    ops_range = (4,15)
    n_ops, vocab = choose_vocab_and_ops(ops_range=ops_range, vocab_size_range=(3, 15), numeric_inputs_possible=True, small_v_large_bias=3)
    program, actual_ops = build_program_of_length(vocab, (3,15), MIN_PROG_LENGTH=max(2, n_ops-2), MAX_PROG_LENGTH=min(n_ops+2, ops_range[1]))
    
    l.append(len(actual_ops))
print(rejections)
