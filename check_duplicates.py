#%%
#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 16 -A KRUEGER-SL3-CPU --qos=INTR
from data.parallelzipfile import ParallelZipFile as ZipFile
import cloudpickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from tqdm import tqdm
from collections import defaultdict
from random import sample, shuffle

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
# df = ZipStreamReader('.data/iTracr_dataset_v2_train.zip')
#df = ZipStreamReader('.data/iTracr_standard_20M.zip')
#df = ZipStreamReader('.data/sequential_fixed.zip')
# df = ZipStreamReader('.data/output.zip')
# df = ZipStreamReader('.data/iTracr_dataset_v2_train_v2.zip')
#df = ZipStreamReader('.data/iTracr_dataset_v3.zip')
df = ZipStreamReader('.data/iTracr_dataset_v4.zip')



duplicates = defaultdict(lambda: 0)
total = 0

def p(d):
    dupes = sum(list(d.values()))
    print(f"{100 * dupes / total}% duplicated")
    for k,v in sorted(list(d.items()), key=lambda x: x[0]):
        print(f"{k}: {v} \t {100 * v / dupes}%")

"""
print(len(df.files))
progs = []
# method is pretty slow since sequeuntially reading
for idx, entry in tqdm(enumerate(df.files), desc='reading from dir'):
    try:
        x, y = df.__getitem__(idx)
        total += 1
        for other in progs:
            # try:
            if other.shape==y.shape and (other == y).all():
                duplicates[y.shape[0]] += 1
                p(duplicates)
                break
            # except:
            #     pass
        progs.append(y)
    except EOFError:
        pass
"""

def read_file(idx):
    try:
        y = df.__getitem__(idx)[1]
    except EOFError:
        return None
    return y

random_sample = sample(df.files, 100000)
#random_sample = list(df.files)
#shuffle(random_sample)
print(len(df.files))
progs = defaultdict(lambda: [])
with ThreadPoolExecutor() as executor:
    futures = []
    for idx, entry in tqdm(enumerate(random_sample), desc='reading from dir'):
        future = executor.submit(read_file, idx)
        futures.append(future)

    for i, future in tqdm(enumerate(as_completed(futures)), desc='scheduling set checks', total=len(df)):
        y = future.result()
        if y is None:
            continue
        total += 1
        duplicated = False
        for other in progs[y.shape[0]]:
            # try:
            if (other == y).all():
                duplicates[y.shape[0]] += 1
                duplicated = True
                break
        if not duplicated:
            progs[y.shape[0]].append(y)

        if (i % 1000) == 0:
            p(duplicates)


"""
class custom_set:
    def __init__(self) -> None:
        self.set = defaultdict(lambda: [])
    def contains(self, x, hash, eq):
        if hash in self.set:
            conflict = self.set[hash]
            for i in conflict:
                if eq(i, x):
                    # print(i)
                    # print(x)
                    # input()
                    return True
        return False

    def add(self, x, hash):
        self.set[hash].append(x)




def read_file(idx):
    try:
        y = df.__getitem__(idx)[1]
    except:
        return None
    y.flags.writeable = False
    return y

master_set = custom_set()
duplicates = defaultdict(lambda: 0)
total = 0

# def set_add(y):
#     if y in master_set:
#         duplicates[y.shape[0]] += 1
#     else:
#         master_set.add(y)

def set_add(y):
    global total
    h = hash(y.data.tobytes())
    if master_set.contains(y, h, eq=lambda x, y: (x==y).all()):
        duplicates[y.shape[0]] += 1
    else:
        master_set.add(y, h)
    total += 1

# with ThreadPoolExecutor() as executor:
#     futures = []
#     for idx, entry in tqdm(enumerate(df.files), desc='reading from dir'):
#         future = executor.submit(read_file, idx)
#         futures.append(future)

#     set_futures = []
#     with ThreadPoolExecutor() as set_executor:
#         for future in tqdm(as_completed(futures), desc='scheduling set checks', total=len(df)):
#             y = future.result()
#             set_future = set_executor.submit(set_add, y)
#             set_futures.append(set_future)
#     for i, future in tqdm(enumerate(as_completed(set_futures)), desc='await set', total=len(df)):
#         if (i % 50000):
#             print(duplicates)



for idx, entry in tqdm(enumerate(df.files), desc='reading from dir'):
    y = read_file(idx)
    if y is not None:
        set_add(y)
        if (idx % 50000):
            # print(dict(duplicates))
            p(duplicates)

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
"""
