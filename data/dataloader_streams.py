#%%
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from os import makedirs
import numpy as np
from tqdm import tqdm
class StreamWriter:
    def __init__(self, dir:str, torch_dataset: TorchDataset, id_N = None, start_idx= None, start_offset=0) -> None:
        """
        id_N: (id, N) - id: int [0, N], N: int
        """
        self.dataset = torch_dataset
        self.dir = dir
        makedirs(self.dir, exist_ok=True)
        self.samples_to_write = len(torch_dataset)
        self.startidx = 0
        if id_N is not None:
            id, N = id_N
            assert id < N
            self.samples_to_write = self.samples_to_write // N
            self.startidx = self.samples_to_write * id + start_offset
        if start_idx:
            self.startidx = start_idx + start_offset
    
    def __write_parallel__(self, num_threads: int):
        assert num_threads > 0
        dataloader = TorchDataLoader(self.dataset, batch_size=1, num_workers=num_threads, prefetch_factor=2 if num_threads else None)
        it = iter(dataloader)
        for idx in tqdm(range(self.startidx, self.samples_to_write + self.startidx), desc=f'Writing samples:'):
            x, y = next(it)#self.dataset.__getitem__(idx)
            np.savez(self.dir + str(idx).zfill(8), x=x, y=y)
    
    def __write_serial__(self):
        it = iter(self.dataset)
        for idx in tqdm(range(self.startidx, self.samples_to_write + self.startidx), desc=f'Writing samples:'):
            x, y = next(it)#self.dataset.__getitem__(idx)
            np.savez(self.dir + str(idx).zfill(8), x=x, y=y)


    def write_samples(self, num_threads=1):
        if num_threads >= 1:
            self.__write_parallel__(num_threads)
        else:
            self.__write_serial__()

from os import listdir
class StreamReader:
    def __init__(self, dir:str) -> None:
        self.dir = dir
        self.files = sorted(listdir(dir))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        loaded = np.load(self.dir + self.files[idx], allow_pickle=True)
        return loaded['x'].squeeze(), loaded['y'].squeeze()

import scipy
import pickle

class SparseConverter:
    def __init__(self, dir:str, out_dir:str) -> None:
        self.dir = dir
        self.out_dir = out_dir
        self.files = sorted(listdir(dir))
        makedirs(self.out_dir, exist_ok=True)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        try:
            loaded = np.load(self.dir + self.files[idx], allow_pickle=True)
            x,y = loaded['x'].squeeze(), loaded['y'].squeeze()
            # x,y = scipy.sparse.csc_matrix(x), scipy.sparse.csc_matrix(y)
            # scipy.sparse.save_npz(self.dir + self.files[idx], x=x, y=y)
            # return idx
            with open(self.out_dir + str(idx).zfill(8) + '.pickle', 'wb') as handle:
                pickle.dump((x,y), handle)#, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as E:
            pass

            
    

#from data.parallelzipfile import ParallelZipFile as ZipFile
from zipfile import ZipFile
from io import BytesIO
# class ZipStreamReader:
#     def __init__(self, dir:str) -> None:
#         self.zip = ZipFile(file=dir, mode='r')
#         self.files = sorted(self.zip.namelist()[1:])
#     def __len__(self):
#         return len(self.files)
#     def __getitem__(self, idx):
#         x = self.zip.read(self.files[idx])
#         loaded = np.load(BytesIO(x), allow_pickle=True)
#         return loaded['x'].squeeze(), loaded['y'].squeeze()

import fcntl
class Locker:
    def __enter__ (self):
        self.fp = open(".data/zip_lock.lck", 'w')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

from threading import Lock
class ZipStreamReader:
    mutex = Lock()
    def __init__(self, dir:str) -> None:
        self.zip = ZipFile(file=dir, mode='r')
        self.files = sorted(self.zip.namelist()[1:])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        with Locker():
            x = BytesIO(self.zip.read(self.files[idx]))
            loaded = np.load(x, allow_pickle=True)
        return loaded['x'].squeeze(), loaded['y'].squeeze()


#%%
import os, subprocess, sys
mode = 'params' # programs
shuffled = True
if __name__ == "__main__":
    # from os import chdir
    # chdir('../')
    if mode == 'programs':
        from data.program_dataloader import TorchProgramDataset
        dataset = TorchProgramDataset(100000, shuffled_inputs=shuffled)
        pth = '.data/p2p_dataset/' if shuffled else '.data/p2p_dataset_unshuffled/'
        sw = StreamWriter(pth, dataset)
        sw.write_samples(num_threads=20)
    else:
        # from parameter_program_dataloader import TorchParameterProgramDataset
        # dataset = TorchProgramDataset( no_samples = 10000, generator_backend='bounded')
        # pth = '.data/iTracr_dataset/'
        # sw = StreamWriter(pth, dataset)
        # sw.write_samples(num_threads=1)
        N = 20
        samples = 1000000
        offset  = 900000
        for i in range(N):
            #os.chdir('../')
            cmd = f"python generate_parameter_partial_dataset.py -off {offset} -s {samples} -pn {N} -idn {i}"
            #os.system(f"python data/generate_parameter_partial_dataset.py -s {samples} -pn {N} -idn {i}")
            #pid = subprocess.Popen([sys.executable, cmd])
            #p = subprocess.Popen(["start", "cmd", "/k", cmd], shell = True)#os.system("start /wait cmd /c {command}")
            #os.system(f"cmd /c {cmd}")
            #subprocess.call(cmd, shell=True)
            subprocess.Popen(cmd, shell=True)
            # print(cmd)

            

    # reader = StreamReader(pth)

    # x,y = reader.__getitem__(1)


    # dataloader = TorchDataLoader(reader, batch_size=1, num_workers=8, prefetch_factor=2)
    # it = iter(dataloader)

    # for i in range(5000):
    #     x,y = next(it)
    
#%%