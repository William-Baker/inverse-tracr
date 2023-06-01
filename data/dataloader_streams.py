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
            np.savez(self.dir + str(idx).zfill(7), x=x, y=y)
    
    def __write_serial__(self):
        it = iter(self.dataset)
        for idx in tqdm(range(self.startidx, self.samples_to_write + self.startidx), desc=f'Writing samples:'):
            x, y = next(it)#self.dataset.__getitem__(idx)
            np.savez(self.dir + str(idx).zfill(7), x=x, y=y)


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
        loaded = np.load(self.dir + self.files[idx])
        return loaded['x'].squeeze(), loaded['y'].squeeze()



#%%
import os, subprocess, sys
mode = 'params' # programs
shuffled = True
if __name__ == "__main__":
    # from os import chdir
    # chdir('../')
    if mode == 'programs':
        from program_dataloader import TorchProgramDataset
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
        N = 10
        samples = 100000
        for i in range(10):
            cmd = f"python data/generate_parameter_partial_dataset.py -off 800 -s {samples} -pn {N} -idn {i}"
            #os.system(f"python data/generate_parameter_partial_dataset.py -s {samples} -pn {N} -idn {i}")
            #pid = subprocess.Popen([sys.executable, cmd])
            #p = subprocess.Popen(["start", "cmd", "/k", cmd], shell = True)#os.system("start /wait cmd /c {command}")
            #os.system(f"cmd /c {cmd}")
            #subprocess.call(cmd, shell=True)
            #subprocess.Popen(cmd, shell=True)
            print(cmd)

            

    # reader = StreamReader(pth)

    # x,y = reader.__getitem__(1)


    # dataloader = TorchDataLoader(reader, batch_size=1, num_workers=8, prefetch_factor=2)
    # it = iter(dataloader)

    # for i in range(5000):
    #     x,y = next(it)
    
#%%