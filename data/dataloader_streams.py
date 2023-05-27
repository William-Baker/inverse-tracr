from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from os import makedirs
import numpy as np
from tqdm import tqdm
class StreamWriter:
    def __init__(self, dir:str, torch_dataset: TorchDataset, id_N = None, start_idx= None) -> None:
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
            self.samples_to_write = self.samples_to_write // N
            self.startidx = self.samples_to_write * id
        if start_idx:
            self.startidx = start_idx
    

    def write_samples(self, num_threads=1):
        dataloader = TorchDataLoader(self.dataset, batch_size=1, num_workers=num_threads, prefetch_factor=2)
        it = iter(dataloader)
        for idx in tqdm(range(self.startidx, self.samples_to_write + self.startidx), desc=f'Writing samples:'):
            x, y = next(it)#self.dataset.__getitem__(idx)
            np.savez(self.dir + str(idx).zfill(7), x=x, y=y)

#%%
from os import chdir
chdir('../')
from data.program_dataloader import TorchProgramDataset
dataset = TorchProgramDataset(500)
sw = StreamWriter('.data/p2p_dataset/', dataset)
sw.write_samples(num_threads=4)

#%%
from os import listdir
class StreamReader:
    def __init__(self, dir:str) -> None:
        self.dir = dir
        self.files = sorted(listdir(dir))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        loaded = np.load(self.dir + self.files[idx])
        return loaded['x'], loaded['y']

reader = StreamReader('.data/p2p_dataset/')
x,y = reader.__getitem__(1)
# %%
