
#%%


from torch.utils.data import DataLoader
from utils.dataloaders import ProgramDataset

dataset = ProgramDataset(30)
train_dataloader = DataLoader(dataset, batch_size=32, collate_fn=ProgramDataset.collate_fn, num_workers=8, prefetch_factor=2, pin_memory=True)





#%%
