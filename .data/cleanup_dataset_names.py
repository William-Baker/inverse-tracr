#%%
import os
from tqdm import tqdm


# trainer.verbose_step(trainer.state, sample, 0)

pth = '.data/iTracr_dataset_train/'
files = os.listdir(pth)
for idx, file in tqdm(enumerate(sorted(files))):
    targ = pth + str(idx).zfill(8) + '.npz'
    if os.path.isfile(targ):
        raise Exception()
    os.rename(pth + file, targ)
#%%