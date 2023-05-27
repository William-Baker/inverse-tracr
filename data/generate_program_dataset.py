#%%

from utils.dataloaders import ProgramDataset
from tqdm import tqdm

import pandas as pd

samples = 100000

dataset = ProgramDataset(30, samples)


it = iter(dataset)
programs = []


for j in tqdm(range(samples // 1000), desc='Generating batches of 1000 programs, quit once you have enough'):
    for i in range(1000):
        x, y = next(it)
        #xs.append(x)
        programs.append(y)
    df = pd.DataFrame({'y': programs})
    df.to_pickle('.data/program_dataset.pkl')
    

#%%




#%%

df = pd.concat(
    [pd.read_pickle(f"program_dataset{x}.pkl") for x in range(1, 8)]
)


# %%


#%%

meta = pd.Series({'OP_VOCAB': list(dataset.op_encoder.keys()), 'VAR_VOCAB': list(dataset.var_encoder.keys()), 'PROG_LEN': dataset.prog_len})
meta.to_pickle('program_meta.pkl')

# %%
