#%%

from utils.dataloaders import ProgramDataset
from tqdm import tqdm


samples = 10000000

dataset = ProgramDataset(30, samples)


it = iter(dataset)
xs = []
ys = []


for i in tqdm(range(1000000)):
    x, y = next(it)
    xs.append(x)
    ys.append(y)

#%%


import pandas as pd

df = pd.DataFrame({'x': xs, 'y': ys})
df.to_pickle('program_dataset2.pkl')


#%%

df = pd.concat(
    [pd.read_pickle(f"program_dataset{x}.pkl") for x in range(1, 8)]
)


# %%


#%%

meta = pd.Series({'OP_VOCAB': list(dataset.op_encoder.keys()), 'VAR_VOCAB': list(dataset.var_encoder.keys()), 'PROG_LEN': dataset.prog_len})
meta.to_pickle('program_meta.pkl')

# %%
