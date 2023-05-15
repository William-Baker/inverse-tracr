#%%
from datasetv4 import craft_dataset
gen, OP_VOCAB, VAR_VOCAB = craft_dataset()
g = gen()

# while True:
weights, program = next(g)

#%%

# test maximum time required to generate a program
from datasetv4 import craft_dataset
import time
import numpy as np
from tqdm import tqdm
gen, OP_VOCAB, VAR_VOCAB = craft_dataset(ops_range=(20,20))
g = gen()
weights, program = next(g)

times = []
for i in tqdm(range(50)):
	start = time.time()
	weights, program = next(g)
	end = time.time()
	times.append(end - start)
print(f"max time: {max(times)}")
print(f"mean time: {np.mean(times)}")
print(f"median time: {np.median(times)}")


#%%
