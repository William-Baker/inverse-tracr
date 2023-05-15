#%%
from datasetv4 import craft_dataset
gen, OP_VOCAB, VAR_VOCAB = craft_dataset()
g = gen()

while True:
    weights, program = next(g)

#%%

# test maximum time required to generate a program
from dataset import craft_dataset
import time
import numpy as np
gen, OP_VOCAB, VAR_VOCAB = craft_dataset()
g = gen()
weights, program = next(g)

times = []
for i in range(10000):
	start = time.time()
	weights, program = next(g)
	end = time.time()
	times.append(end - start)
print(f"max time: {max(times)}")
print(f"mean time: {np.mean(times)}")
print(f"median time: {np.median(times)}")


#%%
import multiprocessing
def a_function(ret_value):
    ret_value.value = 3.145678

ret_value = multiprocessing.Value("d", 0.0, lock=False)
reader_process = multiprocessing.Process(target=a_function, args=[ret_value])
reader_process.start()
reader_process.join()

print(ret_value.value)

#%%
