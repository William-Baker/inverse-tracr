#%%
#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 16 -A KRUEGER-SL3-CPU --qos=INTR
# srun -t 10:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 76 -A KRUEGER-SL3-CPU --pty=bash
from data.parallelzipfile import ParallelZipFile
from zipfile import ZipFile, ZIP_DEFLATED
import cloudpickle
import cloudpickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

class ZipStreamReader:
    def __init__(self, dir: str) -> None:
        self.zipfile = ParallelZipFile(file=dir, mode='r')
        self.files = sorted(self.zipfile.namelist())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.zipfile.read(self.files[idx])
        params, program = cloudpickle.loads(x)
        return params, program


print('Initializing reader...')
df = ZipStreamReader('.data/iTracr_dataset_v4.zip')
total = 0
print("Done initializing. Total nr of files:", len(df.files))


def print_info(d):
    dupes = sum(list(d.values()))
    print()
    print(f"Programs duplicated: {(100 * dupes / total):.3f}%  (total {dupes})")
    for k, v in sorted(list(d.items()), key=lambda x: x[0]):
        print(f"Program length {k}: {(100 * v / dupes):.3f}%  (total {v}).")


def read_file(idx):
    try:
        return df[idx]
    except EOFError:
        return None


#NUM_SAMPLES = 10**5
NUM_SAMPLES = len(df.files)
deduplicated = set()



out = ZipFile(file='.data/deduplicated.zip', mode='w', compression=ZIP_DEFLATED, compresslevel=4)
out_names = [str(x).zfill(9) + '.pkl' for x in range(NUM_SAMPLES)]
shuffle(out_names)
out_i = 0

with ThreadPoolExecutor() as executor:
    futures = []
    for idx in tqdm(range(NUM_SAMPLES), desc='collecting futures'):
        future = executor.submit(read_file, idx)
        futures.append(future)

    for i, future in tqdm(enumerate(as_completed(futures)), desc='scheduling set checks', total=NUM_SAMPLES):
        res = future.result()
        if res is None:
            continue
        x, program = res
        b = program.tobytes()
        if b not in deduplicated:
            deduplicated.add(b)

            # write to zip
            b = cloudpickle.dumps(res)
            out.writestr(out_names[out_i], b)
            out_i += 1
        del x
            



print("Original length:", NUM_SAMPLES)
print("Length of deduplicated dataset:", len(deduplicated))
