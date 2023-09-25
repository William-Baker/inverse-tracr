#%%
#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 16 -A KRUEGER-SL3-CPU --qos=INTR
# srun -t 10:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 76 -A KRUEGER-SL3-CPU --pty=bash
from data.parallelzipfile import ParallelZipFile
from zipfile import ZipFile, ZIP_DEFLATED
import cloudpickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

DISABLE_TQDM = False


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


def read_file(idx):
    try:
        return df[idx]
    except EOFError:
        return None


NUM_SAMPLES = len(df.files)
deduplicated = set()


out = ZipFile(file='.data/deduplicated.zip', mode='w', compression=ZIP_DEFLATED, compresslevel=4)
out_names = [str(x).zfill(9) + '.pkl' for x in range(NUM_SAMPLES)]
shuffle(out_names)
out_i = 0


with ThreadPoolExecutor() as executor:
    futures = []
    for idx in tqdm(range(NUM_SAMPLES), desc='collecting futures', disable=DISABLE_TQDM):
        future = executor.submit(read_file, idx)  # probably better to use map here
        futures.append(future)

    for i, future in tqdm(enumerate(as_completed(futures)), desc='scheduling set checks', total=NUM_SAMPLES, disable=DISABLE_TQDM):
        res = future.result()
        if res is None:
            continue
        x, program = res
        b = program.tobytes()
        if b not in deduplicated:
            deduplicated.add(b)

            # write to zip
            pickled = cloudpickle.dumps(res)
            out.writestr(out_names[out_i], pickled)
            out_i += 1
            

print("Original length:", NUM_SAMPLES)
print("Length of deduplicated dataset:", len(deduplicated))