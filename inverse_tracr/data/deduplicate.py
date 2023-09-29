from zipfile import ZipFile, ZIP_DEFLATED
import cloudpickle
from tqdm import tqdm
from random import shuffle

from inverse_tracr.data.parallelzipfile import ParallelZipFile

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


deduplicated = set()


out = ZipFile(file='.data/deduplicated-v6.zip', mode='a', compression=ZIP_DEFLATED, compresslevel=4)



input_zips = ['.data/output.zip', '.data/output_2.zip', '.data/output_3.zip', '.data/output_4.zip',
              '.data/output_first_gen.zip']
max_name = 0

for input_zip in input_zips:

    print('Initializing reader...')
    df = ZipStreamReader(input_zip)
    print("Done initializing. Total nr of files:", len(df.files))

    def read_file(idx):
        try:
            return df[idx]
        except EOFError:
            return None

    NUM_SAMPLES = len(df.files)
    out_names = [str(x).zfill(9) + '.pkl' for x in range(max_name, max_name+NUM_SAMPLES)]
    shuffle(out_names)
    out_i = 0

    max_name += NUM_SAMPLES


    print("Iterating through files...")
    for idx in tqdm(range(NUM_SAMPLES), desc='load and save deduped', disable=DISABLE_TQDM):
        sample = read_file(idx)
        if sample is None:
            continue

        x, program = sample
        b = program.tobytes()
        if b not in deduplicated:
            deduplicated.add(b)

            # write to zip
            pickled = cloudpickle.dumps(sample)
            out.writestr(out_names[out_i], pickled)
            out_i += 1
                

    print("Original length:", NUM_SAMPLES)
    print("Length of deduplicated dataset:", len(deduplicated))
