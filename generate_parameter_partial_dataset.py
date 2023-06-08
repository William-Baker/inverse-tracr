#%%
import argparse
import os
print(os.listdir())
import sys

from data.dataloader_streams import StreamReader, StreamWriter
from data.parameter_program_dataloader import TorchParameterProgramDataset


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="number of samples", type=int)
ap.add_argument("-pn", "--proc_num", required=True, help="number of processes", type=int)
ap.add_argument("-idn", "--id_number", required=True, help="program call id", type=int)
ap.add_argument("-off", "--offset", required=False, help="intiial n ofset", type=int, default=0)

args = ap.parse_args()
# args = argparse.Namespace(proc_num=1, id_number=0, samples=10000, offset=900000)


dataset = TorchParameterProgramDataset(15, no_samples = args.samples, generator_backend='bounded', bounded_timeout_multiplier=1)
pth = '.data/iTracr_dataset/'
sw = StreamWriter(pth, dataset, id_N=(args.id_number, args.proc_num), start_offset=args.offset)
sw.write_samples(num_threads=0)
# %%
