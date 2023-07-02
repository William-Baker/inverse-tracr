#%%
import argparse
import sys

from data.dataloader_streams import StreamReader, StreamWriter
from data.parameter_program_dataloader import TorchParameterProgramDataset


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", required=True, help="number of samples", type=int)
ap.add_argument("-pn", "--proc_num", required=True, help="number of processes", type=int)
ap.add_argument("-idn", "--id_number", required=True, help="program call id", type=int)
ap.add_argument("-off", "--offset", required=False, help="intial n ofset", type=int, default=0)
ap.add_argument("-vmin", "--vocab_min", required=False, help="lower bound of vocab size", type=int, default=6)
ap.add_argument("-vmax", "--vocab_max", required=False, help="upper bound of vocab size", type=int, default=6)
ap.add_argument("-nmin", "--number_min", required=False, help="lower bound of numeric range size", type=int, default=6)
ap.add_argument("-nmax", "--number_max", required=False, help="upper bound of numeric range size", type=int, default=6)
ap.add_argument("-num", "--numeric_inputs", required=False, help="whether to generate samples with numeric inputs", type=bool, default=False)


args = ap.parse_args()
# args = argparse.Namespace(proc_num=1, id_number=0, samples=10000, offset=900000)


dataset = TorchParameterProgramDataset(15, no_samples = args.samples, generator_backend='bounded', bounded_timeout_multiplier=1,
                                       vocab_size_range=(args.vocab_min, args.vocab_max),
                                       numeric_range=(args.number_min, args.number_max),
                                       numeric_inputs_possible=args.numeric_inputs
                                    )
pth = '.data/iTracr_dataset_v2_test/'
sw = StreamWriter(pth, dataset, id_N=(args.id_number, args.proc_num), start_offset=args.offset)
sw.write_samples(num_threads=0)
# %%
