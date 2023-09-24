
#srun -t 12:0:0 --nodes 1 --cpus-per-task 3 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --pty bash
# source venv/bin/activate



from tqdm import tqdm
from collections import defaultdict
from data.parameter_program_dataloader import TorchParameterProgramDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from data.parameter_encoder import encode_sample
START_TOKEN = 'PROGRAM_START'
END_TOKEN = 'PROGRAM_END'

from data.encoded_dataloaders import craft_dataset, program_craft_generator_bounded, program_craft_generator_unbounded
from data.dataloaders import ProgramEncoder

vocab_range = (3, 15)
numeric_range = (3, 15)
numeric_inputs_possible = True

# dataset = TorchParameterProgramDataset(3, 15, generator_backend='bounded', bounded_timeout_multiplier=1,
#                                        vocab_size_range=vocab_range,
#                                        numeric_range=numeric_range,
#                                        numeric_inputs_possible=numeric_inputs_possible
#                                     )

(min_prog_len,max_prog_len) = 3,15
vocab_size_range = vocab_range
func = program_craft_generator_bounded
gen, OP_VOCAB, VAR_VOCAB = craft_dataset((min_prog_len,max_prog_len), func=func, timeout_multiplier=int(1).as_integer_ratio,
                                vocab_size_range=vocab_size_range, numeric_range=numeric_range, numeric_inputs_possible=numeric_inputs_possible)


prog_enc = ProgramEncoder(max_prog_len)



progs = []

duplicates = defaultdict(lambda: 0)
total = 0

def p(d):
    dupes = sum(list(d.values()))
    print(f"{100 * dupes / total}% duplicated")
    for k,v in d.items():
        print(f"{k}: {v} \t {100 * v / dupes}%")

"""
# V1 too slow since iterator is not fetched in parallel
it = iter(dataset)
for idx in tqdm(range(100000), desc='reading from dir'):
    try:
        x, y = next(it)
        for other in progs:
            try:
                if (other == y).all():
                    duplicates[y.shape[0]] += 1
                    total += 1
                    p(duplicates)
            except:
                pass
        progs.append(y)
    except EOFError:
        pass
"""


def get():
    x,y = gen()
    y = prog_enc.tokenise_program(y)
    return x,y

cores = int(len(os.sched_getaffinity(0)))
while True:
    set_futures = []
    with ThreadPoolExecutor(cores * 5) as set_executor:
        for future in tqdm(range(100), desc='scheduling set checks'):
            set_future = set_executor.submit(get)
            set_futures.append(set_future)
        for i, future in tqdm(enumerate(as_completed(set_futures)), desc='await set'):
            x,y = future.result()
            total += 1
            for other in progs:
                try:
                    if (other == y).all():
                        duplicates[y.shape[0]] += 1
                        p(duplicates)
                except:
                    pass
            progs.append(y)
        