
import traceback
import argparse
from os import makedirs
from tqdm import tqdm
import cloudpickle
from datetime import datetime

from inverse_tracr.data.encoded_dataloaders import craft_dataset, program_craft_generator_bounded, program_craft_generator_unbounded
from inverse_tracr.data.dataloaders import ProgramEncoder


try:
   # construct the argument parse and parse the arguments
   ap = argparse.ArgumentParser()
   ap.add_argument("-pth", "--output_path", required=False, help="path to export files to", type=str, default='.data/iTracr_dataset_v3_train/')


   args = ap.parse_args()
   # args = argparse.Namespace(proc_num=1, id_number=0, samples=10000, offset=900000)

   print(args.output_path)

   samples = 10000000

   vocab_range = (3, 15)
   numeric_range = (3, 15)
   numeric_inputs_possible = True

   (min_prog_len,max_prog_len) = 3,15
   vocab_size_range = vocab_range
   func = program_craft_generator_bounded
   gen, OP_VOCAB, VAR_VOCAB = craft_dataset((min_prog_len,max_prog_len), func=func, timeout_multiplier=int(1).as_integer_ratio,
                                 vocab_size_range=vocab_size_range, numeric_range=numeric_range, numeric_inputs_possible=numeric_inputs_possible)


   prog_enc = ProgramEncoder(max_prog_len)

   def get():
    x,y = gen()
    y = prog_enc.tokenise_program(y)
    return x,y


   makedirs(args.output_path, exist_ok=True)
   for idx in tqdm(range(samples), desc=f'Writing samples:'):
      x, y = get()
      # np.savez(self.dir + str(idx).zfill(8), x=x, y=y)
      pth = str(datetime.now().strftime("%m-%d %H.%M.%S.%f")) + '.pkl'
      writes = 0
      for i in range(100):
         try:
            with open(args.output_path + "/" +  pth, 'wb') as f:
               cloudpickle.dump((x,y), f)
               writes += 1
            break
         except:
            print("failed to save to zip archive")
      assert writes <= 1
except Exception as E:
    makedirs('logs', exist_ok=True)
    with open(f'logs/{str(datetime.now().strftime("%m-%d %H.%M.%S.%f"))}.txt','w') as f:
        f.write(str(E))
        tb = traceback.format_exc()
        f.write(str(tb))


