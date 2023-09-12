
import argparse
from os import makedirs
from tqdm import tqdm
import cloudpickle
from datetime import datetime
   
from data.dataloader_streams import StreamWriter
from data.parameter_program_dataloader import TorchParameterProgramDataset

try:
   # construct the argument parse and parse the arguments
   ap = argparse.ArgumentParser()
   ap.add_argument("-s", "--samples", required=True, help="number of samples", type=int)
   ap.add_argument("-vmin", "--vocab_min", required=False, help="lower bound of vocab size", type=int, default=6)
   ap.add_argument("-vmax", "--vocab_max", required=False, help="upper bound of vocab size", type=int, default=6)
   ap.add_argument("-nmin", "--number_min", required=False, help="lower bound of numeric range size", type=int, default=6)
   ap.add_argument("-nmax", "--number_max", required=False, help="upper bound of numeric range size", type=int, default=6)
   ap.add_argument("-num", "--numeric_inputs", required=False, help="whether to generate samples with numeric inputs", type=bool, default=False)
   ap.add_argument("-pth", "--output_path", required=False, help="path to export files to", type=str, default=False)


   args = ap.parse_args()
   # args = argparse.Namespace(proc_num=1, id_number=0, samples=10000, offset=900000)

   print(args.output_path)

   dataset = TorchParameterProgramDataset(3, 15, no_samples = args.samples, generator_backend='bounded', bounded_timeout_multiplier=1,
                                          vocab_size_range=(args.vocab_min, args.vocab_max),
                                          numeric_range=(args.number_min, args.number_max),
                                          numeric_inputs_possible=args.numeric_inputs
                                       )
   #pth = '.data/iTracr_dataset_v2_test/'
   # sw = StreamWriter(args.pth, dataset, id_N=(args.id_number, args.proc_num), start_offset=args.offset)
   # sw.write_samples(num_threads=0)




   makedirs(args.output_path, exist_ok=True)
   it = iter(dataset)
   for idx in tqdm(range(args.samples), desc=f'Writing samples:'):
      x, y = next(it)
      # np.savez(self.dir + str(idx).zfill(8), x=x, y=y)
      for i in range(2000):
         try:
            with open(args.output_path + "/" +  str(datetime.now().strftime("%m-%d %H.%M.%S.%f")) + '.pkl', 'wb') as f:
               cloudpickle.dump((x,y), f)
            
         except:
            print("failed to save to zip archive")
except Exception as E:
    from os import makedirs
    makedirs('logs', exist_ok=True)
    with open(f'logs/{str(datetime.now().strftime("%m-%d %H.%M.%S.%f"))}.txt','w') as f:
        import traceback
        f.write(str(E))
        tb = traceback.format_exc()
        f.write(str(tb))


