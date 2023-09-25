#%%
# sintr -t 1:0:0 --nodes 1 --cpus-per-task 76 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --qos=INTR
# srun -t 4:0:0 --nodes 1 --cpus-per-task 76 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --pty bash
# sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 1 -A KRUEGER-SL2-CPU --qos=INTR
# module load cuda/11.8 cudnn/8.9_cuda-11.8
# source venv/bin/activate
# squeue -u wb326 -o "%a %c %C %D %e %F %L %M %p %q"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib

import subprocess
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import os
# from utils.export_compressed_params import transfer_to_archive
from data.parallel_read_sequential_zip import get_directory_contents_and_write
import time

mode = 'standard' # 'standard'
cmd = ''
if mode == 'compressed':
    cmd = 'python train_compressed_w_emb_multiproc.py'
    output_path = 1/0 # TODO
elif mode == 'standard':
    output_path = '.data/iTracr_dataset_v3_train/'
    cmd = f"python generate_standard_dataset.py -pth \"{output_path}\""

# print(cmd)
# 1/0
# #%%

# python generate_standard_dataset.py -pth .data/iTracr_dataset_v2_train_v2 -s 1000000 -vmin 3 -vmax 15 -nmin 3 -nmax 15 -num True

def run_experiments(id):
    logger = SummaryWriter(log_dir=f"pool compressed Tracr/{id}")
    count = 0
    while True:
        try:
            print(f'proc{id}-{count}')
            logger.add_scalar('proc', count, count)
            try:
                ret = subprocess.call(cmd, shell=True, timeout=30*60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as E:
                print(f'proc {id} {count} did not terminate in time')
                logger.add_scalar('proc', -1, count)      

            count += 1
        except Exception as E:
            import traceback
            print(str(E))
            tb = traceback.format_exc()
            print(str(tb))
        
compression = False
if __name__ == '__main__':
    #processes = int(os.cpu_count() // 1.25)
    #processes = 5#int(os.cpu_count() * 1.5)
    cores = int(len(os.sched_getaffinity(0)))#os.cpu_count()
    if compression:
        compressing_cores = max(int(cores * 0.1), 1)
        generating_cores = cores - compressing_cores
        print({'Generating cores': generating_cores, 'Compressing Cores': compressing_cores})
        threads = [Thread(target = run_experiments, args = (idx, )) for idx in range(generating_cores)]
        [thread.start() for thread in threads]
        while True:
            time.sleep(30)
            get_directory_contents_and_write(output_path, '.data/output.zip')
    else:
        generating_cores = cores
        print({'Generating cores': generating_cores})
        threads = [Thread(target = run_experiments, args = (idx, )) for idx in range(generating_cores)]
        [thread.start() for thread in threads]

    # while True:
    #     print("Archiving samples...")
    #     try:
    #         transfer_to_archive(source_dir = 'cp_dataset_train_all')
    #     except:
    #         pass
    #     try:
    #         transfer_to_archive(source_dir = 'cp_dataset_train_w')
    #     except:
    #         pass
    #     time.sleep(180)


#%%


#%%


# %%
