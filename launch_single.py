#%%
# srun -t 1:0:0 --nodes 1 --cpus-per-task 38 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --pty bash --qos=INTR
# sintr -t 24:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 76 -A KRUEGER-SL3-CPU --qos=INTR
# module load cuda/11.8 cudnn/8.9_cuda-11.8
# source venv/bin/activate
# squeue -u wb326 -o "%a %c %C %D %e %F %L %M %p %q"

import subprocess
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import os
from utils.export_compressed_params import transfer_to_archive
import time


def run_experiments(id):
    logger = SummaryWriter(log_dir=f"pool compressed Tracr/{id}")
    count = 0
    while True:
        try:
            print(f'proc{id}-{count}')
            logger.add_scalar('proc', count, count)
            try:
                ret = subprocess.call('python train_compressed_w_emb_multiproc.py', shell=True, timeout=30*60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as E:
                print(f'proc {id} {count} did not terminate in time')
                logger.add_scalar('proc', -1, count)      

            count += 1
        except Exception as E:
            import traceback
            print(str(E))
            tb = traceback.format_exc()
            print(str(tb))
        

if __name__ == '__main__':
    run_experiments(0)


#%%


#%%


# %%
