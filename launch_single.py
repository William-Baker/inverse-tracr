#%%

# sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 76 -A KRUEGER-SL3-CPU --qos=INTR
#srun -t 12:0:0 --nodes 1 --cpus-per-task 3 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --pty bash
# module load cuda/11.8 cudnn/8.9_cuda-11.8
# source venv/bin/activate
# squeue -u wb326 -o "%a %c %C %D %e %F %L %M %p %q"
#!conda init bash
#!source ~/miniconda3/bin/activate
#!conda activate /rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs
#/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs
#!module load cuda/11.8 cudnn/8.9_cuda-11.8
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib

import subprocess
from threading import Thread
import os, sys
os.environ["LD_LIBRARY_PATH"] += ":/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib"
sys.path.append("/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib")
from torch.utils.tensorboard import SummaryWriter

mode = 'standard' # 'standard'
cmd = ''
if mode == 'compressed':
    cmd = 'python train_compressed_w_emb_multiproc.py'
elif mode == 'standard':
    samples = 10#1000000
    vocab_range = (1, 10)
    numeric_range = (1, 10)
    numeric_inputs_possible = True
    output_path = '.data/iTracr_dataset_v2_train/'
    cmd = f"python generate_parameter_partial_dataset.py -pth \"{output_path}\" -s {samples} -vmin {vocab_range[0]} -vmax {vocab_range[1]} -nmin {numeric_range[0]} -nmax {numeric_range[1]} -num {numeric_inputs_possible}"



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
        

if __name__ == '__main__':
    run_experiments(0)


#%%


#%%


# %%
