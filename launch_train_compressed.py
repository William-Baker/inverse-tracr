#%%

import subprocess
from threading import Thread
import wandb
from utils.time_sensitive import time_sensitive
import os

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="pool compressed Tracr",
)

def run_experiments(id):
    count = 0
    while True:
        try:
            print(f'proc{id}-{count}')
            wandb.log({f'proc{id}': count})
            try:
                ret = subprocess.call('python train_compressed_w_emb_multiproc.py', shell=True, timeout=10*60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired as E:
                print(f'proc {id} {count} did not terminate in time')
                wandb.log({f'proc{id}': -1})            

            count += 1
        except:
            pass
        

if __name__ == '__main__':
    processes = int(os.cpu_count() // 1)
    #processes = 1
    wandb.log({'Processes': processes})
    threads = [Thread(target = run_experiments, args = (idx, )) for idx in range(processes)]
    [thread.start() for thread in threads]

#%%