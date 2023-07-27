#%%

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
    #processes = int(os.cpu_count() // 1.25)
    processes = int(os.cpu_count() * 1.5)
    print({'Processes': processes})
    threads = [Thread(target = run_experiments, args = (idx, )) for idx in range(processes)]
    [thread.start() for thread in threads]
    while True:
        print("Archiving samples...")
        try:
            transfer_to_archive(source_dir = 'cp_dataset_train_all')
        except:
            pass
        try:
            transfer_to_archive(source_dir = 'cp_dataset_train_w')
        except:
            pass
        time.sleep(180)


#%%


#%%


# %%
