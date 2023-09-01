# from utils.export_compressed_params import transfer_to_archive
#srun -t 12:0:0 --nodes 1 --cpus-per-task 3 -p icelake --ntasks 1 -A KRUEGER-SL3-CPU --pty bash
# source venv/bin/activate
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib
import time
from collections import defaultdict
from zipfile import ZipFile
from cloudpickle import dumps
import numpy as np
from time import sleep
from tqdm import tqdm

def transfer_to_archive(source_dir: str):
    dest_dir = source_dir + '.zip'
    from zipfile import ZipFile, ZIP_DEFLATED
    import os
    source_files = os.listdir(source_dir)
    zip = ZipFile(dest_dir, mode='a', compression=ZIP_DEFLATED, compresslevel=9)
    dest_files = [x.filename for x in zip.filelist]
    transfer_files = set(source_files) - set(dest_files)
    for file in tqdm(transfer_files):
        try:
            zip.write(os.path.join(source_dir, file), file)
            # os.remove(os.path.join(source_dir, file)) # TODO REMOVE
        except:
            pass
    zip.close()
    for file in transfer_files:
        try:
            os.remove(os.path.join(source_dir, file))
        except Exception as E:
            pass

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
    try:
        transfer_to_archive(source_dir = '.data/iTracr_dataset_v2_train')
    except:
        pass
    print("done, waiting")
    time.sleep(60 * 30)