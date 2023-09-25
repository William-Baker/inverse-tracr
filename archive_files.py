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

def transfer_to_archive(source_dir: str, dest_dir=None):
    if dest_dir is None:
        dest_dir = source_dir + '.zip'
    from zipfile import ZipFile, ZIP_DEFLATED
    import os
    source_files = os.listdir(source_dir)
    zip = ZipFile(dest_dir, mode='a', compression=ZIP_DEFLATED, compresslevel=9)
    dest_files = [x.filename for x in zip.filelist]
    transfer_files = set(source_files) - set(dest_files)
    print(f"doing {source_dir}")
    while len(transfer_files) > 0:
        num = min(len(transfer_files), 1000)
        working_files = transfer_files[:num]
        transfer_files = transfer_files[num:]
        print(f"doing {len(working_files)}")
        for file in tqdm(working_files):
            try:
                zip.write(os.path.join(source_dir, file), file)
                # os.remove(os.path.join(source_dir, file)) # TODO REMOVE
            except:
                pass
        zip.close()
        for file in working_files:
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
        #transfer_to_archive(source_dir = '.data/iTracr_dataset_v2_train')
        transfer_to_archive(source_dir = '.data/iTracr_dataset_v3_train', dest_dir='.data/output.zip')
    except:
        pass
    print("done, waiting")
    time.sleep(60 * 30)