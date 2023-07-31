from utils.export_compressed_params import transfer_to_archive
import time

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
    time.sleep(60 * 30)