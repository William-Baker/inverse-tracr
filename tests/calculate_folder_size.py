import os
from tqdm import tqdm




def get_directory_contents_and_write(dir_path):
    for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
        pass

if __name__ == '__main__':
    get_directory_contents_and_write(".data/iTracr_dataset_v3_train")