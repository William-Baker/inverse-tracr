import os
from tqdm import tqdm




def get_directory_contents_and_write(dir_path):
    for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
        pass

if __name__ == '__main__':
    for pth in [".data/iTracr_dataset_v3_train", ".data/iTracr_dataset_v3_train_2", ".data/iTracr_dataset_v3_train_3", ".data/iTracr_dataset_v3_train_4"]:
        get_directory_contents_and_write(pth)