#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake-himem --ntasks 76 -A KRUEGER-SL3-CPU --qos=INTR

# icelake-himem

import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from tqdm import tqdm


def read_file(entry, dir_name):
    with open(entry.path, 'rb') as f:
        zip_path = os.path.join(dir_name, entry.name)
        return zip_path, f.read()


def get_directory_contents(dir_path):
    contents = {}
    dir_name = os.path.basename(dir_path)
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
            if entry.is_file():
                future = executor.submit(read_file, entry, dir_name)
                futures.append(future)
                    
        for future in tqdm(as_completed(futures), desc='unpacking results'):
            zip_path, file_content = future.result()
            contents[zip_path] = file_content
            
    return contents


def create_zip_from_contents(contents, zip_name):
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for filename, file_content in tqdm(contents.items(), desc='writing zip'):
            zf.writestr(filename, file_content)


if __name__ == "__main__":
    dir_path = "iTracr_dataset_v2_train"
    zip_name = "output.zip"
    
    print("reading dir contents")
    contents = get_directory_contents(dir_path)
    print("writing contents to zip")
    create_zip_from_contents(contents, zip_name)