#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake-himem --ntasks 76 -A KRUEGER-SL3-CPU --qos=INTR
#sintr -t 1:0:0 --nodes 1 --cpus-per-task 1 -p icelake --ntasks 16 -A KRUEGER-SL3-CPU --qos=INTR

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
    
def dir_not_empty(dir):
    for i in os.scandir(dir):
        return True
    return False


def get_directory_contents_and_write(dir_path, zip_name, batch_size=100000):
    dir_name = os.path.basename(dir_path)
    while dir_not_empty(dir_path):
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
                
                if i > batch_size:
                    break
                if entry.is_file():
                    future = executor.submit(read_file, entry, dir_name)
                    futures.append(future)

            rm_futures = []
            with ThreadPoolExecutor() as rm_executor:
                with zipfile.ZipFile(zip_name, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:          
                    for future in tqdm(as_completed(futures), desc='await future + write zip'):
                        zip_path, file_content = future.result()
                        
                        # TODO check zip path ends with .pkl
                        # contents[zip_path] = file_content
                        zf.writestr(zip_path, file_content)
                        # del file_content
                        rm_future = rm_executor.submit(os.remove, os.path.join(dir_path, os.path.basename(zip_path)))
                        rm_futures.append(rm_future)
            for future in tqdm(as_completed(rm_futures), desc='await finishing deleting'):
                pass
    

# def get_directory_contents_and_write(dir_path, zip_name, batch_size=100000):
#     dir_name = os.path.basename(dir_path)
#     while dir_not_empty(dir_path):
#         with ThreadPoolExecutor() as executor:
#             futures = []
#             for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
                
#                 if i > batch_size:
#                     break
#                 if entry.is_file():
#                     future = executor.submit(read_file, entry, dir_name)
#                     futures.append(future)

#             to_delete = []
#             with zipfile.ZipFile(zip_name, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:          
#                 for future in tqdm(as_completed(futures), desc='await future + write zip'):
#                     zip_path, file_content = future.result()
                    
#                     # TODO check zip path ends with .pkl
#                     # contents[zip_path] = file_content
#                     zf.writestr(zip_path, file_content)
#                     # del file_content
#                     to_delete.append(os.path.join(dir_path, os.path.basename(zip_path)))

                        

#         rm_futures = []
#         with ThreadPoolExecutor() as rm_executor:
#             for d in tqdm(to_delete, desc='queueing files to delete'):
#                 rm_future = rm_executor.submit(os.remove, d)
#                 rm_futures.append(rm_future)
#             for future in tqdm(as_completed(rm_futures), desc='await finishing deleting'):
#                 pass

# def get_directory_contents_and_write(dir_path, zip_name, batch_size=1000000):
#     dir_name = os.path.basename(dir_path)
#     while dir_not_empty(dir_path):
#         with ThreadPoolExecutor() as executor:
#             futures = []
#             for i, entry in tqdm(enumerate(os.scandir(dir_path)), desc='reading from dir'):
                
#                 if i > batch_size:
#                     break
#                 if entry.is_file():
#                     future = executor.submit(read_file, entry, dir_name)
#                     futures.append(future)

#             to_delete = []
#             with zipfile.ZipFile(zip_name, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zf:          
#                 for future in tqdm(as_completed(futures), desc='await future + write zip'):
#                     zip_path, file_content = future.result()
                    
#                     # TODO check zip path ends with .pkl
#                     # contents[zip_path] = file_content
#                     zf.writestr(zip_path, file_content)
#                     # del file_content
#                     to_delete.append(os.path.join(dir_path, os.path.basename(zip_path)))

                        
#         for d in tqdm(to_delete, desc='queueing files to delete'):
#             os.remove(d)
#         # rm_futures = []
#         # with ThreadPoolExecutor() as rm_executor:
#         #     for d in tqdm(to_delete, desc='queueing files to delete'):
#         #         rm_future = rm_executor.submit(os.remove, d)
#         #         rm_futures.append(rm_future)
#         #     for future in tqdm(as_completed(rm_futures), desc='await finishing deleting'):
#         #         pass
    


if __name__ == "__main__":
    dir_path = ".data/iTracr_dataset_v3_train_4"
    zip_name = ".data/output_4.zip"
    
    
    print("reading dir contents")
    while True:
        try:
            contents = get_directory_contents_and_write(dir_path, zip_name)
        except FileNotFoundError:
            pass
        sleep(10)