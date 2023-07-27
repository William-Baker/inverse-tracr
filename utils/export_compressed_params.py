from data.dataloaders import ProgramEncoder
from collections import defaultdict
from zipfile import ZipFile
from cloudpickle import dumps
import numpy as np
from time import sleep

def compress_params(params):
    # we first need to find the compression matrix
    w = params['compressed_transformer']['w_emb'].T
    compressed_params = dict()
    for key in params.keys():
        if 'compressed_transformer/' in key:
            p = params[key]['w']
            key = key.replace( 'compressed_transformer/', '')
            if not (key.endswith('linear') or key.endswith('linear_2')):
                compressed_params[key] = np.array((p.T @ w).T)
            else:
                compressed_params[key] = np.array((p @ w))
    return compressed_params


def encode_jax_params(params):
    collected_by_block = defaultdict(lambda: dict())
    for key, val in params.items():
        layer_no, layer_type, param_name = key.split('/')
        collected_by_block[layer_no + layer_type][param_name] = val
    
    model_params = []
    for key, val in collected_by_block.items():
        if 'attn' in layer_type:
            model_params.append({'MHA': val})
        elif 'mlp' in layer_type:
            model_params.append({'MLP': val})
        else:
            raise NotImplementedError()
    return model_params

import cloudpickle
from os import makedirs

def export_params(params, max_ops, actual_ops, trn_all, run_id):
    compressed = compress_params(params)
    prog_enc = ProgramEncoder(max_ops)
    encoded_ops = ProgramEncoder.encode_ops(actual_ops)
    tokenised_program = prog_enc.tokenise_program(encoded_ops)
    encoded_params = encode_jax_params(compressed)
    sample = (encoded_params, tokenised_program)

    target_db_path = 'cp_dataset'
    if trn_all == True:
        target_db_path += '_train_all'
    else:
        target_db_path += '_train_w'

    for i in range(2000):
        try:
            # zip = ZipFile(file=target_db_path+'.zip', mode='a')
            # zip.writestr(run_id + '.pkl', dumps(sample))
            # zip.close()
            makedirs(target_db_path, exist_ok=True)
            with open(target_db_path + "/" + run_id + '.pkl', 'wb') as f:
                cloudpickle.dump(sample, f)
            
        except:
            sleep(1)
            print("failed to save to zip archive")

def transfer_to_archive(source_dir: str):
    dest_dir = source_dir + '.zip'
    from zipfile import ZipFile
    import os
    source_files = os.listdir(source_dir)
    zip = ZipFile(dest_dir, mode='a')
    dest_files = [x.filename for x in zip.filelist]
    transfer_files = set(source_files) - set(dest_files)
    for file in transfer_files:
        zip.write(os.path.join(source_dir, file), file)
    zip.close()
    for file in transfer_files:
        try:
            os.remove(os.path.join(source_dir, file))
        except Exception as E:
            pass