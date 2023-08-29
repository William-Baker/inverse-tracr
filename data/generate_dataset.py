
import os, subprocess, sys
N = 20
samples = 1000000
offset  = 0
vocab_range = (1, 10)
numeric_range = (1, 10)
numeric_inputs_possible = True
output_path = '.data/iTracr_dataset_v2_train/'
for i in range(N):
    cmd = f"python generate_parameter_partial_dataset.py -pth \"{output_path}\" -off {offset} -s {samples} -pn {N} -idn {i} -vmin {vocab_range[0]} -vmax {vocab_range[1]} -nmin {numeric_range[0]} -nmax {numeric_range[1]} -num {numeric_inputs_possible}"
    subprocess.Popen(cmd, shell=True)

#zip -r iTracr_dataset_v2_train.zip iTracr_dataset/