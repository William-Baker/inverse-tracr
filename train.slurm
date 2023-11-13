#!/bin/bash
#SBATCH -J train-large-iii
#SBATCH -A KRUEGER-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH -p ampere
###SBATCH --array=0-20

#SBATCH --output=./slurm_out/lauro/%u_%j_%x.out

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

. "/rds/user/lsl38/hpc-work/miniconda3/etc/profile.d/conda.sh"
conda activate inverse-tracr
export WANDB_DIR=/rds/user/lsl38/hpc-work/wandb
export PYTHONUNBUFFERED=1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/rds/user/lsl38/hpc-work/miniconda3/envs/inverse-tracr/lib/python3.10/site-packages/jaxlib/cuda


###################################
### DID YOU SET THE JOB NAME??? ###
### REMEMBER TO SET JOB NAME!!! ###
###################################

# OPTIONS="
#     --config MEDIUM
#     --batch_size 128
#     --model GPT2
#     --trail_name medium-run_i
# "

# OPTIONS="
#     --config pythia_125m
#     --batch_size 512
#     --model GPTNEO
#     --trail_name neo-run
# "



OPTIONS="
    --config LARGE
    --batch_size 128
    --model GPT2
    --trail_name large-run_iii
"
python train/gpt2_train_param_to_prog_ar.py $OPTIONS








CMD="$CMD $OPTIONS"
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo -e "\nExecuting command:\n==================\n$CMD\n"


start_time=$(date +%s)
eval $CMD


end_time=$(date +%s)
elapsed_seconds=$(( end_time - start_time ))
hours=$(( elapsed_seconds / 3600 ))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$(( elapsed_seconds % 60 ))

echo
echo
echo "Exited"
echo "Current Date: `date`"
printf "Time elapsed: %02d:%02d:%02d\n" $hours $minutes $seconds

