
###################################
### DID YOU SET THE JOB NAME??? ###
### REMEMBER TO SET JOB NAME!!! ###
###################################

#!/bin/bash
#SBATCH -J itracr_compressed
#SBATCH -A KRUEGER-SL3-CPU
#SBATCH --output=test_dmtcp_%A_%a.out
#SBATCH --error=test_dmtcp_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=3380mb
#SBATCH --array=0-5000
#SBATCH -p icelake

#SBATCH --mail-type=ALL
#SBATCH --mail-user=wb326@cam.ac.uk

#! Don't put any #SBATCH directives below this line

#! Modify the environment seen by the application. For this example we need the default modules.
. /etc/profile.d/modules.sh                # This line enables the module command
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment
module load cuda/11.8 cudnn/8.9_cuda-11.8
source venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rds/project/rds-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value
echo "This is job" $SLURM_ARRAY_TASK_ID

#! Command line that we want to run:
python launch_single.py






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

