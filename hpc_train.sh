module load cuda/11.8 cudnn/8.9_cuda-11.8
eval "$(conda shell.bash hook)"
conda activate venv
source venv/bin/activate
python gpt2_train_param_to_prog.py