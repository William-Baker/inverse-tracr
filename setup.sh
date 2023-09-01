python3 -m pip install virtualenv
python3 -m virtualenv venv --python=python3.10
source venv/bin/activate

pip3 install antlr4-python3-runtime==4.9.1
pip3 install graphviz
pip3 install jax chex einops dm-haiku jax networkx numpy typing_extensions matplotlib pandas tensorboard tqdm kaleido plotly torch flax dill optax jax_smi transformers cloudpickle
#pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python -m pip install -e 

conda create --prefix ./envs
conda install -c "nvidia/label/cuda-11.7.0" cuda-cupti
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wb326/rds/rds-dsk-lab-eWkDxBhxBrQ/iTracr/inverse-tracr/envs/lib
# if your LD_LIB_PATH is not persistent, you can add the export LD_... command to the end of your venv/bin/activate script

# conda install -c conda-forge xz