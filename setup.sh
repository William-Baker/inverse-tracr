python3 -m pip install virtualenv
python3 -m virtualenv venv --python=python3.10
source venv/bin/activate

pip3 install antlr4-python3-runtime==4.9.1
pip3 install graphviz
pip3 install jax chex einops dm-haiku jax networkx numpy typing_extensions