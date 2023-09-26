from setuptools import setup                                                                                          

setup(name='inverse-tracr',  # For pip. E.g. `pip show`, `pip uninstall`                                              
      version='0.0.1',                                                                                                
      author="William Baker",                                                                                         
      packages=["inverse_tracr"], # For python. E.g. `import python_template`                                         
      install_requires=[                                                                                              
          "flax",                                                                                                     
          "numpy",                                                                                                    
          "einops",                                                                                                   
          "chex",                                                                                                  
          "einops",
          "networkx",
          "plotly",
          "flax",
          "dill",
          "optax",
          "jax_smi",
          "kaleido",
          "transformers",
          "cloudpickle",
          "graphviz",
          "antlr4-python3-runtime==4.9.1",
          ],                                                                                                          
      )                                                                                                               

