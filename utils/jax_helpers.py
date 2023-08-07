from typing import Any
from jax import random

class JaxSeeder:
    def __init__(self, seed=0) -> None:
        self.main_rng = random.PRNGKey(seed)
    def next_seed(self):
        self.main_rng, new_key = random.split(self.main_rng, 2)
        return new_key
    def __call__(self):
        return self.next_seed()
    
class JaxMemUsage:
    usage = 0
    usage_unit = 'B'
    max_usage = 0
    max_usage_unit = 'B'
    usage_str = ''
    max_usage_str = ''

    import threading
    import time
    import jax
    import subprocess
    import posix
    interval = 0.01
    

    def inner():
        unit_dict = {'B': 1, 'kB': 1000, 'MB': 1000000, 'GB': 1000000000, 'TB': 1000000000000}
        while True:
            dir_prefix='/dev/shm'
            # JaxMemUsage.jax.profiler.save_device_memory_profile(f'{dir_prefix}/memory.prof.new')
            # JaxMemUsage.time.sleep(JaxMemUsage.interval//2)
            # JaxMemUsage.posix.rename(f'{dir_prefix}/memory.prof.new', f'{dir_prefix}/memory.prof')  # atomic
            # JaxMemUsage.time.sleep(JaxMemUsage.interval//2)
            output = JaxMemUsage.subprocess.run(
                    args=['go', 'tool', 'pprof', '-tags', f'{dir_prefix}/memory.prof'],
                    stdout=JaxMemUsage.subprocess.PIPE,
                    stderr=JaxMemUsage.subprocess.DEVNULL,
                ).stdout.decode('utf-8')

            if output != '':
                output = output.split('device: Total')
                if len(output) == 1:
                    continue # we're using a CPU, give up
                else:
                    output = output[1]
                JaxMemUsage.usage_str = output.split('\n')[0].split('\n')[0]
                if JaxMemUsage.usage_str.endswith('kB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'kB'
                elif JaxMemUsage.usage_str.endswith('MB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'MB'
                elif JaxMemUsage.usage_str.endswith('GB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'GB'
                elif JaxMemUsage.usage_str.endswith('TB'):
                    output = JaxMemUsage.usage_str[:-2]
                    JaxMemUsage.usage_unit = 'TB'
                elif JaxMemUsage.usage_str.endswith('B'):
                    output = JaxMemUsage.usage_str[:-1]
                    JaxMemUsage.usage_unit = 'B'
                else:
                    print(output)
                JaxMemUsage.usage = float(output)
                if JaxMemUsage.usage * unit_dict[JaxMemUsage.usage_unit] > JaxMemUsage.max_usage * unit_dict[JaxMemUsage.max_usage_unit]:
                    JaxMemUsage.max_usage = JaxMemUsage.usage
                    JaxMemUsage.max_usage_unit = JaxMemUsage.usage_unit
                #usage_str = str(usage) + usage_unit
                JaxMemUsage.max_usage_str = str(JaxMemUsage.max_usage) + JaxMemUsage.max_usage_unit
               
            JaxMemUsage.time.sleep(JaxMemUsage.interval)
            

    def launch(interval = 0.01):
        from jax_smi import initialise_tracking
        initialise_tracking(interval=interval)
        JaxMemUsage.interval = interval
        JaxMemUsage.time.sleep(JaxMemUsage.interval * 5)
        thread = JaxMemUsage.threading.Thread(target=JaxMemUsage.inner, daemon=True)
        thread.start()


from flax.core.frozen_dict import FrozenDict
import jax, optax
import jax.numpy as jnp

def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'adam'
    mask = {}
    _map(params, mask, label_fn)
    return mask

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


