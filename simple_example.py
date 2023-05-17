#%%

from flax import linen as nn
import numpy as np
from jax import lax, random, numpy as jnp




class MLP(nn.Module):
  def setup(self):
    # Submodule names are derived by the attributes you assign to. In this
    # case, "dense1" and "dense2". This follows the logic in PyTorch.
    self.dense1 = nn.Dense(32)
    self.dense2 = nn.Dense(32)

  def __call__(self, x):
    x = self.dense1(x)
    x = nn.relu(x)
    x = self.dense2(x)
    return x


main_rng = random.PRNGKey(0)
main_rng, key2 = random.split(main_rng, 2)
#x = random.uniform(key1, (4,4))
x = jnp.ones((5,5))

# m = MLP()
# #m.init()
# i = m.apply(x)

m = MLP()
params = m.init(key2, x)
y = m.apply(params, x)


#%%
class MLP2(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32, name="dense1")(x)
    x = nn.relu(x)
    x = nn.Dense(32, name="dense2")(x)
    return x

m = MLP2()
#m.setup()
i = m.apply(x)