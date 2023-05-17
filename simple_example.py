#%%

from flax import linen as nn
import numpy as np

x = np.ones((5,5))

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


m = MLP()
i = m(x)