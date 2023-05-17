
#%%

from utils.jax_helpers import JaxSeeder
seeder = JaxSeeder(0)

from torch.utils.data import DataLoader
from utils.dataloaders import ProgramDataset

dataset = ProgramDataset(30)
train_dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.get_collate_fn()) # , num_workers=8, prefetch_factor=2, pin_memory=True)


it = iter(train_dataloader)
x,y = next(it)
print(dataset.decode_pred(y, 0))
print(dataset.decode_pred(x, 0))

#%%




from models import EncoderDecoder, TransformerEncoder, EncoderBlock
from jax import random
import jax.numpy as jnp

x = jnp.ones((1,1,500))

enc = EncoderDecoder(5, dropout_prob=.50)

# enc = TransformerEncoder(num_layers=5,
#                               input_dim=128,
#                               num_heads=4,
#                               dim_feedforward=256,
#                               dropout_prob=0.15)

# enc = EncoderBlock(input_dim=128,
#                             num_heads=4,
#                             dim_feedforward=256,
#                             dropout_prob=0.15)


init_rng, dropout_init_rng, dropout_apply_rng = seeder(), seeder(), seeder()
params = enc.init({'params': init_rng, 'dropout': dropout_init_rng}, x, True)['params']
# Apply encoder block with parameters on the inputs
# Since dropout is stochastic, we need to pass a rng to the forward

# will return the same thing every time
out = enc.apply({'params': params}, x, train=True, rngs={'dropout': dropout_apply_rng}) 
out.sum()

#%%

# will return something different each time
out = enc.apply({'params': params}, x, train=True, rngs={'dropout': seeder()})

out.sum()






#%%

import flax.linen as nn
import jax.numpy as jnp

class MyModel(nn.Module):
  num_neurons: int

  @nn.compact
  def __call__(self, x, training: bool):
    x = nn.Dense(self.num_neurons)(x)
    # Set the dropout layer with a `rate` of 50%.
    # When the `deterministic` flag is `True`, dropout is turned off.
    x = nn.Dropout(rate=0.5, deterministic=not training)(x)
    return x



my_model = MyModel(num_neurons=3)
x = jnp.empty((3, 4, 4))
# Dropout is disabled with `training=False` (that is, `deterministic=True`).
variables = my_model.init(seeder(), x, training=False)
params = variables['params']

#%%

# Dropout is enabled with `training=True` (that is, `deterministic=False`).
y = my_model.apply({'params': params}, x, training=True, rngs={'dropout': seeder()})

#%%

from flax import linen as nn
from jax.random import PRNGKey
from jax.numpy import zeros

class Foo(nn.Module):
  @nn.compact
  def __call__(self, x):
    rng1 = self.make_rng('dropout')
    rng2 = self.make_rng('dropout')
    assert (rng1 != rng2).any()

rngs = {'params': PRNGKey(0), 'dropout': PRNGKey(1)}
_ = Foo().init(rngs, zeros((2,)))

#%%

class Bar(nn.Module):
  @nn.compact
  def __call__(self, x):
    rng1 = self.make_rng('dropout')
    return rng1

rngs = {'params': PRNGKey(0), 'dropout': PRNGKey(1)}

y1, _ = Bar().init_with_output(rngs, zeros((2,)))
y2, _ = Bar().init_with_output(rngs, zeros((2,)))

assert (y1 == y2).all()


#%%

# @jax.jit
# def train_step(state: TrainState, batch, dropout_key):
#   dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
#   def loss_fn(params):
#     logits = state.apply_fn(
#       {'params': params},
#       x=batch['image'],
#       training=True,
#       rngs={'dropout': dropout_train_key}
#       )
#     loss = optax.softmax_cross_entropy_with_integer_labels(
#       logits=logits, labels=batch['label'])
#     return loss, logits
#   grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#   (loss, logits), grads = grad_fn(state.params)
#   state = state.apply_gradients(grads=grads)
#   return state