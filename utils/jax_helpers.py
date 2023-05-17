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