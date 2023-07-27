# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Didactic example of an autoregressive Transformer-based language model.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
- V: Vocabulary size.

Forked from: haiku.examples.transformer.model
"""

import collections
import dataclasses
from typing import Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.transformer import attention

# hk.Modules are not always callable: github.com/deepmind/dm-haiku/issues/52
# Ideally, we'd want a type:
# CallableHaikuModule = Intersection[Callable[..., jax.Array], hk.Module]
# But Intersection does not exist (yet): github.com/python/typing/issues/213
CallableHaikuModule = Callable[..., jax.Array]

from tracr.transformer.model import TransformerOutput, TransformerConfig, Transformer, CompiledTransformerModelOutput


@dataclasses.dataclass
class CompiledTransformerModel(hk.Module):
  """A transformer model with one-hot embeddings."""
  transformer: Transformer
  token_embed: CallableHaikuModule
  position_embed: CallableHaikuModule
  unembed: CallableHaikuModule
  use_unembed_argmax: bool
  pad_token: Optional[int] = None

  def embed(self, tokens: jax.Array) -> jax.Array:
    token_embeddings = self.token_embed(tokens)
    positional_embeddings = self.position_embed(jnp.indices(tokens.shape)[-1])
    return token_embeddings + positional_embeddings  # [B, T, D]

  def __call__(
      self,
      tokens: jax.Array,
      use_dropout: bool = True,
  ) -> CompiledTransformerModelOutput:
    """Embed tokens, pass through model, and unembed output."""
    if self.pad_token is None:
      input_mask = jnp.ones_like(tokens)
    else:
      input_mask = (tokens != self.pad_token)
    input_embeddings = self.embed(tokens)

    transformer_output = self.transformer(
        input_embeddings,
        input_mask,
        use_dropout=use_dropout,
    )
    return CompiledTransformerModelOutput(
        transformer_output=transformer_output,
        unembedded_output=self.unembed(
            transformer_output.output,
            use_unembed_argmax=self.use_unembed_argmax,
        ),
    )
  
  def no_emb(
    self,
    input_embeddings: jax.Array,
    use_dropout: bool = True,
  ) -> CompiledTransformerModelOutput:
    """Embed tokens, pass through model, and unembed output."""

    input_mask = jnp.ones(input_embeddings.shape[:-1])
  

    transformer_output = self.transformer(
        input_embeddings,
        input_mask,
        use_dropout=use_dropout,
    )
    return CompiledTransformerModelOutput(
        transformer_output=transformer_output,
        unembedded_output=self.unembed(
            transformer_output.output,
            use_unembed_argmax=self.use_unembed_argmax,
        ),
    )
