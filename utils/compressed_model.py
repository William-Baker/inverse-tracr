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
"""Modified transformer to learn a linear compression of the residual stream.

CompressedTransformer adds three arguments compared to Transformer:
- embedding_size: the size of the compressed residual stream.
- unembed_at_every_layer: whether to apply the unembedding before applying
    attention and MLP layers
- return_activations: whether to return all model activations rather than just
    the outputs
"""

from tracr.compiler.assemble import AssembledTransformerModelOutput, ModelForward
from tracr.transformer import encoder
from tracr.craft import bases
import jax.numpy as jnp
from typing import Any, Callable, Optional, List, Tuple
import collections
import dataclasses
from typing import Optional

import haiku as hk
import jax
import numpy as np

from tracr.transformer import attention
from tracr.transformer import model


@dataclasses.dataclass
class CompressedTransformer(hk.Module):
    """A transformer stack with linearly compressed residual stream."""

    config: model.TransformerConfig
    name: Optional[str] = None
    embedding_size: int = None

    def __call__(
        self,
        embeddings: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, T]
        *,
        use_dropout: bool = True,
        # embedding_size: Optional[int] = None,
        unembed_at_every_layer: bool = True,
    ) -> model.TransformerOutput:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences.

        Args:
          embeddings: Input embeddings to pass through the model.
          mask: Boolean mask to restrict the inputs the model uses.
          use_dropout: Turns dropout on/off.
          embedding_size: Dimension to compress the residual stream to.
          unembed_at_every_layer: Whether to unembed the residual stream when
            reading the input for every layer (keeping the layer input sizes) or to
            only unembed before the model output (compressing the layer inputs).

        Returns:
          The outputs of the forward pass through the transformer.
        """

        def layer_norm(x: jax.Array) -> jax.Array:
            """Applies a unique LayerNorm to x with default settings."""
            if self.config.layer_norm:
                return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            return x

        initializer = hk.initializers.VarianceScaling(
            2 / self.config.num_layers)
        dropout_rate = self.config.dropout_rate if use_dropout else 0.
        _, seq_len, model_size = embeddings.shape

        # To compress the model, we multiply with a matrix W when reading from
        # the residual stream, and with W^T when writing to the residual stream.
        if self.embedding_size is not None:
            # [to_size, from_size]
            w_emb = hk.get_parameter(
                "w_emb", (self.embedding_size, model_size),
                init=hk.initializers.RandomNormal())

            def write_to_residual(x): return x @ w_emb.T
            def read_from_residual(x): return x @ w_emb

            if not unembed_at_every_layer:
                model_size = self.embedding_size
        else:
            def write_to_residual(x): return x
            def read_from_residual(x): return x

        # Compute causal mask for autoregressive sequence modelling.
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
        mask = mask.repeat(seq_len, axis=2)  # [B, H=1, T, T]

        if self.config.causal:
            causal_mask = np.ones((1, 1, seq_len, seq_len))  # [B=1, H=1, T, T]
            causal_mask = np.tril(causal_mask)
            mask = mask * causal_mask  # [B, H=1, T, T]

        # Set up activation collection.
        collected = collections.defaultdict(list)

        def collect(**kwargs):
            for k, v in kwargs.items():
                collected[k].append(v)

        residual = write_to_residual(embeddings)

        for layer in range(self.config.num_layers):
            with hk.experimental.name_scope(f"layer_{layer}"):
                # First the attention block.
                attn_block = attention.MultiHeadAttention(
                    num_heads=self.config.num_heads,
                    key_size=self.config.key_size,
                    model_size=model_size,
                    w_init=initializer,
                    name="attn")

                attn_in = residual
                if unembed_at_every_layer:
                    attn_in = read_from_residual(attn_in)
                attn_in = layer_norm(attn_in)
                attn_out = attn_block(attn_in, attn_in, attn_in, mask=mask)
                attn_out, attn_logits = attn_out.out, attn_out.logits
                if dropout_rate > 0:
                    attn_out = hk.dropout(
                        hk.next_rng_key(), dropout_rate, attn_out)

                if unembed_at_every_layer:
                    collect(layer_outputs=attn_out, attn_logits=attn_logits)
                else:
                    collect(
                        layer_outputs=read_from_residual(attn_out),
                        attn_logits=attn_logits,
                    )

                if unembed_at_every_layer:
                    attn_out = write_to_residual(attn_out)
                residual = residual + attn_out

                collect(residuals=residual)

                # Then the dense block.
                with hk.experimental.name_scope("mlp"):
                    dense_block = hk.Sequential([
                        hk.Linear(
                            self.config.mlp_hidden_size,
                            w_init=initializer,
                            name="linear_1"),
                        self.config.activation_function,
                        hk.Linear(model_size, w_init=initializer,
                                  name="linear_2"),
                    ])

                dense_in = residual
                if unembed_at_every_layer:
                    dense_in = read_from_residual(dense_in)
                dense_in = layer_norm(dense_in)
                dense_out = dense_block(dense_in)
                if dropout_rate > 0:
                    dense_out = hk.dropout(
                        hk.next_rng_key(), dropout_rate, dense_out)

                if unembed_at_every_layer:
                    collect(layer_outputs=dense_out)
                else:
                    collect(layer_outputs=read_from_residual(dense_out))

                if unembed_at_every_layer:
                    dense_out = write_to_residual(dense_out)
                residual = residual + dense_out

                collect(residuals=residual)

        output = read_from_residual(residual)
        output = layer_norm(output)

        return model.TransformerOutput(
            layer_outputs=collected["layer_outputs"],
            residuals=collected["residuals"],
            attn_logits=collected["attn_logits"],
            output=output,
            input_embeddings=embeddings,
        )


@dataclasses.dataclass
class AssembledTransformerModel:
    """Model architecture and parameters from assembling a model."""
    forward: ModelForward
    get_compiled_model: Callable[[], model.CompiledTransformerModel]
    params: hk.Params
    model_config: model.TransformerConfig
    residual_labels: List[str]
    input_encoder: Optional[encoder.Encoder] = None
    output_encoder: Optional[encoder.Encoder] = None

    def encode_input(self, tokens: List[bases.Value]):
        if self.input_encoder:
            tokens = self.input_encoder.encode(tokens)
        tokens = jnp.array([tokens])
        return tokens

    def decode_output(self, output):
        decoded = output.unembedded_output[0].tolist()
        if self.output_encoder:
            decoded = self.output_encoder.decode(decoded)

        if self.input_encoder.bos_token:
            # Special case for decoding the bos token position, for which the output
            # decoder might have unspecified behavior.
            decoded = [self.input_encoder.bos_token] + decoded[1:]
        return decoded

    def apply(self, tokens: List[bases.Value]) -> AssembledTransformerModelOutput:
        """Returns output from running the model on a set of input tokens."""
        tokens = self.encode_input(tokens)
        output = self.forward(self.params, tokens)
        decoded = self.decode_output(output)

        return AssembledTransformerModelOutput(
            decoded=decoded,
            unembedded=output.unembedded_output,
            layer_outputs=output.transformer_output.layer_outputs,
            residuals=output.transformer_output.residuals,
            attn_logits=output.transformer_output.attn_logits,
            transformer_output=output.transformer_output.output,
            input_embeddings=output.transformer_output.input_embeddings)
