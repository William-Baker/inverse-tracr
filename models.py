import jax
import jax.numpy as jnp
from flax import linen as nn
#from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2BlockCollection
from flax_gpt2 import FlaxGPT2BlockCollection
from transformers import GPT2Config
import math
import numpy as np


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o, attention
    


class EncoderBlock(nn.Module):
    input_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(embed_dim=self.input_dim,
                                            num_heads=self.num_heads)
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x
    
class TransformerEncoder(nn.Module):
    num_layers : int
    input_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)
        return attention_maps
    
class PositionalEncoding(nn.Module):
    d_model : int         # Hidden dimensionality of the input.
    max_len : int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x
    


class EncoderDecoder(nn.Module):
    num_classes: int
    dropout_prob: float = 0.15
    input_dropout_prob: float = 0.0
    enc_layers: int =5
    dec_layers: int =5
    input_dense: int =512
    attention_dim: int =128
    attention_heads: int =20
    dim_feedforward: int =512
    latent_dim: int =256
    def setup(self):
        # attention heads will not work unless this equality is integer
        assert ((3*self.attention_dim / self.attention_heads) / 3).is_integer()
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.input_dense)
        self.input_pos_encoder = PositionalEncoding(self.input_dense)
        self.encoder = TransformerEncoder(num_layers=self.enc_layers,
                        input_dim=self.attention_dim,
                        num_heads=self.attention_heads,
                        dim_feedforward=self.dim_feedforward,
                        dropout_prob=self.dropout_prob)
        self.adapter_dense = nn.Dense(self.latent_dim * self.attention_heads)
        self.pos_enc = PositionalEncoding(self.latent_dim)
        self.decoder = TransformerEncoder(num_layers=self.dec_layers,
                        input_dim=self.attention_dim,
                        num_heads=self.attention_heads,
                        dim_feedforward=self.dim_feedforward,
                        dropout_prob=self.dropout_prob)
        self.output_net = [
            nn.Dense(self.dim_feedforward),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]


    def __call__(self, x, mask=None, train=True):
        x = self.input_dropout(x, deterministic=not train)
        i = self.input_layer(x)
        i = self.input_pos_encoder(i)
        e = self.encoder(i, mask, train=train)
        # e = jnp.mean(e, axis=1)
        e = self.adapter_dense(e)
        e = e.reshape(-1, self.attention_heads, self.latent_dim)
        e = self.pos_enc(e)
        d = self.decoder(e, mask, train=train)
        o = d
        for l in self.output_net:
            o = l(o) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)
        return o



class Decoder(nn.Module):
    num_classes: int
    dropout_prob: float = 0.15
    input_dropout_prob: float = 0.0
    enc_layers: int =5
    dec_layers: int =5
    input_dense: int =512
    attention_dim: int =128
    attention_heads: int =20
    dim_feedforward: int =512
    latent_dim: int =256
    def setup(self):
        # attention heads will not work unless this equality is integer
        assert ((3*self.attention_dim / self.attention_heads) / 3).is_integer()
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.input_dense)
        self.input_pos_encoder = PositionalEncoding(self.input_dense)
        self.decoder = TransformerEncoder(num_layers=self.dec_layers,
                        input_dim=self.attention_dim,
                        num_heads=self.attention_heads,
                        dim_feedforward=self.dim_feedforward,
                        dropout_prob=self.dropout_prob)
        self.output_net = [
            nn.Dense(self.dim_feedforward),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]

    
    def __call__(self, x, mask=None, train=True):
        x = self.input_dropout(x, deterministic=not train)
        i = self.input_layer(x)
        i = self.input_pos_encoder(i)
        d = self.decoder(i, mask, train=train)
        o = d
        for l in self.output_net:
            o = l(o) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)
        return o

class GPT_Decoder(nn.Module):
    num_classes: int
    gpt_config: GPT2Config
    input_dropout_prob: float = 0.0
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.gpt_config.n_embd)
        self.input_pos_encoder = PositionalEncoding(self.gpt_config.n_embd)
        self.h = FlaxGPT2BlockCollection(self.gpt_config)
        
        self.output_net = [
            nn.Dense(1024),
            nn.LayerNorm(),
            nn.relu,
            #nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes)
        ]
    
    def __call__(self, x, attention_mask=None, train=True):
        x = self.input_dropout(x, deterministic=not train)
        i = self.input_layer(x)
        i = self.input_pos_encoder(i)
         # hidden_states, all_hidden_states, all_attentions, all_cross_attentions
        hidden_states, _, _, _ = self.h(i, attention_mask=attention_mask)
        o = hidden_states
        for l in self.output_net:
            o = l(o) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)
        return o

