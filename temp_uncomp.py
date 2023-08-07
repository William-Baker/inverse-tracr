import numpy as np
import jax

def attn_layer(x, Q, K, V, linear, heads):
    def project(x, w, b, heads):
        query_heads = x @ w + b
        *leading_dims, _ = x.shape
        return query_heads.reshape((*leading_dims, heads, -1))
    query_heads = project(x, Q['w'], Q['b'], heads) # [T', H, Q=K]
    key_heads = project(x, K['w'], K['b'], heads) # [T, H, Q=K]
    value_heads = project(x, V['w'], V['b'], heads) # [T, H, V]
    attn_logits = np.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(key_heads.shape[-1]).astype(x.dtype)
    attn_weights = np.array(jax.nn.softmax(attn_logits))  # [H, T', T]
    attn = np.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    *leading_dims, sequence_length, _ = x.shape
    attn = np.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = attn @ linear['w'] + linear['b']
    return final_projection

def mlp_layer(x, fst, snd):
    f = x @ fst['w'] + fst['b']
    f = np.array(jax.nn.relu(f))
    s = f @ snd['w'] + snd['b']
    return s

def execute_compressed(params, x, heads, w_emb):
    compress = lambda x: x @ w_emb.T 
    decompress = lambda x: x @ w_emb
    resid = compress(x)
    for layer, blocks in params.items():
        block_1, p_1 = list(blocks.keys())[0].split('/')
        print(layer, blocks.keys())
        decompressed = decompress(resid)
        if block_1 == 'attn':
            Q, K, V, linear = blocks['attn/query'], blocks['attn/key'], blocks['attn/value'], blocks['attn/linear']
            res = attn_layer(decompressed, Q, K, V, linear, heads)
            resid += compress(res)
        elif block_1 == 'mlp':
            fst, snd = blocks['mlp/linear_1'], blocks['mlp/linear_2']
            res = mlp_layer(decompressed, fst, snd)
            resid += compress(res)
