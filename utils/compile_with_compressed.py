from tracr.compiler import basis_inference
from tracr.compiler import craft_graph_to_model
from tracr.compiler import craft_model_to_transformer
from tracr.compiler import expr_to_craft_graph
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.rasp import rasp
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft import transformers
from tracr.rasp import rasp
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from tracr.craft import vectorspace_fns
from tracr.transformer import model
from tracr.transformer import encoder
# , AssembledTransformerModel
from tracr.compiler.assemble import _get_model_config_and_module_names, _make_embedding_modules
from utils import compressed_model

COMPILER_BOS = "compiler_bos"
COMPILER_PAD = "compiler_pad"


def compile_with_compressed(
        program,
        vocab,
        max_seq_len: int,
        causal=False,
        compiler_bos=COMPILER_BOS,
        compiler_pad=COMPILER_PAD,
        mlp_exactness=100,
        compression: float = None):
    
    assert compression >= 1.0

    if compiler_bos in vocab:
        raise ValueError("Compiler BOS token must not be present in the vocab. "
                         f"Found '{compiler_bos}' in {vocab}")

    if compiler_pad in vocab:
        raise ValueError("Compiler PAD token must not be present in the vocab. "
                         f"Found '{compiler_pad}' in {vocab}")

    rasp_model = rasp_to_graph.extract_rasp_graph(program)
    graph, sources, sink = rasp_model.graph, rasp_model.sources, rasp_model.sink

    basis_inference.infer_bases(
        graph,
        sink,
        vocab,
        max_seq_len,
    )

    expr_to_craft_graph.add_craft_components_to_rasp_graph(
        graph,
        bos_dir=bases.BasisDirection(rasp.tokens.label, compiler_bos),
        mlp_exactness=mlp_exactness,
    )

    craft_model = craft_graph_to_model.craft_graph_to_model(graph, sources)

    # craft_model_to_transformer.craft_model_to_transformer(
    #     craft_model=craft_model,
    #     graph=graph,
    #     sink=sink,
    #     max_seq_len=max_seq_len,
    #     causal=causal,
    #     compiler_bos=compiler_bos,
    #     compiler_pad=compiler_pad,
    # )

    """Turn a craft model into a transformer model."""

    # Add the compiler BOS token.
    tokens_value_set = (
        set(graph.nodes[rasp.tokens.label][nodes.VALUE_SET]).union(
            {compiler_bos, compiler_pad}))
    tokens_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label,
                                                          tokens_value_set)

    indices_space = bases.VectorSpaceWithBasis.from_values(
        rasp.indices.label, range(max_seq_len))

    categorical_output = rasp.is_categorical(sink[nodes.EXPR])
    output_space = bases.VectorSpaceWithBasis(sink[nodes.OUTPUT_BASIS])

    assembled_model, compressed_assembled_model = assemble_craft_model(
        craft_model=craft_model,
        tokens_space=tokens_space,
        indices_space=indices_space,
        output_space=output_space,
        categorical_output=categorical_output,
        causal=causal,
        compression=compression
    )

    for ca_model in [assembled_model, compressed_assembled_model]:
        ca_model.input_encoder = encoder.CategoricalEncoder(
            basis=tokens_space.basis,
            enforce_bos=compiler_bos is not None,
            bos_token=compiler_bos,
            pad_token=compiler_pad,
            max_seq_len=max_seq_len + 1 if compiler_bos is not None else max_seq_len,
        )

        if categorical_output:
            ca_model.output_encoder = encoder.CategoricalEncoder(
                basis=output_space.basis,
                enforce_bos=False,
                bos_token=None,
                pad_token=None)
        else:
            ca_model.output_encoder = encoder.NumericalEncoder()

    return assembled_model, compressed_assembled_model


def assemble_craft_model(
    craft_model: transformers.SeriesWithResiduals,
    tokens_space: bases.VectorSpaceWithBasis,
    indices_space: bases.VectorSpaceWithBasis,
    output_space: bases.VectorSpaceWithBasis,
    categorical_output: bool,
    causal: bool = False,
    compression: float = None
) -> compressed_model.AssembledTransformerModel:
    """Assembles the given components into a Haiku model with parameters.

    Args:
        craft_model: Model to assemble weights for.
        tokens_space: Vectorspace to embed the input tokens to.
        indices_space: Vectorspace to embed the indices to (position encodings).
        output_space: Vectorspace that the model will write outputs to that should
        be unembedded.
        categorical_output: Whether the output is categorical. If True, we take an
        argmax when unembedding.
        causal: Whether to output a causally-masked model.

    Returns:
        An AssembledTransformerModel that contains the model and parameters of the
        assembled transformer.
    """

    model_config, module_names = _get_model_config_and_module_names(
        craft_model)
    model_config.causal = causal

    residual_space = bases.join_vector_spaces(craft_model.residual_space,
                                              tokens_space, indices_space,
                                              output_space)
    residual_labels = [str(basis_dir) for basis_dir in residual_space.basis]

    # Build model with embedding and unembedding layers
    def get_compiled_model():
        transformer = model.Transformer(model_config)
        embed_modules = _make_embedding_modules(
            residual_space=residual_space,
            tokens_space=tokens_space,
            indices_space=indices_space,
            output_space=output_space)
        return model.CompiledTransformerModel(
            transformer=transformer,
            token_embed=embed_modules.token_embed,
            position_embed=embed_modules.pos_embed,
            unembed=embed_modules.unembed,
            use_unembed_argmax=categorical_output)

    compressed_embdding = len(residual_space.basis) if compression is None else int(
        len(residual_space.basis) // compression)

    def get_compressed_compiled_model():
        transformer = compressed_model.CompressedTransformer(
            model_config, embedding_size=compressed_embdding)
        embed_modules = _make_embedding_modules(
            residual_space=residual_space,
            tokens_space=tokens_space,
            indices_space=indices_space,
            output_space=output_space)
        return model.CompiledTransformerModel(
            transformer=transformer,
            token_embed=embed_modules.token_embed,
            position_embed=embed_modules.pos_embed,
            unembed=embed_modules.unembed,
            use_unembed_argmax=categorical_output)

    @hk.without_apply_rng
    @hk.transform
    def forward(emb):
        compiled_model = get_compiled_model()
        return compiled_model(emb, use_dropout=False)

    params = forward.init(jax.random.PRNGKey(0), jnp.array([[1, 2, 3]]))
    params = {k: dict(v) for k, v in params.items()}

    for key in params:
        if "transformer" in key:
            for par in params[key]:
                params[key][par] = np.zeros_like(params[key][par])

    # repeat the same for the compressed transformer
    @hk.without_apply_rng
    @hk.transform
    def compressed_forward(emb):
        compiled_model = get_compressed_compiled_model()
        return compiled_model(emb, use_dropout=False)

    w_emb = compressed_forward.init(
        jax.random.PRNGKey(0), jnp.array([[1, 2, 3]]))['compressed_transformer']['w_emb']
    
    # Assemble attention and MLP weights.
    def project(space): return vectorspace_fns.project(
        residual_space, space).matrix

    for module_name, module in zip(module_names, craft_model.blocks):
        if isinstance(module, transformers.MLP):
            hidden_size = module.fst.output_space.num_dims
            residual_to_fst_input = project(module.fst.input_space)
            snd_output_to_residual = project(module.snd.output_space).T
            params[f"{module_name}/linear_1"]["w"][:, :hidden_size] = (
                residual_to_fst_input @ module.fst.matrix)
            params[f"{module_name}/linear_2"]["w"][:hidden_size, :] = (
                module.snd.matrix @ snd_output_to_residual)
        else:  # Attention module
            query, key, value, linear = [], [], [], []
            for head in module.as_multi().heads():
                key_size = head.w_qk.matrix.shape[1]
                query_mat = np.zeros(
                    (residual_space.num_dims, model_config.key_size))
                residual_to_query = project(head.w_qk.left_space)
                query_mat[:, :key_size] = residual_to_query @ head.w_qk.matrix
                query.append(query_mat)

                key_mat = np.zeros(
                    (residual_space.num_dims, model_config.key_size))
                key_mat[:, :key_size] = project(head.w_qk.right_space)
                key.append(key_mat)

                value_size = head.w_ov.matrix.shape[1]
                value_mat = np.zeros(
                    (residual_space.num_dims, model_config.key_size))
                residual_to_ov_input = project(head.w_ov.input_space)
                value_mat[:, :value_size] = residual_to_ov_input @ head.w_ov.matrix
                value.append(value_mat)

                linear_mat = np.zeros(
                    (model_config.key_size, residual_space.num_dims))
                linear_mat[:value_size, :] = project(head.w_ov.output_space).T
                linear.append(linear_mat)

                # Fill up heads that are not used with zero weights
                for _ in range(model_config.num_heads - module.as_multi().num_heads):
                    query.append(np.zeros_like(query[0]))
                    key.append(np.zeros_like(key[0]))
                    value.append(np.zeros_like(value[0]))
                    linear.append(np.zeros_like(linear[0]))

                query = einops.rearrange(query,
                                         "heads input output -> input (heads output)")
                key = einops.rearrange(
                    key, "heads input output -> input (heads output)")
                value = einops.rearrange(value,
                                         "heads input output -> input (heads output)")
                linear = einops.rearrange(linear,
                                          "heads input output -> (heads input) output")

                params[f"{module_name}/query"]["w"][:, :] = query
                params[f"{module_name}/key"]["w"][:, :] = key
                params[f"{module_name}/value"]["w"][:, :] = value
                params[f"{module_name}/linear"]["w"][:, :] = linear


    

    params = jax.tree_util.tree_map(jnp.array, params)
    compressed_params = jax.tree_util.tree_map(jnp.array, params)

    new_compressed_params = dict()
    for key, val in compressed_params.items():
        if 'transformer/' in key:
            new_compressed_params[key.replace('transformer/','compressed_transformer/')] = val
        else:
            new_compressed_params[key] = val
    compressed_params = new_compressed_params

    compressed_params['compressed_transformer'] = dict()
    if compression is None:
        # make the compression weights identity for testing
        compressed_params['compressed_transformer']['w_emb'] = jnp.eye(
            *w_emb.shape)
    else:
        compressed_params['compressed_transformer']['w_emb'] = jnp.array(w_emb)

    

    assembled_transformer = compressed_model.AssembledTransformerModel(
        forward=forward.apply,
        get_compiled_model=get_compiled_model,
        params=params,
        model_config=model_config,
        residual_labels=residual_labels,
    )
    assembled_compressed_transformer = compressed_model.AssembledTransformerModel(
        forward=compressed_forward.apply,
        get_compiled_model=get_compressed_compiled_model,
        params=compressed_params,
        model_config=model_config,
        residual_labels=residual_labels,
    )
    assembled_compressed_transformer.residual_space=residual_space
    assembled_compressed_transformer.tokens_space=tokens_space
    assembled_compressed_transformer.indices_space=indices_space
    assembled_compressed_transformer.output_space=output_space



    @hk.without_apply_rng
    @hk.transform
    def forward_no_emb(emb):
        compiled_model = get_compiled_model()
        return compiled_model.no_emb(emb, use_dropout=False)
    
    @hk.without_apply_rng
    @hk.transform
    def forward_emb(tokens):
        compiled_model = get_compiled_model()
        return compiled_model.embed(tokens)
    
    @hk.without_apply_rng
    @hk.transform
    def comp_forward_no_emb(emb):
        compiled_model = get_compressed_compiled_model()
        return compiled_model.no_emb(emb, use_dropout=False)
    
    @hk.without_apply_rng
    @hk.transform
    def comp_forward_emb(tokens):
        compiled_model = get_compressed_compiled_model()
        return compiled_model.embed(tokens)
    
    assembled_transformer.forward_no_emb = forward_no_emb
    assembled_transformer.forward_emb = forward_emb

    assembled_compressed_transformer.forward_no_emb = comp_forward_no_emb
    assembled_compressed_transformer.forward_emb = comp_forward_emb

    return (assembled_transformer, assembled_compressed_transformer)
