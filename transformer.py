"""
Simple decode-only transformer architecture.
"""

import jax
import jax.numpy as jnp
import einops
import strux

from typing import Self 
from jaxtyping import Array, Float, Int, PRNGKeyArray


@strux.struct
class LinearTransform:
    weights: Float[Array, "num_inputs num_outputs"]


    @property
    def num_inputs(self: Self) -> int:
        return self.weights.shape[0]
    

    @staticmethod
    @jax.jit(static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        return LinearTransform(weights=weights)


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights


@strux.struct
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]


    @staticmethod
    @jax.jit(static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights=jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        biases=jnp.zeros(num_outputs)
        return AffineTransform(weights=weights, biases=biases)


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights + self.biases


@strux.struct(static_fieldnames=["num_heads"])
class MultiHeadedCausalSelfAttention:
    QKV: LinearTransform
    output_transform: LinearTransform
    num_heads: int


    @staticmethod
    @jax.jit(static_argnames=["embed_size", "num_heads", "head_size"])
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
        head_size: int,
    ) -> Self:
        inner_size = num_heads * head_size
        key_qkv, key = jax.random.split(key)
        QKV = jax.vmap(
            LinearTransform.init,
            in_axes=(0,None,None),
        )(
            jax.random.split(key_qkv, 3),
            embed_size,
            inner_size,
        )
        key_out, key = jax.random.split(key)
        output_transform = LinearTransform.init(
            key_out,
            inner_size,
            embed_size,
        )
        return MultiHeadedCausalSelfAttention(
            QKV=QKV,
            output_transform=output_transform,
            num_heads=num_heads,
        )


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # perform query, key, value transformations (on all heads at once)
        qkv = jax.vmap(
            type(self.QKV).forward, # two-argument version of self.QKV.forward
            in_axes=(0, None),
        )(self.QKV, x)

        # reshape the embed dimension into separate heads
        qkv_perhead = einops.rearrange(
            qkv,
            'qkv t (num_heads head_size) -> qkv t num_heads head_size',
            num_heads=self.num_heads,
        )

        # vmap the attention computation across each head
        def single_head_attention(
            qkv: Float[Array, "3 t head_size"],
        ) -> Float[Array, "t head_size"]:
            q, k, v = qkv
            t, head_size = q.shape
            # compute raw affinities                tq c @ c tk -> tq tk
            a = (q @ k.T)                                   
            # scale                                 tq tk / . . -> tq tk
            a = a * jax.lax.rsqrt(jnp.float32(head_size))
            # apply causal mask                     tq tk + t t -> tq tk
            a = jnp.where(
                jnp.tril(jnp.ones((t, t), dtype=bool)), # lower triangular mask
                a,
                -jnp.inf,
            )
            # convert affinities to mixing weights  tq tk -> tq prob(tk)
            p = jax.nn.softmax(a, axis=-1)
            # mix values for each key               tq prob(tk) @ tv c -> t c
            y = p @ v
            return y
        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=2,  # qkv t vmap(num_heads) head_size
            out_axes=1, #     t vmap(num_heads) head_size
        )(qkv_perhead)
        
        # recombine heads into new embedding dimension
        y = einops.rearrange(
            y_perhead,
            't num_heads head_size -> t (num_heads head_size)',
        )

        # for each token, project back into residual stream
        y_ = jax.vmap(self.output_transform.forward)(y)
        
        return y_
    

@strux.struct
class MLP:
    layer1: AffineTransform
    layer2: AffineTransform


    @staticmethod
    @jax.jit(static_argnames=["num_inputs", "num_hidden", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        layer1 = AffineTransform.init(k1, num_inputs, num_hidden)
        layer2 = AffineTransform.init(k2, num_hidden, num_outputs)
        return MLP(layer1=layer1, layer2=layer2)


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = self.layer1.forward(x)
        x = jax.nn.relu(x)
        x = self.layer2.forward(x)
        return x


@strux.struct
class LayerNorm:
    loc: Float[Array, "size"]
    scale: Float[Array, "size"]
    

    @staticmethod
    @jax.jit(static_argnames=["size"])
    def init(
        key: PRNGKeyArray,
        size: int,
    ) -> Self:
        return LayerNorm(
            loc=jnp.zeros(size),
            scale=jnp.ones(size),
        )


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "size"],
    ) -> Float[Array, "size"]:
        x_mean = jnp.mean(x)
        x_rstd = jax.lax.rsqrt(jnp.var(x) + 1e-5)
        x_norm = (x - x_mean) * x_rstd
        return x_norm * self.scale + self.loc


@strux.struct
class DecodeTransformerBlock:
    layernorm1: LayerNorm
    attention: MultiHeadedCausalSelfAttention
    layernorm2: LayerNorm
    compute: MLP


    @staticmethod
    @jax.jit(static_argnames=["embed_size", "num_heads", "head_size", "mlp_size"])
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
        head_size: int,
        mlp_size: int,
    ) -> Self:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        layernorm1 = LayerNorm.init(key=k1, size=embed_size)
        attention = MultiHeadedCausalSelfAttention.init(
            key=k2,
            embed_size=embed_size,
            num_heads=num_heads,
            head_size=head_size,
        )
        layernorm2 = LayerNorm.init(key=k3, size=embed_size)
        compute = MLP.init(
            key=k4,
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )
        return DecodeTransformerBlock(
            layernorm1=layernorm1,
            attention=attention,
            layernorm2=layernorm2,
            compute=compute,
        )


    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm1.forward)(x)
        # attention (between tokens, residual)
        x = x + self.attention.forward(x_norm)
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm2.forward)(x)
        # compute (per-token, residual)
        x = x + jax.vmap(self.compute.forward)(x_norm)
        return x


@strux.struct
class DecodeTransformer:
    token_embedding: LinearTransform
    postn_embedding: LinearTransform
    blocks: tuple
    unembedding_layernorm: LayerNorm
    unembedding: AffineTransform

    @property
    def num_inputs(self: Self) -> int:
        return self.token_embedding.num_inputs

    @staticmethod
    @jax.jit(static_argnames=[
        "num_inputs",
        "max_context_length",
        "num_blocks",
        "num_heads",
        "head_size",
        "embed_size",
        "mlp_size",
        "num_outputs",
    ])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        max_context_length: int,
        num_blocks: int,
        num_heads: int,
        head_size: int,
        embed_size: int,
        mlp_size: int,
        num_outputs: int,
    ) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # embeddings
        token_embedding = LinearTransform.init(
            key=k1,
            num_inputs=num_inputs,
            num_outputs=embed_size,
        )
        postn_embedding = LinearTransform.init(
            key=k2,
            num_inputs=max_context_length,
            num_outputs=embed_size,
        )
        
        # transformer blocks
        blocks = jax.vmap(
            DecodeTransformerBlock.init,
            in_axes=(0,None,None,None,None),
        )(
            jax.random.split(k3, num_blocks),
            embed_size,
            num_heads,
            head_size,
            mlp_size,
        )

        # unembedding
        unembedding_layernorm = LayerNorm.init(
            key=k4,
            size=embed_size,
        )
        unembedding = AffineTransform.init(
            key=k5,
            num_inputs=embed_size,
            num_outputs=num_outputs,
        )
        return DecodeTransformer(
            token_embedding=token_embedding,
            postn_embedding=postn_embedding,
            blocks=blocks,
            unembedding_layernorm=unembedding_layernorm,
            unembedding=unembedding,
        )

    def _embed(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> Float[Array, "t embed_size"]:
        context_length, _num_inputs = ts.shape
        x_semantic = jax.vmap(self.token_embedding.forward)(ts)
        x_position = self.postn_embedding.weights[:context_length, :]
        return x_semantic + x_position


    @jax.jit
    def forward(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> Float[Array, "t num_outputs"]:
        x = self._embed(ts)
        # apply the num_blocks attention blocks in sequence
        x, _ = jax.lax.scan(
            lambda x, block: (block.forward(x), None),
            x,
            self.blocks,
        )                                                   # -> t embed_size
        # unembedding: transform back to predicted next token probs
        x_norm = jax.vmap(self.unembedding_layernorm.forward)(x)
        x = jax.vmap(self.unembedding.forward)(x_norm)      # -> t num_outputs
        return x


    @jax.jit
    def forward_with_activations(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> tuple[
        Float[Array, "t num_outputs"],
        Float[Array, "num_blocks_plus_1 t embed_size"],
    ]:
        x0 = self._embed(ts)
        # collect residual stream after each block
        def scan_fn(x, block):
            y = block.forward(x)
            return y, y
        x, layer_xs = jax.lax.scan(scan_fn, x0, self.blocks)
        # prepend the embedding layer activations
        activations = jnp.concatenate([x0[None], layer_xs])
        # unembedding
        x_norm = jax.vmap(self.unembedding_layernorm.forward)(x)
        x = jax.vmap(self.unembedding.forward)(x_norm)
        return x, activations


@strux.struct
class SequenceTransformer:
    transformer: DecodeTransformer


    @property
    def num_symbols(self: Self) -> int:
        return self.transformer.num_inputs


    @staticmethod
    @jax.jit(static_argnames=[
        "num_symbols",
        "sequence_length",
        "num_blocks",
        "embed_size",
        "num_heads",
        "head_size",
        "mlp_size",
    ])
    def init(
        key: PRNGKeyArray,
        num_symbols: int,
        sequence_length: int,
        num_blocks: int,
        embed_size: int,
        num_heads: int,
        head_size: int,
        mlp_size: int,
    ) -> Self:
        transformer = DecodeTransformer.init(
            key=key,
            num_inputs=num_symbols,
            max_context_length=sequence_length,
            num_blocks=num_blocks,
            embed_size=embed_size,
            num_heads=num_heads,
            head_size=head_size,
            mlp_size=mlp_size,
            num_outputs=num_symbols,
        )
        return SequenceTransformer(transformer=transformer)


    @jax.jit
    def forward(
        self,
        xs: Int[Array, "sequence_length"],
    ) -> Float[Array, "sequence_length num_symbols"]:
        toks = jax.nn.one_hot(
            x=xs,
            num_classes=self.num_symbols,
        )
        logits = self.transformer.forward(toks)
        return logits
    

    @jax.jit
    def forward_with_activations(
        self: Self,
        xs: Int[Array, "sequence_length"],
    ) -> tuple[
        Float[Array, "sequence_length num_symbols"],
        Float[Array, "num_layers t embed_size"],
    ]:
        toks = jax.nn.one_hot(x=xs, num_classes=self.num_symbols)
        return self.transformer.forward_with_activations(toks)


    @jax.jit
    def forward_batch(
        self: Self,
        xss: Int[Array, "batch_size sequence_length"],
    ) -> Float[Array, "batch_size sequence_length num_symbols"]:
        return jax.vmap(self.forward)(xss)


    @jax.jit
    def forward_batch_with_activations(
        self: Self,
        xss: Int[Array, "batch_size sequence_length"],
    ) -> tuple[
        Float[Array, "batch_size sequence_length num_symbols"],
        Float[Array, "batch_size num_layers sequence_length embed_size"],
    ]:
        return jax.vmap(self.forward_with_activations)(xss)



