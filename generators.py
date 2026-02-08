import jax
import jax.numpy as jnp

import strux

from typing import Self
from jaxtyping import Array, Float, Int, PRNGKeyArray


# # # 
# Sequences


@strux.struct
class Sequence:
    states: Int[Array, "n+1"]
    symbols: Int[Array, "n"]

    def __str__(self: Self) -> str:
        s = "S" + " {} S".join([str(state) for state in self.states])
        s = s.format(*[chr(ord('a')+symbol) for symbol in self.symbols])
        return s


# # # 
# Sequence generator


@strux.struct
class SequenceGenerator:
    transition_distributions: Float[Array, "symbols states states"]
    initial_distribution: Float[Array, "states"] 
    
   
    @property
    def num_symbols(self) -> int:
        num_symbols, _, _ = self.transition_distributions.shape
        return num_symbols
    
   
    @property
    def num_states(self) -> int:
        num_states, = self.initial_distribution.shape
        return num_states
    
    
    @jax.jit(static_argnames=["sequence_length"])
    def sample(
        self,
        key: PRNGKeyArray,
        sequence_length: int,
    ) -> Sequence: # sequence_length
        # choose first state from initial distribution
        key_init, key = jax.random.split(key)
        state0 = jax.random.choice(
            key=key_init,
            a=self.num_states,
            shape=(),
            p=self.initial_distribution,
        )

        # choose a subsequent state
        def step(state: int, key_step: PRNGKeyArray):
            emission_transition_distribution : Float[array, "symbols states"]
            emission_transition_distribution = self.transition_distributions[
                :,
                state,
                :,
            ]
            chosen_index = jax.random.choice(
                key=key_step,
                a=self.num_symbols * self.num_states,
                shape=(),
                p=emission_transition_distribution.flatten(),
            )
            symbol, state = jnp.divmod(chosen_index, self.num_states)
            return state, (state, symbol)
        _, (states, symbols) = jax.lax.scan(
            step,
            state0,
            jax.random.split(key, sequence_length),
        )
        
        states = jnp.concatenate([state0[None], states])
        return Sequence(
            states=states,
            symbols=symbols,
        )
    

    @jax.jit(static_argnames=["sequence_length", "batch_size"])
    def sample_batch(
        self: Self,
        key: PRNGKeyArray,
        sequence_length: int,
        batch_size: int,
    ) -> Sequence["batch_size"]:
        return jax.vmap(
            self.sample,
            in_axes=(0, None),
        )(
            jax.random.split(key, batch_size),
            sequence_length,
        )


# # # 
# Example sequence generators


MESS3 = SequenceGenerator(
    transition_distributions=jnp.array([
        [
            [0.765,  0.00375, 0.00375,],
            [0.0425, 0.0675,  0.00375,],
            [0.0425, 0.00375, 0.0675, ],
        ],
        [
            [0.0675, 0.0425,  0.00375,],
            [0.00375, 0.765,  0.00375,],
            [0.00375, 0.0425, 0.0675, ],
        ],
        [
            [0.0675,  0.00375, 0.0425,],
            [0.00375, 0.0675,  0.0425,],
            [0.00375, 0.00375, 0.765, ],
        ],
    ]),
    initial_distribution=jnp.ones(3) / 3,
)


ZOR = SequenceGenerator(
    transition_distributions=jnp.array([
        [
            [0., 1., 0.],
            [0., 0., 0.],
            [.5, 0., 0.],
        ],
        [
            [0., 0., 0.],
            [0., 0., 1.],
            [.5, 0., 0.],
        ],
    ]),
    initial_distribution=jnp.ones(3) / 3,
)


ALT = SequenceGenerator(
    transition_distributions=jnp.array([
        [
            [0., 1.],
            [0., 0.],
        ],
        [
            [0., 0.],
            [1., 0.],
        ],
    ]),
    initial_distribution=jnp.ones(2) / 2,
)



