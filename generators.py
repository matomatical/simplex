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
    

    @jax.jit
    def belief_states(
        self: Self,
        symbols: Int[Array, "n"],
    ) -> Float[Array, "n+1 states"]:
        """Compute ground truth belief states for an observation sequence.

        Returns the posterior P(S_t | o_1, ..., o_t) at each timestep,
        starting with the prior (before any observations).
        """
        def step(belief, symbol):
            # b_new(s') = Σ_s b(s) * T[symbol, s, s']
            belief = belief @ self.transition_distributions[symbol]
            belief = belief / belief.sum()
            return belief, belief
        belief0 = self.initial_distribution
        _, beliefs = jax.lax.scan(step, belief0, symbols)
        # prepend the prior
        beliefs = jnp.concatenate([belief0[None], beliefs])
        return beliefs


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


# # # 
# Testing


def main(
    generator: str = "mess3",
    num_sequences: int = 1000,
    sequence_length: int = 100,
    seed: int = 0,
    save: str | None = None,
):
    import numpy as np
    import matthewplotlib as mp

    GENERATORS = {
        "mess3": MESS3,
        "zor": ZOR,
        "alt": ALT,
    }
    gen = GENERATORS[generator]
    key = jax.random.key(seed)

    # sample sequences and compute belief states
    batch = gen.sample_batch(key, sequence_length, num_sequences)
    # compute belief states for each sequence: batch_size x (n+1) x states
    all_beliefs = jax.vmap(gen.belief_states)(batch.symbols)
    # flatten to (batch_size * (n+1)) x states, drop the prior (first timestep)
    beliefs = all_beliefs[:, 1:, :].reshape(-1, gen.num_states)
    beliefs = np.array(beliefs)

    # project 3-state simplex to 2D equilateral triangle
    # vertices: v0=(0,0), v1=(1,0), v2=(0.5, sqrt(3)/2)
    xs = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    ys = (np.sqrt(3) / 2) * beliefs[:, 2]

    # color by belief state (RGB = probability of each state)
    cs = np.stack([beliefs[:, 0], beliefs[:, 1], beliefs[:, 2]], axis=-1)
    cs = (cs * 255).astype(np.uint8)

    plot = mp.axes(
        mp.scatter(
            (xs, ys, cs),
            height=30,
            width=60,
            xrange=(-0.05, 1.05),
            yrange=(-0.05, 0.95),
        ),
        title=f" {generator} belief states ",
        xlabel="",
        ylabel="",
    )
    print(plot)
    if save:
        plot.saveimg(save)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

