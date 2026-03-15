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


    def __mul__(self: Self, other: Self) -> Self:
        """Product of two independent generators.

        Combines sub-tokens from each factor into joint tokens via the
        cartesian product: x = z1 * other.num_symbols + z2.
        The joint transition matrix for each combined symbol is the
        Kronecker product of the individual transition matrices.
        """
        s1, n1, _ = self.transition_distributions.shape
        s2, n2, _ = other.transition_distributions.shape
        # build joint transition distributions: (s1*s2, n1*n2, n1*n2)
        # for combined symbol x = z1 * s2 + z2
        joint_T = jnp.zeros((s1 * s2, n1 * n2, n1 * n2))
        for z1 in range(s1):
            for z2 in range(s2):
                x = z1 * s2 + z2
                joint_T = joint_T.at[x].set(
                    jnp.kron(
                        self.transition_distributions[z1],
                        other.transition_distributions[z2],
                    )
                )
        joint_init = jnp.kron(self.initial_distribution, other.initial_distribution)
        return SequenceGenerator(
            transition_distributions=joint_T,
            initial_distribution=joint_init,
        )


# # #
# Parameterised generators


def mess3(alpha: float = 0.85, x: float = 0.05) -> SequenceGenerator:
    """Construct a Mess3 generator with given parameters.

    The Mess3 process has 3 hidden states and 3 observable tokens.
    Parameters alpha and x control the transition structure:
      beta = (1 - alpha) / 2
      y = 1 - 2*x
    """
    beta = (1 - alpha) / 2
    y = 1 - 2 * x
    T0 = jnp.array([
        [alpha*y, beta*x, beta*x],
        [alpha*x, beta*y, beta*x],
        [alpha*x, beta*x, beta*y],
    ])
    T1 = jnp.array([
        [beta*y, alpha*x, beta*x],
        [beta*x, alpha*y, beta*x],
        [beta*x, alpha*x, beta*y],
    ])
    T2 = jnp.array([
        [beta*y, beta*x, alpha*x],
        [beta*x, beta*y, alpha*x],
        [beta*x, beta*x, alpha*y],
    ])
    return SequenceGenerator(
        transition_distributions=jnp.stack([T0, T1, T2]),
        initial_distribution=jnp.ones(3) / 3,
    )


# # #
# Example sequence generators


MESS3 = mess3(alpha=0.85, x=0.05)


# Figure 1a two-state processes.
# Sminus only emits symbol 1; Splus emits 1 (stay) or 0 (switch to Sminus).
# T[symbol, from_state, to_state] where states are [Sminus, Splus].
FIG1A_TOP = SequenceGenerator(
    transition_distributions=jnp.array([
        [[0.00, 0.00],   # symbol 0 from Sminus: never
         [0.35, 0.00]],  # symbol 0 from Splus: switch to Sminus
        [[0.70, 0.30],   # symbol 1 from Sminus: stay or switch
         [0.00, 0.65]],  # symbol 1 from Splus: stay
    ]),
    initial_distribution=jnp.array([0.5, 0.5]),
)

FIG1A_BOTTOM = SequenceGenerator(
    transition_distributions=jnp.array([
        [[0.00, 0.00],   # symbol 0 from Sminus: never
         [0.35, 0.00]],  # symbol 0 from Splus: switch to Sminus
        [[0.60, 0.40],   # symbol 1 from Sminus: stay or switch
         [0.00, 0.65]],  # symbol 1 from Splus: stay
    ]),
    initial_distribution=jnp.array([0.5, 0.5]),
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
    num_frames: int = 0,
    fps: int = 10,
):
    import numpy as np
    import matthewplotlib as mp
    import time

    GENERATORS = {
        "mess3": MESS3,
        "zor": ZOR,
        "alt": ALT,
        "fig1a": FIG1A_TOP * FIG1A_BOTTOM,
    }
    gen = GENERATORS[generator]
    key = jax.random.key(seed)

    # sample sequences and compute belief states
    batch = gen.sample_batch(key, sequence_length, num_sequences)
    all_beliefs = jax.vmap(gen.belief_states)(batch.symbols)
    # flatten, drop the prior (first timestep)
    beliefs = all_beliefs[:, 1:, :].reshape(-1, gen.num_states)
    beliefs = np.array(beliefs)

    # visualisation
    if generator == "fig1a":

        # 4 joint states from 2x2 product: colour by factor marginals
        b = beliefs.reshape(-1, 2, 2)
        b1 = b.sum(axis=2)[:, 1]  # P(factor1 = state 1)
        b2 = b.sum(axis=1)[:, 1]  # P(factor2 = state 1)
        cs = np.stack([b1, b2, 0.5*np.ones(len(b1))], axis=-1)
        cs = (cs * 255).astype(np.uint8)

        # embed 3-simplex as regular tetrahedron
        verts = np.array([[-1,1,-1], [1,-1,-1], [-1,-1,1], [1,1,1]], dtype=float)
        pts = beliefs @ verts

        # tetrahedron edges
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                t = np.linspace(0, 1, 200)[:, None]
                edges.append(verts[i] * (1 - t) + verts[j] * t)
        edge_pts = np.concatenate(edges)
        edge_cs = np.full((len(edge_pts), 3), 80, dtype=np.uint8)

        # rotating camera
        orbit_radius = 5.0
        orbit_height = 1.5
        orbit_speed = 0.4 * np.pi
        plot = None
        frame = 0
        while num_frames == 0 or frame < num_frames:
            angle = orbit_speed * frame / fps
            cam = np.array([
                orbit_radius * np.sin(angle),
                orbit_height,
                orbit_radius * np.cos(angle),
            ])
            if plot:
                print(-plot, end="")
            plot = mp.scatter3(
                (edge_pts, edge_cs),
                (pts, cs),
                camera_position=cam,
                vertical_fov_degrees=30,
                height=40,
                width=100,
            )
            print(plot)
            frame += 1
            time.sleep(1 / fps)
    else:
        # 3-state simplex projection
        xs = beliefs[:, 1] + 0.5 * beliefs[:, 2]
        ys = (np.sqrt(3) / 2) * beliefs[:, 2]
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


if __name__ == "__main__":
    import tyro
    tyro.cli(main)

