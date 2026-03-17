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
        parts = []
        for i, state in enumerate(self.states):
            parts.append(f"S{int(state)}")
            if i < len(self.symbols):
                parts.append(f"[{int(self.symbols[i])}]")
        return ' '.join(parts)


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
            belief_new = belief @ self.transition_distributions[symbol]
            total = belief_new.sum()
            # if observation is impossible under model, keep previous belief
            belief_new = jnp.where(total > 0, belief_new / total, belief)
            return belief_new, belief_new
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


    def __add__(self: Self, other: Self) -> Self:
        """Disjoint union of two generators. See disjoint_union()."""
        return disjoint_union(self, other)


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


def disjoint_union(*generators: SequenceGenerator) -> SequenceGenerator:
    """Disjoint union of N generators (shared symbol alphabet).

    The union HMM first chooses which component to enter (uniformly 1/N),
    then samples from that component forever. The transition matrix is
    block-diagonal: no transitions between components.
    """
    assert len(generators) >= 1
    num_symbols = generators[0].num_symbols
    for gen in generators[1:]:
        assert gen.num_symbols == num_symbols, (
            f"Disjoint union requires same symbol alphabet, "
            f"got {num_symbols} vs {gen.num_symbols} symbols"
        )
    total_states = sum(gen.num_states for gen in generators)
    N = len(generators)
    union_T = jnp.zeros((num_symbols, total_states, total_states))
    init_parts = []
    offset = 0
    for gen in generators:
        n = gen.num_states
        union_T = union_T.at[:, offset:offset+n, offset:offset+n].set(
            gen.transition_distributions,
        )
        init_parts.append(gen.initial_distribution / N)
        offset += n
    return SequenceGenerator(
        transition_distributions=union_T,
        initial_distribution=jnp.concatenate(init_parts),
    )


def noisy_channel(gen: SequenceGenerator, epsilon: float) -> SequenceGenerator:
    """Apply memoryless epsilon-noise to a generator.

    With probability (1-ε), emit the original token.
    With probability ε, replace with a uniformly random other token.
    """
    S = gen.num_symbols
    T = gen.transition_distributions  # (symbols, states, states)
    # sum over all symbols to get state-transition-given-state: (states, states)
    T_state = T.sum(axis=0)
    # noisy version: (1-ε)*T[x] + (ε/(S-1))*Σ_{x'≠x} T[x']
    #              = (1-ε)*T[x] + (ε/(S-1))*(T_state - T[x])
    T_noisy = (1 - epsilon) * T + (epsilon / (S - 1)) * (T_state[None] - T)
    return SequenceGenerator(
        transition_distributions=T_noisy,
        initial_distribution=gen.initial_distribution,
    )


def decompose_union_beliefs(
    beliefs: Float[Array, "... total_states"],
    component_sizes: list[int],
) -> tuple[Float[Array, "... num_components"], list[Float[Array, "... states_i"]]]:
    """Decompose joint beliefs from a disjoint union into per-component parts.

    Returns:
        weights: mixture weights per component (sum to 1)
        component_beliefs: normalized belief within each component
    """
    splits = []
    offset = 0
    for size in component_sizes:
        splits.append(beliefs[..., offset:offset + size])
        offset += size
    weights = jnp.stack([block.sum(axis=-1) for block in splits], axis=-1)
    component_beliefs = [
        block / jnp.maximum(block.sum(axis=-1, keepdims=True), 1e-30)
        for block in splits
    ]
    return weights, component_beliefs


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
    fps: int = 20,
):
    import numpy as np
    import matthewplotlib as mp
    import time

    GENERATORS = {
        "mess3": MESS3,
        "zor": ZOR,
        "alt": ALT,
        "fig1a": FIG1A_TOP * FIG1A_BOTTOM,
        "union": mess3(0.6, 0.15) + mess3(0.85, 0.05),
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
    if generator == "union":

        # decompose joint beliefs into mixture weights + per-component
        weights, comp_beliefs = decompose_union_beliefs(
            beliefs, [3, 3],
        )
        w1 = weights[:, 0]  # mixture weight for component 1

        # 2D simplex coordinates for each component's beliefs
        def simplex_xy(b):
            return -0.5 + b[:, 1] + 0.5 * b[:, 2], (np.sqrt(3) / 2) * b[:, 2]
        x1, y1 = simplex_xy(np.array(comp_beliefs[0]))
        x2, y2 = simplex_xy(np.array(comp_beliefs[1]))
        w1 = np.array(w1)

        # place triangle 1 at z=+1.5, triangle 2 at z=-1.5
        # interpolate each point's 3D position by mixture weight
        sep = 1.0
        pts = np.stack([
            w1 * x1 + (1 - w1) * x2,
            w1 * y1 + (1 - w1) * y2,
            w1 * sep - (1 - w1) * sep,
        ], axis=-1)

        # color: red = component 1, blue = component 2
        cs = np.stack([
            w1,
            0.2 * np.ones(len(w1)),
            1 - w1,
        ], axis=-1)
        cs = (cs * 255).astype(np.uint8)

        # triangle wireframes at z = ±sep
        tri_verts = np.array([
            [-0.5, 0], [0.5, 0], [0, np.sqrt(3)/2],
        ])
        edge_pts_list = []
        for z in [sep, -sep]:
            for i in range(3):
                j = (i + 1) % 3
                t = np.linspace(0, 1, 200)[:, None]
                edge_2d = tri_verts[i] * (1 - t) + tri_verts[j] * t
                edge_3d = np.column_stack([
                    edge_2d, np.full(200, z),
                ])
                edge_pts_list.append(edge_3d)
        edge_pts = np.concatenate(edge_pts_list)
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
                height=30,
                width=80,
            )
            print(plot)
            frame += 1
            time.sleep(1 / fps)

    elif generator == "fig1a":

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

