import datetime
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import einops
import optax

import tyro
import tqdm

import matthewplotlib as mp

import strux
from transformer import SequenceTransformer
from generators import (
    Sequence, SequenceGenerator, mess3, noisy_channel,
    FIG1A_TOP, FIG1A_BOTTOM,
)

from typing import Self, Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray


def visualise(
    step,
    factors,
    factor_beliefs,      # list of (num_points,) per-factor P(S+) ground truth
    factor_predictions,  # list of (num_points,) per-factor P(S+) predicted
    joint_beliefs,       # (num_points, num_joint_states) for tetrahedron
    joint_predictions,   # (num_points, num_joint_states) probed
    train_losses,
    probe_losses,        # list of (step, total_mse, *per_factor_mse)
    vis_period,
):
    xmax = max(step, 1)

    # loss curve
    steps = np.arange(len(train_losses)) * vis_period
    loss_plot = mp.axes(
        mp.scatter(
            (steps, np.array(train_losses), 'cyan'),
            height=6,
            width=20,
            xrange=(0, xmax),
            yrange=(0, max(max(train_losses), 1.2)),
        ),
        title=" loss ",
        ylabel="nats",
    )

    # probe MSE curves
    probe_steps = np.array([s for s, *_ in probe_losses])
    total_mse = np.array([v[0] for _, *v in probe_losses])
    per_factor_mses = [
        np.array([v[i] for _, *v in probe_losses])
        for i in range(1, 1 + len(factors))
    ]
    joint_mse = np.array([v[-1] for _, *v in probe_losses])

    ymax_probe = max(float(total_mse.max()), float(joint_mse.max()), 0.01)
    probe_layers = [
        (probe_steps, total_mse, 'magenta'),
        (probe_steps, joint_mse, 'yellow'),
    ]
    colours = ['red', 'green', 'blue']
    for i, mse in enumerate(per_factor_mses):
        probe_layers.append(
            (probe_steps, mse, colours[i % len(colours)])
        )
    probe_loss_plot = mp.axes(
        mp.scatter(
            *probe_layers,
            height=6,
            width=20,
            xrange=(0, xmax),
            yrange=(0, ymax_probe),
        ),
        title=" probe mse (magenta=factored, yellow=joint, r/g=factors) ",
    )

    # per-factor probe accuracy: gt P(S+) vs probed P(S+)
    factor_plot_list = []
    for i, (factor, gt, pred) in enumerate(
        zip(factors, factor_beliefs, factor_predictions)
    ):
        factor_plot_list.append(mp.axes(
            mp.scatter(
                (np.array(gt), np.array(pred), colours[i % len(colours)]),
                height=6,
                width=14,
                xrange=(-0.1, 1.1),
                yrange=(-0.1, 1.1),
            ),
            title=f" factor {i} ",
            xlabel="true P(S+)",
            ylabel="probed",
        ))

    # rotating tetrahedron: ground truth beliefs in joint 4-state space
    jb = np.array(joint_beliefs)
    jp = np.array(joint_predictions)
    # colour by factor marginals (reshape 4 joint states as 2x2)
    n_factors = len(factors)
    n_states = [f.num_states for f in factors]
    b_reshaped = jb.reshape(-1, *n_states)
    b1 = b_reshaped.sum(axis=2)[:, -1]  # P(factor0 = S+)
    b2 = b_reshaped.sum(axis=1)[:, -1]  # P(factor1 = S+)
    gt_cs = np.stack([b1, b2, 0.5 * np.ones(len(b1))], axis=-1)
    gt_cs = (gt_cs * 255).astype(np.uint8)

    # embed 3-simplex as regular tetrahedron
    verts = np.array([[-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1]],
                     dtype=float)
    gt_pts = jb @ verts
    pr_pts = jp @ verts

    # tetrahedron edges
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            t = np.linspace(0, 1, 200)[:, None]
            edges.append(verts[i] * (1 - t) + verts[j] * t)
    edge_pts = np.concatenate(edges)
    edge_cs = np.full((len(edge_pts), 3), 80, dtype=np.uint8)

    # probed points: magenta
    pr_cs = np.full((len(pr_pts), 3), 0, dtype=np.uint8)
    pr_cs[:, 0] = 255
    pr_cs[:, 2] = 255

    # camera orbit
    angle = 0.3 * step / 64
    orbit_radius = 5.0
    cam = np.array([
        orbit_radius * np.sin(angle),
        1.5,
        orbit_radius * np.cos(angle),
    ])

    gt_tetra = mp.border(
        mp.scatter3(
            (edge_pts, edge_cs),
            (gt_pts, gt_cs),
            camera_position=cam,
            vertical_fov_degrees=30,
            height=7,
            width=14,
        ),
        title="truth",
    )
    pr_tetra = mp.border(
        mp.scatter3(
            (edge_pts, edge_cs),
            (pr_pts, pr_cs),
            camera_position=cam,
            vertical_fov_degrees=30,
            height=7,
            width=14,
        ),
        title="probed",
    )

    return mp.hstack(
        mp.vstack(loss_plot, probe_loss_plot),
        mp.vstack(*factor_plot_list),
        mp.vstack(gt_tetra, pr_tetra),
    )


def main(
    # data config
    sequence_length: int  = 8,
    generator: Literal[
        'fig1a',
        'mess3x2',
    ]                     = 'fig1a',
    # model config
    num_blocks: int       = 4,
    embed_size: int       = 120,
    num_heads: int        = 3,
    head_size: int        = 40,
    mlp_size: int         = 480,
    # training config
    learning_rate: float  = 5e-4,
    batch_size: int       = 256,
    num_steps: int        = 1024 * 1024,
    opt: Literal[
        'sgd',
        'adam',
    ]                     = 'adam',
    # probe config
    probe_layer: int      = -1,
    probe_num_seqs: int   = 512,
    vis_period: int       = 64,
    # noise config
    epsilon: float        = 0.0,
    # output config
    results_file: str     = '',
    # experiment config
    seed: int             = 42,
    train: bool           = True,
):
    args = ' '.join(sys.argv)
    start_time = datetime.datetime.now()
    print("configuration:")
    config = locals()
    for config_key, config_value in config.items():
        print(f"* {config_key:30s}: {config_value!r}")
    key = jax.random.key(seed=seed)


    print("initialising training distribution...")
    key_tasks, key = jax.random.split(key)
    if generator == 'fig1a':
        factors = [FIG1A_TOP, FIG1A_BOTTOM]
    elif generator == 'mess3x2':
        factors = [mess3(), mess3()]

    sequence_generator = factors[0]
    for f in factors[1:]:
        sequence_generator = sequence_generator * f

    if epsilon > 0:
        sequence_generator = noisy_channel(sequence_generator, epsilon)

    factor_num_symbols = [f.num_symbols for f in factors]
    factor_num_states = [f.num_states for f in factors]
    total_belief_dim = sum(factor_num_states)
    print(f"factors: {len(factors)}, symbols per factor: {factor_num_symbols}")
    print(f"joint symbols: {sequence_generator.num_symbols}, "
          f"joint states: {sequence_generator.num_states}")

    print("example sequences")
    key_examples, key = jax.random.split(key)
    for i in range(10):
        key_gen, key_examples = jax.random.split(key_examples)
        seq = sequence_generator.sample(
            key=key_gen,
            sequence_length=sequence_length,
        )
        print("*", seq)


    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = SequenceTransformer.init(
        key=key_model,
        num_symbols=sequence_generator.num_symbols,
        sequence_length=sequence_length,
        num_blocks=num_blocks,
        embed_size=embed_size,
        num_heads=num_heads,
        head_size=head_size,
        mlp_size=mlp_size,
    )
    print(strux.size(model), "parameters")


    print("initialising optimiser")
    if opt == 'sgd':
        optimiser = optax.sgd(learning_rate=learning_rate)
    elif opt == 'adam':
        optimiser = optax.adam(learning_rate=learning_rate)
    opt_state = optimiser.init(model)


    print("preparing probe eval set...")
    key_probe, key = jax.random.split(key)
    probe_batch = sequence_generator.sample_batch(
        key_probe,
        sequence_length=sequence_length,
        batch_size=probe_num_seqs,
    )
    probe_symbols = probe_batch.symbols

    # decompose joint symbols into per-factor sub-tokens and compute beliefs
    # joint symbol x = z1 * s2 + z2  (for 2 factors)
    # generalise: x = z1 * (s2*s3*...) + z2 * (s3*...) + ... + zN
    def decompose_symbols(joint_symbols):
        """Decompose joint symbols into per-factor sub-token sequences."""
        sub_tokens = []
        remainder = joint_symbols
        for i in range(len(factors)):
            # product of all subsequent factor symbol counts
            divisor = 1
            for j in range(i + 1, len(factors)):
                divisor *= factor_num_symbols[j]
            sub_tok = remainder // divisor
            remainder = remainder % divisor
            sub_tokens.append(sub_tok)
        return sub_tokens

    sub_tokens_list = decompose_symbols(probe_symbols)

    # compute per-factor belief states
    per_factor_beliefs = []
    for i, factor in enumerate(factors):
        # (probe_num_seqs, sequence_length+1, num_states_i)
        beliefs_i = jax.vmap(factor.belief_states)(sub_tokens_list[i])
        # drop prior, align with model positions
        beliefs_i = beliefs_i[:, 1:, :]
        per_factor_beliefs.append(beliefs_i)

    # concatenated probe targets: (probe_num_seqs * sequence_length, total_belief_dim)
    probe_beliefs_flat = jnp.concatenate(
        [b.reshape(-1, b.shape[-1]) for b in per_factor_beliefs],
        axis=1,
    )

    # joint beliefs from the (possibly noisy) generator for joint probe
    joint_beliefs_all = jax.vmap(sequence_generator.belief_states)(probe_symbols)
    joint_beliefs_all = joint_beliefs_all[:, 1:, :]  # drop prior
    joint_beliefs_flat = joint_beliefs_all.reshape(-1, sequence_generator.num_states)

    # for visualisation: per-factor P(S+) (last state)
    vis_n = min(500, probe_num_seqs * sequence_length)
    factor_beliefs_vis = [
        b.reshape(-1, b.shape[-1])[:vis_n, -1]
        for b in per_factor_beliefs
    ]


    print("defining train step...")
    @jax.jit
    def train_step(key, model, opt_state):
        sequences = sequence_generator.sample_batch(
            key,
            sequence_length=sequence_length+1,
            batch_size=batch_size,
        )
        def loss_fn(model, symbols):
            next_symbols_pred = model.forward_batch(
                symbols[:, :-1]
            ).reshape(-1, model.num_symbols)
            next_symbols_target = jax.nn.one_hot(
                symbols[:, 1:].reshape(-1),
                num_classes=model.num_symbols,
            )
            per_token_losses = optax.softmax_cross_entropy(
                logits=next_symbols_pred,
                labels=next_symbols_target,
            )
            return per_token_losses.mean()
        loss, grads = jax.value_and_grad(loss_fn)(model, sequences.symbols)
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss


    @jax.jit
    def eval_loss(key, model):
        sequences = sequence_generator.sample_batch(
            key,
            sequence_length=sequence_length+1,
            batch_size=batch_size,
        )
        logits = model.forward_batch(
            sequences.symbols[:, :-1],
        ).reshape(-1, model.num_symbols)
        targets = jax.nn.one_hot(
            sequences.symbols[:, 1:].reshape(-1),
            num_classes=model.num_symbols,
        )
        return optax.softmax_cross_entropy(
            logits=logits, labels=targets,
        ).mean()


    @jax.jit
    def probe(model):
        # get activations at probe_layer
        _, activations = model.forward_batch_with_activations(probe_symbols)
        acts = activations[:, probe_layer, :, :]
        acts_flat = acts.reshape(-1, acts.shape[-1])
        # affine least-squares probe: [activations | 1] @ W ≈ beliefs
        ones = jnp.ones((acts_flat.shape[0], 1))
        X = jnp.concatenate([acts_flat, ones], axis=1)
        # factored probe: predict per-factor beliefs from sub-token decomposition
        W, _, _, _ = jnp.linalg.lstsq(X, probe_beliefs_flat)
        predicted = X @ W
        # total MSE
        total_mse = jnp.mean((probe_beliefs_flat - predicted) ** 2)
        # per-factor MSE
        offset = 0
        per_factor_mse = []
        for ns in factor_num_states:
            factor_pred = predicted[:, offset:offset+ns]
            factor_true = probe_beliefs_flat[:, offset:offset+ns]
            per_factor_mse.append(
                jnp.mean((factor_true - factor_pred) ** 2)
            )
            offset += ns
        # joint probe: predict true Bayesian beliefs from noisy joint generator
        W_joint, _, _, _ = jnp.linalg.lstsq(X, joint_beliefs_flat)
        predicted_joint = X @ W_joint
        joint_mse = jnp.mean((joint_beliefs_flat - predicted_joint) ** 2)
        return predicted, total_mse, per_factor_mse, predicted_joint, joint_mse


    if not train: return

    print("starting training loop...")
    # seed with initial model loss and probe
    key_eval, key = jax.random.split(key)
    train_losses = [float(eval_loss(key_eval, model))]
    predicted, total_mse, per_factor_mse, predicted_joint, joint_mse = probe(model)
    probe_losses = [(0, float(total_mse), *[float(m) for m in per_factor_mse], float(joint_mse))]

    # extract per-factor predicted P(S+) for scatter plots
    def extract_factor_predictions(predicted):
        factor_preds = []
        offset = 0
        for ns in factor_num_states:
            factor_preds.append(np.array(predicted[:vis_n, offset + ns - 1]))
            offset += ns
        return factor_preds

    # true joint beliefs for tetrahedron (from the possibly-noisy generator)
    joint_beliefs_vis = np.array(joint_beliefs_flat[:vis_n])

    factor_preds_vis = extract_factor_predictions(predicted)
    joint_preds_vis = np.array(predicted_joint[:vis_n])
    plot = visualise(
        0, factors, factor_beliefs_vis, factor_preds_vis,
        joint_beliefs_vis, joint_preds_vis,
        train_losses, probe_losses, vis_period,
    )
    print(plot)

    # train!
    for t in tqdm.trange(num_steps):
        key_sgd, key = jax.random.split(key)
        model, opt_state, loss = train_step(
            key_sgd,
            model,
            opt_state,
        )

        if (t + 1) % vis_period == 0:
            train_losses.append(float(loss))
            predicted, total_mse, per_factor_mse, predicted_joint, joint_mse = probe(model)
            probe_losses.append(
                (t + 1, float(total_mse), *[float(m) for m in per_factor_mse], float(joint_mse))
            )
            factor_preds_vis = extract_factor_predictions(predicted)
            joint_preds_vis = np.array(predicted_joint[:vis_n])
            new_plot = visualise(
                t + 1, factors, factor_beliefs_vis, factor_preds_vis,
                joint_beliefs_vis, joint_preds_vis,
                train_losses, probe_losses, vis_period,
            )
            tqdm.tqdm.write(f"{-plot}{new_plot}")
            plot = new_plot


    # log final probe MSEs to results file
    if results_file:
        final_probe = probe_losses[-1]
        # final_probe = (step, total_mse, *per_factor_mse, joint_mse)
        result = {
            "epsilon": epsilon,
            "factored_mse": final_probe[1],
            "joint_mse": final_probe[-1],
            "train_loss": train_losses[-1],
            "seed": seed,
        }
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"wrote result to {results_file}")

    print("done!")
    end_time = datetime.datetime.now()


if __name__ == "__main__":
    tyro.cli(main)
