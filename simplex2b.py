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
)

from typing import Self, Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray


def simplex_project(beliefs_3state):
    """Project 3-state beliefs to 2D simplex triangle coordinates."""
    xs = beliefs_3state[:, 1] + 0.5 * beliefs_3state[:, 2]
    ys = (np.sqrt(3) / 2) * beliefs_3state[:, 2]
    return xs, ys


def visualise(
    step,
    factors,
    factor_beliefs_3,    # list of (num_points, 3) per-factor belief vectors
    factor_predicted_3,  # list of (num_points, 3) per-factor probed beliefs
    train_losses,
    probe_losses,        # list of (step, total_mse, *per_factor_mse, joint_mse)
    r2_history,          # list of (step, factored_r2, joint_r2)
    dims_history,        # list of (step, dims_95)
    vis_period,
    factored_dof,        # number of factored degrees of freedom (for reference line)
):
    xmax = max(step, 1)
    wide_w = 72
    simplex_w = 20
    colours = ['red', 'green', 'blue', 'cyan', 'white']

    # loss curve
    steps = np.arange(len(train_losses)) * vis_period
    loss_plot = mp.axes(
        mp.scatter(
            (steps, np.array(train_losses), 'cyan'),
            height=6,
            width=wide_w,
            xrange=(0, xmax),
            yrange=(0, max(max(train_losses), 1.2)),
        ),
        title=f" loss ({train_losses[-1]:.4f}) ",
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
    for i, mse in enumerate(per_factor_mses):
        probe_layers.append(
            (probe_steps, mse, colours[i % len(colours)])
        )
    probe_loss_plot = mp.axes(
        mp.scatter(
            *probe_layers,
            height=6,
            width=wide_w,
            xrange=(0, xmax),
            yrange=(0, ymax_probe),
        ),
        title=" probe mse (magenta=factored, yellow=joint, colours=factors) ",
    )

    # R² curves
    r2_steps = np.array([s for s, *_ in r2_history])
    factored_r2 = np.array([v[0] for _, *v in r2_history])
    joint_r2 = np.array([v[1] for _, *v in r2_history])
    r2_plot = mp.axes(
        mp.scatter(
            (r2_steps, factored_r2, 'magenta'),
            (r2_steps, joint_r2, 'yellow'),
            height=6,
            width=wide_w,
            xrange=(0, xmax),
            yrange=(0, 1),
        ),
        title=" R² (magenta=factored, yellow=joint) ",
    )

    # PCA dimensionality plot
    dims_steps = np.array([s for s, *_ in dims_history])
    dims_95 = np.array([v for _, v in dims_history])
    dims_max = max(float(dims_95.max()), factored_dof + 2, 1)
    # reference line at factored DoF
    ref_x = np.array([0, xmax])
    ref_y = np.array([factored_dof, factored_dof])
    dims_plot = mp.axes(
        mp.scatter(
            (ref_x, ref_y, 'green'),
            (dims_steps, dims_95, 'yellow'),
            height=6,
            width=wide_w,
            xrange=(0, xmax),
            yrange=(0, dims_max),
        ),
        title=f" PCA dims for 95% variance (green=factored DoF={factored_dof}) ",
    )

    # per-factor simplex plots: ground truth beliefs as coloured triangle
    factor_plot_list = []
    for i, (gt, pred) in enumerate(
        zip(factor_beliefs_3, factor_predicted_3)
    ):
        gt = np.array(gt)
        pred = np.array(pred)
        gt_cs = (gt * 255).astype(np.uint8)
        pred_cs = np.full((len(pred), 3), 0, dtype=np.uint8)
        pred_cs[:, 0] = 255
        pred_cs[:, 2] = 255  # magenta
        gt_x, gt_y = simplex_project(gt)
        pr_x, pr_y = simplex_project(pred)
        factor_plot_list.append(mp.axes(
            mp.scatter(
                (gt_x, gt_y, gt_cs),
                (pr_x, pr_y, pred_cs),
                height=simplex_w // 2,
                width=simplex_w,
                xrange=(-0.05, 1.05),
                yrange=(-0.05, 0.95),
            ),
            title=f" F{i} ",
        ))

    return mp.vstack(
        loss_plot,
        probe_loss_plot,
        r2_plot,
        dims_plot,
        mp.wrap(*factor_plot_list, cols=3),
    )


def main(
    # data config
    sequence_length: int  = 8,
    num_factors: int      = 3,
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
    alphas = np.linspace(0.55, 0.85, num_factors)
    xs = np.linspace(0.10, 0.45, num_factors)
    factors = [mess3(alpha=float(a), x=float(x)) for a, x in zip(alphas, xs)]
    for i, (a, x) in enumerate(zip(alphas, xs)):
        print(f"  F{i}: alpha={a:.3f}, x={x:.3f}")

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
    # joint symbol x = z1 * (s2*s3*...) + z2 * (s3*...) + ... + zN
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

    # for visualisation: per-factor full 3-state beliefs
    vis_n = min(500, probe_num_seqs * sequence_length)
    factor_beliefs_vis = [
        b.reshape(-1, b.shape[-1])[:vis_n]
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


    def compute_r2(true, predicted):
        """Compute R² (coefficient of determination)."""
        ss_res = jnp.sum((true - predicted) ** 2)
        ss_tot = jnp.sum((true - jnp.mean(true, axis=0)) ** 2)
        return 1 - ss_res / ss_tot


    @jax.jit
    def probe(model):
        # get activations at probe_layer
        _, activations = model.forward_batch_with_activations(probe_symbols)
        acts = activations[:, probe_layer, :, :]
        acts_flat = acts.reshape(-1, acts.shape[-1])
        # affine least-squares probe: [activations | 1] @ W ≈ beliefs
        ones = jnp.ones((acts_flat.shape[0], 1))
        X = jnp.concatenate([acts_flat, ones], axis=1)
        # factored probe: predict per-factor beliefs
        W_fac, _, _, _ = jnp.linalg.lstsq(X, probe_beliefs_flat)
        predicted_fac = X @ W_fac
        # total factored MSE
        factored_mse = jnp.mean((probe_beliefs_flat - predicted_fac) ** 2)
        # per-factor MSE
        offset = 0
        per_factor_mse = []
        for ns in factor_num_states:
            factor_pred = predicted_fac[:, offset:offset+ns]
            factor_true = probe_beliefs_flat[:, offset:offset+ns]
            per_factor_mse.append(
                jnp.mean((factor_true - factor_pred) ** 2)
            )
            offset += ns
        # factored R²
        factored_r2 = compute_r2(probe_beliefs_flat, predicted_fac)
        # joint probe: predict true Bayesian beliefs from joint generator
        W_jnt, _, _, _ = jnp.linalg.lstsq(X, joint_beliefs_flat)
        predicted_jnt = X @ W_jnt
        joint_mse = jnp.mean((joint_beliefs_flat - predicted_jnt) ** 2)
        # joint R²
        joint_r2 = compute_r2(joint_beliefs_flat, predicted_jnt)
        # PCA: dims for 95% variance
        acts_centered = acts_flat - jnp.mean(acts_flat, axis=0)
        S = jnp.linalg.svd(acts_centered, full_matrices=False, compute_uv=False)
        explained = jnp.cumsum(S ** 2) / jnp.sum(S ** 2)
        dims_95 = jnp.argmax(explained >= 0.95) + 1
        return (
            predicted_fac, factored_mse, per_factor_mse, factored_r2,
            predicted_jnt, joint_mse, joint_r2,
            dims_95,
        )


    if not train: return

    # factored degrees of freedom: each factor's beliefs sum to 1
    factored_dof = sum(ns - 1 for ns in factor_num_states)

    print("starting training loop...")
    # seed with initial model loss and probe
    key_eval, key = jax.random.split(key)
    train_losses = [float(eval_loss(key_eval, model))]
    (predicted_fac, factored_mse, per_factor_mse, factored_r2,
     predicted_jnt, joint_mse, joint_r2, dims_95) = probe(model)
    probe_losses = [
        (0, float(factored_mse), *[float(m) for m in per_factor_mse], float(joint_mse))
    ]
    r2_history = [(0, float(factored_r2), float(joint_r2))]
    dims_history = [(0, int(dims_95))]

    # extract per-factor predicted belief vectors for simplex plots
    def extract_factor_predictions(predicted):
        factor_preds = []
        offset = 0
        for ns in factor_num_states:
            factor_preds.append(np.array(predicted[:vis_n, offset:offset+ns]))
            offset += ns
        return factor_preds

    factor_preds_vis = extract_factor_predictions(predicted_fac)
    plot = visualise(
        0, factors, factor_beliefs_vis, factor_preds_vis,
        train_losses, probe_losses, r2_history, dims_history,
        vis_period, factored_dof,
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
            (predicted_fac, factored_mse, per_factor_mse, factored_r2,
             predicted_jnt, joint_mse, joint_r2, dims_95) = probe(model)
            probe_losses.append(
                (t + 1, float(factored_mse), *[float(m) for m in per_factor_mse], float(joint_mse))
            )
            r2_history.append((t + 1, float(factored_r2), float(joint_r2)))
            dims_history.append((t + 1, int(dims_95)))
            factor_preds_vis = extract_factor_predictions(predicted_fac)
            new_plot = visualise(
                t + 1, factors, factor_beliefs_vis, factor_preds_vis,
                train_losses, probe_losses, r2_history, dims_history,
                vis_period, factored_dof,
            )
            tqdm.tqdm.write(f"{-plot}{new_plot}")
            plot = new_plot


    # log final results
    if results_file:
        final_probe = probe_losses[-1]
        final_r2 = r2_history[-1]
        final_dims = dims_history[-1]
        result = {
            "num_factors": num_factors,
            "epsilon": epsilon,
            "factored_mse": final_probe[1],
            "joint_mse": final_probe[-1],
            "factored_r2": final_r2[1],
            "joint_r2": final_r2[2],
            "dims_95": final_dims[1],
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
