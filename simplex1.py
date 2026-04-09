import datetime
import json
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
from generators import Sequence, MESS3, ZOR, ALT

from typing import Self, Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray


# HYPER-PARAMETER   Shai+2024   Notes
# ----------------------------------------------------------------------------
# max_examples      10
# num_blocks        4
# embed_size        64
# num_heads         1
# head_size         8           num_heads x head_size NOT equal to embed size!
# mlp_size          256
# learning_rate     0.01
# batch_size        64
# optimiser         SGD
# num_steps         1 million   seems like overkill


def visualise(
    step,
    gt_simplex,
    probe_predicted,
    train_losses,
    probe_losses,
    vis_period,
):
    gt_xs, gt_ys, gt_cs = (np.array(x) for x in gt_simplex)
    gt_cs = (gt_cs * 255).astype(np.uint8)
    predicted = np.array(probe_predicted)

    # probed residual stream simplex
    pr_xs = predicted[:, 1] + 0.5 * predicted[:, 2]
    pr_ys = (np.sqrt(3) / 2) * predicted[:, 2]
    pr_cs = (np.clip(predicted, 0, 1) * 255).astype(np.uint8)

    # ground truth simplex
    gt_plot = mp.axes(
        mp.scatter(
            (gt_xs, gt_ys, gt_cs),
            height=15,
            width=30,
            xrange=(-0.2, 1.2),
            yrange=(-0.2, 1.2),
        ),
        title=" ground truth ",
    )

    # probed residual stream simplex
    pr_plot = mp.axes(
        mp.scatter(
            (pr_xs, pr_ys, pr_cs),
            height=15,
            width=30,
            xrange=(-0.2, 1.2),
            yrange=(-0.2, 1.2),
        ),
        title=" residual stream ",
    )

    # loss curves
    steps = np.arange(len(train_losses)) * vis_period
    probe_steps = np.array([s for s, _ in probe_losses])
    probe_vals = np.array([v for _, v in probe_losses])

    xmax = max(step, 1)

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

    probe_loss_plot = mp.axes(
        mp.scatter(
            (probe_steps, probe_vals, 'magenta'),
            height=6,
            width=20,
            xrange=(0, xmax),
            yrange=(0, max(max(probe_vals), 0.01)),
        ),
        title=" probe mse ",
    )

    return gt_plot + pr_plot + (loss_plot / probe_loss_plot)


def main(
    # data config
    sequence_length: int  = 10,
    generator: Literal[
        'mess3',
        'zor',
        'alt',
    ]                     = 'mess3',
    # model config
    num_blocks: int       = 4,
    embed_size: int       = 64,
    num_heads: int        = 1,
    head_size: int        = 8,
    mlp_size: int         = 256,
    # training config
    learning_rate: float  = 0.01,
    batch_size: int       = 64,
    num_steps: int        = 1024 * 1024,
    opt: Literal[
        'sgd',
        'adam',
    ]                     = 'sgd',
    # probe config
    probe_layer: int      = -1,
    probe_num_seqs: int   = 512,
    vis_period: int       = 64,
    # display
    vis: bool             = True,
    # logging
    metrics_file: str     = "",
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
    if generator == 'zor':
        sequence_generator = ZOR
    elif generator == 'mess3':
        sequence_generator = MESS3
    elif generator == 'alt':
        sequence_generator = ALT

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
    # ground truth belief states: (probe_num_seqs, sequence_length+1, num_states)
    probe_beliefs_all = jax.vmap(
        sequence_generator.belief_states,
    )(probe_batch.symbols)
    # drop the prior, beliefs[i] = P(state | symbols[0:i+1])
    # aligns with model activations at position i
    probe_beliefs_flat = probe_beliefs_all[:, 1:, :].reshape(
        -1, sequence_generator.num_states,
    )
    probe_symbols = probe_batch.symbols

    # project ground truth beliefs to 2D simplex for visualisation
    gt_simplex = (
        probe_beliefs_flat[:, 1] + 0.5 * probe_beliefs_flat[:, 2],
        (jnp.sqrt(3) / 2) * probe_beliefs_flat[:, 2],
        probe_beliefs_flat,
    )


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
        W, _, _, _ = jnp.linalg.lstsq(X, probe_beliefs_flat)
        predicted = X @ W
        probe_loss = jnp.mean((probe_beliefs_flat - predicted) ** 2)
        return predicted, probe_loss


    if not train: return

    print("starting training loop...")
    # seed with initial model loss and probe
    key_eval, key = jax.random.split(key)
    predicted, p_loss = probe(model)

    if vis:
        train_losses = [float(eval_loss(key_eval, model))]
        probe_losses = [(0, float(p_loss))]
        plot = visualise(0, gt_simplex, predicted, train_losses, probe_losses, vis_period)
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
            current_loss = float(loss)
            predicted, p_loss = probe(model)

            if metrics_file:
                metric = {
                    "step": t + 1,
                    "train_loss": current_loss,
                    "probe_mse": float(p_loss),
                }
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metric) + "\n")

            if vis:
                train_losses.append(current_loss)
                probe_losses.append((t + 1, float(p_loss)))
                new_plot = visualise(
                    t + 1, gt_simplex, predicted,
                    train_losses, probe_losses, vis_period,
                )
                tqdm.tqdm.write(f"{-plot}{new_plot}")
                plot = new_plot


    print("done!")
    end_time = datetime.datetime.now()


if __name__ == "__main__":
    tyro.cli(main)
