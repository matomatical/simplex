import datetime
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

import tyro
import tqdm

import matthewplotlib as mp

import strux
from transformer import SequenceTransformer
from generators import mess3, decompose_union_beliefs
from baselines import compute_baselines

from typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray


def simplex_project(beliefs):
    """Project 3-state beliefs to 2D simplex coordinates."""
    xs = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    ys = (np.sqrt(3) / 2) * beliefs[:, 2]
    return xs, ys


def _loss_plot(title, steps, values, baseline, xmax, yrange,
               width=15, height=6):
    """Helper for a single loss-over-training plot with a baseline line."""
    bx = np.linspace(0, max(xmax, 1), 100)
    by = np.full_like(bx, baseline)
    return mp.axes(
        mp.scatter(
            (steps, np.array(values), 'cyan'),
            (bx, by, 'red'),
            height=height,
            width=width,
            xrange=(0, max(xmax, 1)),
            yrange=yrange,
        ),
        title=title,
    )


def _mse_plot(title, steps, values, color, xmax, yrange,
              width=15, height=6):
    """Helper for a single MSE-over-training plot."""
    return mp.axes(
        mp.scatter(
            (np.array(steps), np.array(values), color),
            height=height,
            width=width,
            xrange=(0, max(xmax, 1)),
            yrange=yrange,
        ),
        title=title,
    )


def visualise(
    step,
    train_losses,
    probe_losses,
    subsystem_mses,
    factored_mses,
    position_losses,
    bayes_optimal,
    bayes_per_position,
    marginal_entropy,
    vis_period,
    # ground truth decomposed beliefs
    gt_comp_beliefs,
    gt_weights,
    # joint probe decomposed beliefs (unclipped)
    pr_comp_beliefs,
    pr_weights,
    # factored probe direct outputs
    fr_comp_beliefs,
    fr_weights,
    # true component labels for coloring w(t) scatter
    true_component,
    sequence_length,
):
    xmax = max(step, 1)
    pos_mid = sequence_length // 2
    pos_end = sequence_length - 1

    # --- row 1: overall loss + per-position loss ---
    steps = np.arange(len(train_losses)) * vis_period
    pos_steps = np.array([s for s, _ in position_losses])
    pos_vals = np.array([v for _, v in position_losses])

    all_loss_vals = np.concatenate([train_losses, pos_vals.ravel()])
    all_baselines = np.concatenate([[bayes_optimal], bayes_per_position])
    loss_ymin = min(float(np.min(all_loss_vals)), float(np.min(all_baselines))) - 0.05
    loss_ymax = max(float(np.max(all_loss_vals)), float(np.max(all_baselines))) + 0.05
    loss_yrange = (loss_ymin, loss_ymax)

    loss_overall = _loss_plot(" loss ", steps, train_losses,
                              bayes_optimal, xmax, loss_yrange)
    loss_pos1 = _loss_plot(" pos 1 ", pos_steps, pos_vals[:, 0],
                           bayes_per_position[0], xmax, loss_yrange)
    loss_mid = _loss_plot(f" pos {pos_mid+1} ", pos_steps, pos_vals[:, pos_mid],
                          bayes_per_position[pos_mid], xmax, loss_yrange)
    loss_end = _loss_plot(f" pos {pos_end+1} ", pos_steps, pos_vals[:, pos_end],
                          bayes_per_position[pos_end], xmax, loss_yrange)

    # --- rows 2-3: joint and factored MSE (shared y-range) ---
    probe_steps = np.array([s for s, _ in probe_losses])
    probe_vals = np.array([v for _, v in probe_losses])

    j_steps = np.array([s for s, _, _, _ in subsystem_mses])
    j_c1 = np.array([c1 for _, c1, _, _ in subsystem_mses])
    j_c2 = np.array([c2 for _, _, c2, _ in subsystem_mses])
    j_w = np.array([w for _, _, _, w in subsystem_mses])

    f_steps = np.array([s for s, _, _, _ in factored_mses])
    f_c1 = np.array([c1 for _, c1, _, _ in factored_mses])
    f_c2 = np.array([c2 for _, _, c2, _ in factored_mses])
    f_w = np.array([w for _, _, _, w in factored_mses])
    f_total = (f_c1 + f_c2 + f_w) / 3

    all_mse = np.concatenate([probe_vals, j_c1, j_c2, j_w,
                              f_total, f_c1, f_c2, f_w])
    mse_ymax = max(float(np.max(all_mse)), 0.01)
    mse_yrange = (0, mse_ymax)

    # row 2: joint probe MSEs
    mse_j_total = _mse_plot(" joint MSE ", probe_steps, probe_vals,
                            'magenta', xmax, mse_yrange)
    mse_j_c1 = _mse_plot(" j.comp1 ", j_steps, j_c1,
                          'red', xmax, mse_yrange)
    mse_j_c2 = _mse_plot(" j.comp2 ", j_steps, j_c2,
                          'blue', xmax, mse_yrange)
    mse_j_w = _mse_plot(" j.w ", j_steps, j_w,
                         'yellow', xmax, mse_yrange)

    # row 3: factored probe MSEs
    mse_f_total = _mse_plot(" fact. MSE ", f_steps, f_total,
                            'magenta', xmax, mse_yrange)
    mse_f_c1 = _mse_plot(" f.comp1 ", f_steps, f_c1,
                          'red', xmax, mse_yrange)
    mse_f_c2 = _mse_plot(" f.comp2 ", f_steps, f_c2,
                          'blue', xmax, mse_yrange)
    mse_f_w = _mse_plot(" f.w ", f_steps, f_w,
                         'yellow', xmax, mse_yrange)

    # --- rows 4-6: simplex + w(t) for true, joint, factored ---
    wt_width = 80 - 2 * (16 + 6) - 6
    simplex_panels = []
    for label, comp_beliefs in [("true", gt_comp_beliefs),
                                ("joint", pr_comp_beliefs),
                                ("fact.", fr_comp_beliefs)]:
        row_panels = []
        for i, beliefs in enumerate(comp_beliefs):
            beliefs_np = np.array(beliefs)
            xs, ys = simplex_project(beliefs_np)
            cs = (np.clip(beliefs_np, 0, 1) * 255).astype(np.uint8)
            row_panels.append(mp.axes(
                mp.scatter(
                    (xs, ys, cs),
                    height=8,
                    width=16,
                    xrange=(-0.2, 1.2),
                    yrange=(-0.2, 1.0),
                ),
                title=f" {label} comp{i+1} ",
            ))
        simplex_panels.append(row_panels)

    # w(t) scatter: x = position, y = w1, color by true component
    num_seqs = len(true_component)
    positions = np.tile(np.arange(1, sequence_length + 1), num_seqs)
    comp_mask = np.repeat(true_component, sequence_length)
    cs_wt = np.zeros((len(comp_mask), 3), dtype=np.uint8)
    cs_wt[comp_mask == 0] = [255, 51, 51]   # red for comp 1
    cs_wt[comp_mask == 1] = [51, 51, 255]   # blue for comp 2

    wt_panels = []
    for label, weights in [("true", gt_weights),
                           ("joint", pr_weights),
                           ("fact.", fr_weights)]:
        w1 = np.array(weights).ravel()
        wt_panels.append(mp.axes(
            mp.scatter(
                (positions, w1, cs_wt),
                height=8,
                width=wt_width,
                xrange=(0, sequence_length + 1),
                yrange=(-0.2, 1.2),
            ),
            title=f" {label} w(t) ",
            xlabel="pos",
        ))

    row1 = loss_overall + loss_pos1 + loss_mid + loss_end
    row2 = mse_j_total + mse_j_c1 + mse_j_c2 + mse_j_w
    row3 = mse_f_total + mse_f_c1 + mse_f_c2 + mse_f_w
    row4 = simplex_panels[0][0] + simplex_panels[0][1] + wt_panels[0]
    row5 = simplex_panels[1][0] + simplex_panels[1][1] + wt_panels[1]
    row6 = simplex_panels[2][0] + simplex_panels[2][1] + wt_panels[2]
    return row1 / row2 / row3 / row4 / row5 / row6


def main(
    # data config
    alpha1: float             = 0.6,
    x1: float                 = 0.15,
    alpha2: float             = 0.85,
    x2: float                 = 0.05,
    sequence_length: int      = 32,
    # model config
    num_blocks: int           = 4,
    embed_size: int           = 64,
    num_heads: int            = 2,
    head_size: int            = 16,
    mlp_size: int             = 256,
    # training config
    learning_rate: float      = 0.01,
    batch_size: int           = 64,
    num_steps: int            = 1024 * 1024,
    opt: Literal[
        'sgd',
        'adam',
    ]                         = 'sgd',
    # probe config
    probe_layer: int          = -1,
    probe_num_seqs: int       = 128,
    # display
    vis_period: int           = 64,
    # logging
    metrics_file: str         = "",
    # experiment config
    seed: int                 = 42,
):
    args = ' '.join(sys.argv)
    start_time = datetime.datetime.now()
    print("configuration:")
    config = locals()
    for config_key, config_value in config.items():
        print(f"* {config_key:30s}: {config_value!r}")
    key = jax.random.key(seed=seed)


    print("initialising generator...")
    gen1 = mess3(alpha1, x1)
    gen2 = mess3(alpha2, x2)
    sequence_generator = gen1 + gen2
    component_sizes = [gen1.num_states, gen2.num_states]
    print(f"  union: {sequence_generator.num_symbols} symbols, "
          f"{sequence_generator.num_states} states")


    print("computing baselines...")
    baselines = compute_baselines(sequence_generator, sequence_length=1000)
    bayes_optimal = baselines["bayes_optimal_loss"]
    marginal_entropy = baselines["marginal_entropy"]
    print(f"  max entropy:       {baselines['max_entropy']:.4f} nats")
    print(f"  marginal entropy:  {baselines['marginal_entropy']:.4f} nats")
    print(f"  bayes-optimal:     {bayes_optimal:.4f} nats")
    for i, (a, x) in enumerate([(alpha1, x1), (alpha2, x2)]):
        comp = compute_baselines(mess3(a, x), sequence_length=1000)
        print(f"  component {i+1} (α={a}, x={x}) bayes-optimal: "
              f"{comp['bayes_optimal_loss']:.4f} nats")


    print("example sequences:")
    key_examples, key = jax.random.split(key)
    for i in range(5):
        key_gen, key_examples = jax.random.split(key_examples)
        seq = sequence_generator.sample(key=key_gen, sequence_length=sequence_length)
        print(" *", seq)


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
    print(f"  {strux.size(model)} parameters")


    print("initialising optimiser...")
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

    # ground truth joint beliefs: (probe_num_seqs, seq_len+1, 6)
    probe_beliefs_all = jax.vmap(
        sequence_generator.belief_states,
    )(probe_symbols)
    # drop prior, align with activations: (probe_num_seqs, seq_len, 6)
    probe_beliefs = probe_beliefs_all[:, 1:, :]
    probe_beliefs_flat = probe_beliefs.reshape(-1, sequence_generator.num_states)

    # decompose GT beliefs into per-component + mixture weights
    gt_weights, gt_comp_beliefs = decompose_union_beliefs(
        probe_beliefs, component_sizes,
    )
    # flatten for visualization: (probe_num_seqs * seq_len, ...)
    gt_weights_flat = gt_weights[:, :, 0].reshape(-1)  # w1 only
    gt_comp_beliefs_flat = [b.reshape(-1, b.shape[-1]) for b in gt_comp_beliefs]

    # flat targets for factored probes (jax arrays, captured by JIT)
    gt_comp1_probe_flat = gt_comp_beliefs[0].reshape(-1, 3)
    gt_comp2_probe_flat = gt_comp_beliefs[1].reshape(-1, 3)
    gt_w1_probe_flat = gt_weights[:, :, 0:1].reshape(-1, 1)

    # true component per sequence (from initial state)
    true_component = np.array(probe_batch.states[:, 0] >= gen1.num_states, dtype=int)

    # separate eval batch for per-position loss (needs seq_len+1 symbols)
    key_eval_batch, key = jax.random.split(key)
    eval_batch = sequence_generator.sample_batch(
        key_eval_batch,
        sequence_length=sequence_length + 1,
        batch_size=probe_num_seqs,
    )
    eval_symbols = eval_batch.symbols  # (N, seq_len+1)

    # per-position Bayes-optimal loss on the same eval batch
    T = np.array(sequence_generator.transition_distributions)
    emission_probs = T.sum(axis=2)  # (num_symbols, num_states)
    eval_beliefs_all = np.array(jax.vmap(
        sequence_generator.belief_states,
    )(eval_symbols))  # (N, seq_len+2, num_states)
    eval_pred_beliefs = eval_beliefs_all[:, 1:-1, :]  # (N, seq_len, num_states)
    eval_pred_dists = eval_pred_beliefs @ emission_probs.T
    eval_targets = np.array(eval_symbols[:, 1:])
    bayes_per_token = -np.log(
        eval_pred_dists[
            np.arange(probe_num_seqs)[:, None],
            np.arange(sequence_length)[None, :],
            eval_targets,
        ] + 1e-10
    )
    bayes_per_position = bayes_per_token.mean(axis=0)
    print(f"  bayes per-position loss: {bayes_per_position}")


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
        updates, opt_state_new = optimiser.update(grads, opt_state, model)
        model_new = optax.apply_updates(model, updates)
        return model_new, opt_state_new, loss


    @jax.jit
    def probe_all(model):
        _, activations = model.forward_batch_with_activations(probe_symbols)
        acts = activations[:, probe_layer, :, :]
        acts_flat = acts.reshape(-1, acts.shape[-1])
        ones = jnp.ones((acts_flat.shape[0], 1))
        X = jnp.concatenate([acts_flat, ones], axis=1)

        # joint probe: activations → 6D belief
        W_j, _, _, _ = jnp.linalg.lstsq(X, probe_beliefs_flat)
        pred_joint = X @ W_j
        joint_mse = jnp.mean((probe_beliefs_flat - pred_joint) ** 2)
        pred_joint_2d = pred_joint.reshape(probe_beliefs.shape)

        # factored probes: activations → b1, b2, w1 separately
        W_b1, _, _, _ = jnp.linalg.lstsq(X, gt_comp1_probe_flat)
        pred_b1_flat = X @ W_b1
        b1_mse = jnp.mean((gt_comp1_probe_flat - pred_b1_flat) ** 2)

        W_b2, _, _, _ = jnp.linalg.lstsq(X, gt_comp2_probe_flat)
        pred_b2_flat = X @ W_b2
        b2_mse = jnp.mean((gt_comp2_probe_flat - pred_b2_flat) ** 2)

        W_w, _, _, _ = jnp.linalg.lstsq(X, gt_w1_probe_flat)
        pred_w_flat = X @ W_w
        w_mse = jnp.mean((gt_w1_probe_flat - pred_w_flat) ** 2)

        pred_b1_2d = pred_b1_flat.reshape(probe_beliefs.shape[0], -1, 3)
        pred_b2_2d = pred_b2_flat.reshape(probe_beliefs.shape[0], -1, 3)
        pred_w_2d = pred_w_flat.reshape(probe_beliefs.shape[0], -1, 1)

        return (pred_joint_2d, joint_mse,
                pred_b1_2d, pred_b2_2d, pred_w_2d,
                b1_mse, b2_mse, w_mse)


    @jax.jit
    def eval_per_position(model):
        logits = model.forward_batch(eval_symbols[:, :-1])
        targets = eval_symbols[:, 1:]
        per_token = optax.softmax_cross_entropy(
            logits.reshape(-1, sequence_generator.num_symbols),
            jax.nn.one_hot(targets.reshape(-1), sequence_generator.num_symbols),
        ).reshape(-1, sequence_length)
        return per_token.mean(axis=0)


    def process_probe_results(results):
        """Extract visualization and MSE data from probe_all outputs."""
        (pred_joint_2d, joint_mse,
         pred_b1_2d, pred_b2_2d, pred_w_2d,
         fb1_mse, fb2_mse, fw_mse) = results

        # joint probe: unclipped decomposition for viz
        pr_w_viz, pr_cb_viz = decompose_union_beliefs(
            pred_joint_2d, component_sizes,
        )
        pr_weights_flat = np.array(pr_w_viz[:, :, 0].reshape(-1))
        pr_comp_beliefs_flat = [
            np.array(b.reshape(-1, b.shape[-1])) for b in pr_cb_viz
        ]
        # joint probe: clipped decomposition for MSE
        pr_w_c, pr_cb_c = decompose_union_beliefs(
            jnp.clip(pred_joint_2d, 0), component_sizes,
        )
        pr_w_c_flat = np.array(pr_w_c[:, :, 0].reshape(-1))
        pr_cb_c_flat = [
            np.array(b.reshape(-1, b.shape[-1])) for b in pr_cb_c
        ]
        j_c1_mse = float(np.mean((gt_comp_beliefs_flat[0] - pr_cb_c_flat[0]) ** 2))
        j_c2_mse = float(np.mean((gt_comp_beliefs_flat[1] - pr_cb_c_flat[1]) ** 2))
        j_w_mse = float(np.mean((np.array(gt_weights_flat) - pr_w_c_flat) ** 2))

        # factored probe: direct outputs for viz
        fr_comp_beliefs_flat = [
            np.array(pred_b1_2d.reshape(-1, 3)),
            np.array(pred_b2_2d.reshape(-1, 3)),
        ]
        fr_weights_flat = np.array(pred_w_2d.reshape(-1))

        return (float(joint_mse),
                pr_comp_beliefs_flat, pr_weights_flat,
                j_c1_mse, j_c2_mse, j_w_mse,
                fr_comp_beliefs_flat, fr_weights_flat,
                float(fb1_mse), float(fb2_mse), float(fw_mse))


    print("starting training loop...")
    key_eval, key = jax.random.split(key)
    # initial eval
    init_seq = sequence_generator.sample_batch(
        key_eval, sequence_length=sequence_length+1, batch_size=batch_size,
    )
    init_logits = model.forward_batch(
        init_seq.symbols[:, :-1],
    ).reshape(-1, model.num_symbols)
    init_targets = jax.nn.one_hot(
        init_seq.symbols[:, 1:].reshape(-1),
        num_classes=model.num_symbols,
    )
    init_loss = float(optax.softmax_cross_entropy(
        logits=init_logits, labels=init_targets,
    ).mean())
    train_losses = [init_loss]

    # initial probe
    results = probe_all(model)
    (p_mse,
     pr_comp_beliefs_flat, pr_weights_flat,
     j_c1, j_c2, j_w,
     fr_comp_beliefs_flat, fr_weights_flat,
     f_c1, f_c2, f_w) = process_probe_results(results)

    probe_losses = [(0, p_mse)]
    subsystem_mses = [(0, j_c1, j_c2, j_w)]
    factored_mses = [(0, f_c1, f_c2, f_w)]

    # initial per-position loss
    init_pos_loss = np.array(eval_per_position(model))
    position_losses = [(0, init_pos_loss)]

    plot = visualise(
        0, train_losses, probe_losses, subsystem_mses, factored_mses,
        position_losses,
        bayes_optimal, bayes_per_position, marginal_entropy, vis_period,
        gt_comp_beliefs_flat, gt_weights_flat,
        pr_comp_beliefs_flat, pr_weights_flat,
        fr_comp_beliefs_flat, fr_weights_flat,
        true_component, sequence_length,
    )
    print(plot)

    for t in tqdm.trange(num_steps):
        key_sgd, key = jax.random.split(key)
        model, opt_state, loss = train_step(key_sgd, model, opt_state)

        if (t + 1) % vis_period == 0:
            train_losses.append(float(loss))

            results = probe_all(model)
            (p_mse,
             pr_comp_beliefs_flat, pr_weights_flat,
             j_c1, j_c2, j_w,
             fr_comp_beliefs_flat, fr_weights_flat,
             f_c1, f_c2, f_w) = process_probe_results(results)

            probe_losses.append((t + 1, p_mse))
            subsystem_mses.append((t + 1, j_c1, j_c2, j_w))
            factored_mses.append((t + 1, f_c1, f_c2, f_w))

            pos_loss = np.array(eval_per_position(model))
            position_losses.append((t + 1, pos_loss))

            # metrics logging
            if metrics_file:
                metric = {
                    "step": t + 1,
                    "train_loss": train_losses[-1],
                    "probe_mse": p_mse,
                    "j_comp1_mse": j_c1, "j_comp2_mse": j_c2, "j_w_mse": j_w,
                    "f_comp1_mse": f_c1, "f_comp2_mse": f_c2, "f_w_mse": f_w,
                    "pos_losses": pos_loss.tolist(),
                }
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(metric) + "\n")

            new_plot = visualise(
                t + 1, train_losses, probe_losses,
                subsystem_mses, factored_mses, position_losses,
                bayes_optimal, bayes_per_position, marginal_entropy, vis_period,
                gt_comp_beliefs_flat, gt_weights_flat,
                pr_comp_beliefs_flat, pr_weights_flat,
                fr_comp_beliefs_flat, fr_weights_flat,
                true_component, sequence_length,
            )
            tqdm.tqdm.write(f"{-plot}{new_plot}")
            plot = new_plot


    print("done!")
    end_time = datetime.datetime.now()
    print(f"  duration: {end_time - start_time}")
    print(f"  final loss: {train_losses[-1]:.4f} nats")
    print(f"  final probe MSE: {probe_losses[-1][1]:.6f}")
    print(f"  bayes-optimal: {bayes_optimal:.4f} nats")


if __name__ == "__main__":
    tyro.cli(main)
