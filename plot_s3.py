"""Generate publication-quality plots for simplex3 seed sweep results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

import jax
import jax.numpy as jnp
from generators import mess3, decompose_union_beliefs

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

LOGDIR = Path("logs")
OUTDIR = Path("report")
OUTDIR.mkdir(exist_ok=True)

# Baselines (from training logs)
BAYES_OPTIMAL = 0.9448
MAX_ENTROPY = 1.0986
MARGINAL_ENTROPY = 1.0986

# Default config (must match the runs)
ALPHA1, X1 = 0.6, 0.15
ALPHA2, X2 = 0.85, 0.05
SEQ_LEN = 32
N_BASELINE_SEQS = 4096


def compute_per_position_baseline():
    """Compute per-position Bayes-optimal loss via Monte Carlo."""
    gen = mess3(ALPHA1, X1) + mess3(ALPHA2, X2)
    key = jax.random.key(0)
    batch = gen.sample_batch(key, sequence_length=SEQ_LEN + 1,
                             batch_size=N_BASELINE_SEQS)
    symbols = batch.symbols  # (N, seq_len+1)
    T = np.array(gen.transition_distributions)
    emission_probs = T.sum(axis=2)  # (num_symbols, num_states)
    beliefs_all = np.array(jax.vmap(gen.belief_states)(symbols))
    pred_beliefs = beliefs_all[:, 1:-1, :]  # (N, seq_len, num_states)
    pred_dists = pred_beliefs @ emission_probs.T
    targets = np.array(symbols[:, 1:])
    per_token = -np.log(
        pred_dists[
            np.arange(N_BASELINE_SEQS)[:, None],
            np.arange(SEQ_LEN)[None, :],
            targets,
        ] + 1e-10
    )
    return per_token.mean(axis=0)  # (seq_len,)


def load_runs(pattern="s3_seed*_20260317_133236.jsonl"):
    """Load all seed runs, return list of dicts-of-arrays."""
    runs = []
    for path in sorted(LOGDIR.glob(pattern)):
        records = [json.loads(line) for line in open(path)]
        run = {}
        for key in records[0]:
            if key == "pos_losses":
                run[key] = np.array([r[key] for r in records])
            else:
                run[key] = np.array([r[key] for r in records])
        runs.append(run)
    print(f"Loaded {len(runs)} runs, {len(runs[0]['step'])} steps each")
    return runs


def mean_and_se(runs, key):
    """Compute mean and standard error across seeds."""
    vals = np.stack([r[key] for r in runs])
    mean = vals.mean(axis=0)
    se = vals.std(axis=0) / np.sqrt(len(runs))
    return mean, se


def smooth(x, window=64):
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def plot_with_band(ax, steps, mean, se, color, label, smooth_window=64):
    """Plot smoothed mean with SE band."""
    s = smooth(mean, smooth_window)
    s_se = smooth(se, smooth_window)
    s_steps = smooth(steps, smooth_window)
    ax.plot(s_steps, s, color=color, label=label)
    ax.fill_between(s_steps, s - s_se, s + s_se, color=color, alpha=0.2)


def simplex_project(beliefs):
    """Project 3-state beliefs to 2D simplex coordinates."""
    xs = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    ys = (np.sqrt(3) / 2) * beliefs[:, 2]
    return xs, ys


def fig_ground_truth_simplices():
    """Ground-truth belief simplex scatter plots and w(t) trajectories."""
    gen1 = mess3(ALPHA1, X1)
    gen2 = mess3(ALPHA2, X2)
    gen = gen1 + gen2
    component_sizes = [gen1.num_states, gen2.num_states]

    key = jax.random.key(42)
    batch = gen.sample_batch(key, sequence_length=SEQ_LEN, batch_size=256)
    symbols = batch.symbols
    beliefs_all = jax.vmap(gen.belief_states)(symbols)
    beliefs = beliefs_all[:, 1:, :]  # drop prior, align with positions
    weights, comp_beliefs = decompose_union_beliefs(
        beliefs, component_sizes,
    )
    weights = np.array(weights)
    comp_beliefs = [np.array(b) for b in comp_beliefs]

    # True component from initial state
    true_comp = np.array(batch.states[:, 0] >= gen1.num_states, dtype=int)

    # Flatten for scatter
    n_seqs, seq_len = symbols.shape[0], SEQ_LEN
    w1_flat = weights[:, :, 0].reshape(-1)
    comp1_flat = comp_beliefs[0].reshape(-1, 3)
    comp2_flat = comp_beliefs[1].reshape(-1, 3)
    comp_flat = np.repeat(true_comp, seq_len)
    positions = np.tile(np.arange(1, seq_len + 1), n_seqs)

    # --- Figure 1: Two simplices ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    xs, ys = simplex_project(comp1_flat)
    ax.scatter(xs, ys, c=comp1_flat, s=1, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title(r'Component 1 beliefs ($\theta$)')
    ax.set_xlabel('simplex x')
    ax.set_ylabel('simplex y')

    ax = axes[1]
    xs, ys = simplex_project(comp2_flat)
    ax.scatter(xs, ys, c=comp2_flat, s=1, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title(r'Component 2 beliefs ($\lambda$)')
    ax.set_xlabel('simplex x')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'ground_truth_simplices.png')
    plt.close(fig)
    print("Saved ground_truth_simplices.png")

    # --- Figure 2: Mixture weight trajectories ---
    fig, ax = plt.subplots(figsize=(7, 4))
    w1_per_seq = weights[:, :, 0]  # (n_seqs, seq_len)
    positions_line = np.arange(1, seq_len + 1)
    for i in range(n_seqs):
        color = 'C3' if true_comp[i] == 0 else 'C0'
        ax.plot(positions_line, w1_per_seq[i], color=color, lw=0.5, alpha=0.2)
    # Legend entries
    ax.plot([], [], color='C3', lw=1.5, label='True comp 1')
    ax.plot([], [], color='C0', lw=1.5, label='True comp 2')
    ax.set_xlabel('Sequence position')
    ax.set_ylabel(r'$\alpha$ (comp 1 weight)')
    # ax.set_title('Mixture weight over sequence')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(OUTDIR / 'ground_truth_weights.png')
    plt.close(fig)
    print("Saved ground_truth_weights.png")


def fig_training_loss(runs):
    """Figure 1: Training loss, raw traces + smoothed overlay, log x-scale."""
    colors = [f'C{i}' for i in range(len(runs))]
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, run in enumerate(runs):
        # Raw traces in background
        ax.plot(run['step'], run['train_loss'], lw=0.3, alpha=0.15,
                color=colors[idx])
        # Smoothed overlay
        s_steps = smooth(run['step'], 256)
        s_loss = smooth(run['train_loss'], 256)
        ax.plot(s_steps, s_loss, lw=1.5, alpha=0.9, color=colors[idx],
                label=f'Seed {idx}')
    ax.axhline(BAYES_OPTIMAL, color='red', ls='--', lw=1, label=f'Bayes-optimal ({BAYES_OPTIMAL:.3f})')
    ax.axhline(MAX_ENTROPY, color='gray', ls=':', lw=1, label=f'Max entropy ({MAX_ENTROPY:.3f})')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss (nats)')
    # ax.set_title('Training loss')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylim(0.9, 1.15)
    fig.savefig(OUTDIR / 'loss.png')
    plt.close(fig)
    print("Saved loss.png")


def _fig_factored_vs_joint(runs, yscale, suffix, xscale='linear'):
    """Factored vs joint-decomposed MSE for each component."""
    steps = runs[0]['step']

    panels = [
        ('j_comp1_mse', 'f_comp1_mse', r'Component 1 ($\theta$)'),
        ('j_comp2_mse', 'f_comp2_mse', r'Component 2 ($\lambda$)'),
        ('j_w_mse', 'f_w_mse', r'Mixture weight ($\alpha$)'),
    ]

    # Compute uniform y-range across all panels
    all_means = []
    for jkey, fkey, _ in panels:
        jm, _ = mean_and_se(runs, jkey)
        fm, _ = mean_and_se(runs, fkey)
        all_means.extend([jm, fm])

    # For log scale, find range of smoothed values
    smoothed_all = [smooth(m, 64) for m in all_means]
    ymin = min(s.min() for s in smoothed_all) * 0.5
    ymax = max(s.max() for s in smoothed_all) * 2.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (jkey, fkey, label) in enumerate(panels):
        ax = axes[i]
        jm, jse = mean_and_se(runs, jkey)
        fm, fse = mean_and_se(runs, fkey)
        plot_with_band(ax, steps, jm, jse, 'C3', 'Joint-decomposed')
        plot_with_band(ax, steps, fm, fse, 'C0', 'Modular probe')
        ax.set_xlabel('Training step')
        if i == 0:
            ax.set_ylabel('MSE')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if yscale == 'log':
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, ymax / 2.0)
        ax.legend(fontsize=9)
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha='center', fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTDIR / f'factored_vs_joint_{suffix}.png')
    plt.close(fig)
    print(f"Saved factored_vs_joint_{suffix}.png")


def fig_factored_vs_joint(runs):
    _fig_factored_vs_joint(runs, 'log', 'log')
    _fig_factored_vs_joint(runs, 'linear', 'linear')
    _fig_factored_vs_joint(runs, 'log', 'loglog', xscale='log')


def fig_per_position_loss_over_training(runs, bayes_per_pos):
    """Per-token loss curves over training, coloured by position (viridis)."""
    steps = runs[0]['step']
    seq_len = runs[0]['pos_losses'].shape[1]
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, seq_len - 1)
    smooth_window = 256

    fig, ax = plt.subplots(figsize=(8, 5))
    for pos in range(seq_len):
        vals = np.stack([r['pos_losses'][:, pos] for r in runs])
        mean = vals.mean(axis=0)
        se = vals.std(axis=0) / np.sqrt(len(runs))
        s_mean = smooth(mean, smooth_window)
        s_se = smooth(se, smooth_window)
        s_steps = smooth(steps, smooth_window)
        color = cmap(norm(pos))
        ax.plot(s_steps, s_mean, color=color, lw=0.8, alpha=0.8)
        ax.fill_between(s_steps, s_mean - s_se, s_mean + s_se,
                         color=color, alpha=0.1)

    # Bayes-optimal asymptotes as dashed lines on right edge
    for pos in range(seq_len):
        ax.plot([steps[-1] * 0.98, steps[-1]], [bayes_per_pos[pos]] * 2,
                color=cmap(norm(pos)), lw=2, ls='--', alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, label='Sequence position')
    cbar.set_ticks([0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1])

    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss (nats)')
    ax.set_ylim(0.9, 1.15)
    fig.savefig(OUTDIR / 'per_token_training.png')
    plt.close(fig)
    print("Saved per_token_training.png")


def fig_per_position_loss(runs, bayes_per_pos):
    """Figure 4: Per-position loss at end of training."""
    # Average pos_losses over last 80% of training across seeds
    n_entries = runs[0]['pos_losses'].shape[0]
    start = n_entries // 5  # skip first 20%
    pos_losses = np.stack([r['pos_losses'][start:].mean(axis=0) for r in runs])
    mean = pos_losses.mean(axis=0)
    se = pos_losses.std(axis=0) / np.sqrt(len(runs))
    positions = np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(positions, mean, yerr=se, fmt='o-', color='C0', markersize=4,
                label='Model loss')
    ax.plot(positions, bayes_per_pos, 's--', color='red', markersize=4,
            label='Bayes-optimal')
    ax.set_xlabel('Sequence position')
    ax.set_ylabel('Loss (nats)')
    ax.legend()
    fig.savefig(OUTDIR / 'per_position_loss.png')
    plt.close(fig)
    print("Saved per_position_loss.png")


def fig_per_token_training_by_seed(runs, bayes_per_pos):
    """Per-token loss over training, one subplot per seed."""
    n_seeds = len(runs)
    seq_len = runs[0]['pos_losses'].shape[1]
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, seq_len - 1)
    smooth_window = 256

    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 4), sharey=True)
    for idx, (run, ax) in enumerate(zip(runs, axes)):
        steps = run['step']
        for pos in range(seq_len):
            vals = run['pos_losses'][:, pos]
            s_vals = smooth(vals, smooth_window)
            s_steps = smooth(steps, smooth_window)
            color = cmap(norm(pos))
            ax.plot(s_steps, s_vals, color=color, lw=0.8, alpha=0.8)
        # Bayes-optimal ticks on right
        for pos in range(seq_len):
            ax.plot([steps[-1] * 0.98, steps[-1]], [bayes_per_pos[pos]] * 2,
                    color=cmap(norm(pos)), lw=2, ls='--', alpha=0.6)
        ax.set_xlabel('Training step')
        ax.set_title(f'Seed {idx}')
        ax.set_ylim(0.9, 1.15)
    axes[0].set_ylabel('Loss (nats)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes.tolist(), label='Sequence position', shrink=0.8)
    fig.savefig(OUTDIR / 'per_token_training_by_seed.png')
    plt.close(fig)
    print("Saved per_token_training_by_seed.png")


def fig_per_position_loss_by_seed(runs, bayes_per_pos):
    """Per-position final loss, one subplot per seed."""
    n_seeds = len(runs)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 4), sharey=True)
    for idx, (run, ax) in enumerate(zip(runs, axes)):
        n_entries = run['pos_losses'].shape[0]
        pos_losses = run['pos_losses'][n_entries // 5:].mean(axis=0)
        positions = np.arange(1, len(pos_losses) + 1)
        ax.plot(positions, pos_losses, 'o-', color='C0', markersize=3,
                label='Model loss')
        ax.plot(positions, bayes_per_pos, 's--', color='red', markersize=3,
                label='Bayes-optimal')
        ax.set_xlabel('Sequence position')
        ax.set_title(f'Seed {idx}')
        if idx == 0:
            ax.legend(fontsize=9)
    axes[0].set_ylabel('Loss (nats)')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'per_position_loss_by_seed.png')
    plt.close(fig)
    print("Saved per_position_loss_by_seed.png")


def fig_long_run_probe_mse():
    """Log-log plot of per-component modular probe MSEs from long runs."""
    long_runs = load_runs("s3_long_seed*_20260319_075710.jsonl")
    steps = long_runs[0]['step']
    colors = [f'C{i}' for i in range(len(long_runs))]

    mse_panels = [
        ('f_comp1_mse', r'Component 1 ($\theta$)'),
        ('f_comp2_mse', r'Component 2 ($\lambda$)'),
        ('f_w_mse', r'Mixture weight ($\alpha$)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Training loss (log-log, same style as fig_training_loss)
    ax = axes[0]
    for idx, run in enumerate(long_runs):
        ax.plot(run['step'], run['train_loss'], lw=0.3, alpha=0.15,
                color=colors[idx])
        s_steps = smooth(run['step'], 256)
        s_loss = smooth(run['train_loss'], 256)
        ax.plot(s_steps, s_loss, lw=1.5, alpha=0.9, color=colors[idx],
                label=f'Seed {idx}')
    ax.axhline(BAYES_OPTIMAL, color='red', ls='--', lw=1,
               label=f'Bayes-optimal ({BAYES_OPTIMAL:.3f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss (nats)')
    ax.legend(fontsize=8)
    ax.text(0.5, 1.02, 'Training loss', transform=ax.transAxes,
            ha='center', fontsize=12)

    # Probe MSE panels
    for i, (key, label) in enumerate(mse_panels):
        ax = axes[i + 1]
        for idx, run in enumerate(long_runs):
            s_steps = smooth(run['step'], 256)
            s_vals = smooth(run[key], 256)
            ax.plot(s_steps, s_vals, lw=1.0, alpha=0.7, color=colors[idx],
                    label=f'Seed {idx}')
        mean, se = mean_and_se(long_runs, key)
        s_mean = smooth(mean, 256)
        s_se = smooth(se, 256)
        s_steps = smooth(steps, 256)
        ax.plot(s_steps, s_mean, lw=2, color='black', label='Mean')
        ax.fill_between(s_steps, s_mean - s_se, s_mean + s_se,
                         color='black', alpha=0.15)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Training step')
        if i == 0:
            ax.set_ylabel('MSE')
        ax.legend(fontsize=8)
        ax.text(0.5, 1.02, label, transform=ax.transAxes,
                ha='center', fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTDIR / 'long_run_probe_mse.png')
    plt.close(fig)
    print(f"Saved long_run_probe_mse.png ({len(steps)} entries, up to step {steps[-1]})")


def fig_factored_vs_joint_final(runs):
    """Bar chart comparing final factored vs joint MSE."""
    keys_j = ['j_comp1_mse', 'j_comp2_mse', 'j_w_mse']
    keys_f = ['f_comp1_mse', 'f_comp2_mse', 'f_w_mse']
    labels = [r'$\theta$ (comp 1)', r'$\lambda$ (comp 2)', r'$\alpha$ (weight)']

    # Average over last 100 entries
    j_means, j_ses, f_means, f_ses = [], [], [], []
    for jk, fk in zip(keys_j, keys_f):
        jvals = np.array([r[jk][-100:].mean() for r in runs])
        fvals = np.array([r[fk][-100:].mean() for r in runs])
        j_means.append(jvals.mean())
        j_ses.append(jvals.std() / np.sqrt(len(runs)))
        f_means.append(fvals.mean())
        f_ses.append(fvals.std() / np.sqrt(len(runs)))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, j_means, width, yerr=j_ses, label='Joint-decomposed',
           color='C3', alpha=0.8, capsize=4)
    ax.bar(x + width/2, f_means, width, yerr=f_ses, label='Modular probe',
           color='C0', alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(OUTDIR / 'factored_vs_joint_bar.png')
    plt.close(fig)
    print("Saved factored_vs_joint_bar.png")

    # Linear scale version, half height
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(x - width/2, j_means, width, yerr=j_ses, label='Joint-decomposed',
           color='C3', alpha=0.8, capsize=4)
    ax.bar(x + width/2, f_means, width, yerr=f_ses, label='Modular probe',
           color='C0', alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MSE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / 'factored_vs_joint_bar_linear.png')
    plt.close(fig)
    print("Saved factored_vs_joint_bar_linear.png")


if __name__ == '__main__':
    runs = load_runs()
    print("Computing per-position Bayes-optimal baseline...")
    bayes_per_pos = compute_per_position_baseline()
    fig_ground_truth_simplices()
    fig_training_loss(runs)
    fig_factored_vs_joint(runs)
    fig_per_position_loss_over_training(runs, bayes_per_pos)
    fig_per_token_training_by_seed(runs, bayes_per_pos)
    fig_per_position_loss(runs, bayes_per_pos)
    fig_per_position_loss_by_seed(runs, bayes_per_pos)
    fig_factored_vs_joint_final(runs)
    fig_long_run_probe_mse()
    print("Done!")
