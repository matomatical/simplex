"""Generate log-log probe MSE plots for simplex1 long runs."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

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
OUTDIR = Path("simplex1-report")
OUTDIR.mkdir(exist_ok=True)

# Bayes-optimal loss for default Mess3 (alpha=0.85, x=0.05)
# Computed from baselines.py
BAYES_OPTIMAL = 0.7941


def load_runs(pattern):
    """Load all seed runs matching pattern, return list of dicts-of-arrays."""
    runs = []
    for path in sorted(LOGDIR.glob(pattern)):
        records = [json.loads(line) for line in open(path)]
        run = {
            key: np.array([r[key] for r in records])
            for key in records[0]
        }
        runs.append(run)
    # truncate to shortest run (in case runs are still in progress)
    min_len = min(len(r['step']) for r in runs)
    for run in runs:
        for key in run:
            run[key] = run[key][:min_len]
    print(f"Loaded {len(runs)} runs from {pattern}, "
          f"{min_len} entries each")
    return runs


def smooth(x, window=256):
    """Simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def mean_and_se(runs, key):
    """Compute mean and standard error across seeds."""
    vals = np.stack([r[key] for r in runs])
    return vals.mean(axis=0), vals.std(axis=0) / np.sqrt(len(runs))


def fig_long_run(runs):
    """Log-log training loss and probe MSE over long run."""
    steps = runs[0]['step']
    colors = [f'C{i}' for i in range(len(runs))]
    smooth_window = 256

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Training loss
    ax = axes[0]
    for idx, run in enumerate(runs):
        ax.plot(run['step'], run['train_loss'], lw=0.3, alpha=0.15,
                color=colors[idx])
        s_steps = smooth(run['step'], smooth_window)
        s_loss = smooth(run['train_loss'], smooth_window)
        ax.plot(s_steps, s_loss, lw=1.5, alpha=0.9, color=colors[idx],
                label=f'Seed {idx}')
    ax.axhline(BAYES_OPTIMAL, color='red', ls='--', lw=1,
               label=f'Bayes-optimal ({BAYES_OPTIMAL:.4f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss (nats)')
    ax.legend(fontsize=8)
    ax.set_title('Training loss')

    # Panel 2: Probe MSE
    ax = axes[1]
    for idx, run in enumerate(runs):
        s_steps = smooth(run['step'], smooth_window)
        s_vals = smooth(run['probe_mse'], smooth_window)
        ax.plot(s_steps, s_vals, lw=1.0, alpha=0.7, color=colors[idx],
                label=f'Seed {idx}')
    mean, se = mean_and_se(runs, 'probe_mse')
    s_mean = smooth(mean, smooth_window)
    s_se = smooth(se, smooth_window)
    s_steps = smooth(steps, smooth_window)
    ax.plot(s_steps, s_mean, lw=2, color='black', label='Mean')
    ax.fill_between(s_steps, s_mean - s_se, s_mean + s_se,
                     color='black', alpha=0.15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=8)
    ax.set_title('Probe MSE (belief state)')

    fig.tight_layout()
    fig.savefig(OUTDIR / 'long_run_probe_mse.png')
    plt.close(fig)
    print(f"Saved long_run_probe_mse.png ({len(steps)} entries, "
          f"up to step {steps[-1]})")


if __name__ == '__main__':
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "s1_long_seed*_20260409_071633.jsonl"
    runs = load_runs(pattern)
    fig_long_run(runs)
    print("Done!")
