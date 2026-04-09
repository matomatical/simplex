"""Visualise probed belief simplex at model checkpoints."""

import sys
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import strux
from transformer import SequenceTransformer
from generators import MESS3

matplotlib.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = Path("simplex1-report")
OUTDIR.mkdir(exist_ok=True)

# Default model config (must match the runs)
MODEL_CONFIG = dict(
    num_symbols=3,
    sequence_length=10,
    num_blocks=4,
    embed_size=64,
    num_heads=1,
    head_size=8,
    mlp_size=256,
)
PROBE_LAYER = -1
PROBE_NUM_SEQS = 512


def make_template_and_probe_data(seed=42):
    """Create template model and ground-truth probe data."""
    key = jax.random.key(seed)
    key_model, key_probe = jax.random.split(key)

    template = SequenceTransformer.init(key=key_model, **MODEL_CONFIG)

    probe_batch = MESS3.sample_batch(
        key_probe,
        sequence_length=MODEL_CONFIG['sequence_length'],
        batch_size=PROBE_NUM_SEQS,
    )
    probe_symbols = probe_batch.symbols
    probe_beliefs_all = jax.vmap(MESS3.belief_states)(probe_symbols)
    # drop prior, align with activations
    probe_beliefs_flat = probe_beliefs_all[:, 1:, :].reshape(-1, MESS3.num_states)

    return template, probe_symbols, probe_beliefs_flat


def probe_model(model, probe_symbols, probe_beliefs_flat):
    """Run least-squares probe and return predicted beliefs + MSE."""
    _, activations = model.forward_batch_with_activations(probe_symbols)
    acts = activations[:, PROBE_LAYER, :, :]
    acts_flat = acts.reshape(-1, acts.shape[-1])
    ones = jnp.ones((acts_flat.shape[0], 1))
    X = jnp.concatenate([acts_flat, ones], axis=1)
    W, _, _, _ = jnp.linalg.lstsq(X, probe_beliefs_flat)
    predicted = X @ W
    mse = float(jnp.mean((probe_beliefs_flat - predicted) ** 2))
    return np.array(predicted), mse


def simplex_scatter(ax, beliefs, gt_beliefs, title):
    """Plot 2D simplex projection coloured by ground-truth belief."""
    xs = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    ys = (np.sqrt(3) / 2) * beliefs[:, 2]
    cs = np.clip(gt_beliefs, 0, 1)
    ax.scatter(xs, ys, c=cs, s=0.5, alpha=0.3)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.0)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    paths = sorted(checkpoint_dir.glob("model_step*.npz"))
    if not paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # sort by step number
    def step_of(p):
        m = re.search(r'step(\d+)', p.stem)
        return int(m.group(1)) if m else 0
    paths = sorted(paths, key=step_of)
    steps = [step_of(p) for p in paths]

    print(f"Found {len(paths)} checkpoints: steps {steps}")
    template, probe_symbols, probe_beliefs_flat = make_template_and_probe_data()

    # ground truth panel
    n_panels = len(paths) + 1
    ncols = min(n_panels, 5)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    # ground truth
    gt_np = np.array(probe_beliefs_flat)
    simplex_scatter(axes[0], gt_np, gt_np, "Ground truth")

    # checkpoint panels
    for i, (path, step) in enumerate(zip(paths, steps)):
        model = strux.load(path, template=template)
        predicted, mse = probe_model(model, probe_symbols, probe_beliefs_flat)
        label = f"Step {step:,}\nMSE={mse:.2e}"
        simplex_scatter(axes[i + 1], predicted, gt_np, label)
        print(f"  step {step:>12,}: MSE = {mse:.6f}")

    # hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Probed belief simplex over training", fontsize=14, y=1.02)
    fig.tight_layout()
    out = OUTDIR / 'simplex_checkpoints.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_s1_simplex.py <checkpoint_dir>")
        sys.exit(1)
    main(sys.argv[1])
