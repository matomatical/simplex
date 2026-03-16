"""Plot PCA dimensionality over training — replicating Figure 4b from Shai et al."""

import json
import re
import sys

import numpy as np
import matthewplotlib as mp


# colors for different epsilon values (matching paper's palette roughly)
COLORS = ['green', 'cyan', 'blue', 'yellow', 'red', 'magenta']


def load_metrics(path):
    """Load JSONL metrics file, return (steps, dims_95, epsilon_label)."""
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    steps = np.array([d["step"] for d in data])
    dims = np.array([d["dims_95"] for d in data])
    # try to extract epsilon from filename
    m = re.search(r'eps([\d.]+)', path)
    label = f"ε={m.group(1)}" if m else path
    return steps, dims, label


def downsample(xs, ys, n=500):
    """Evenly downsample to ~n points."""
    if len(xs) <= n:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, n, dtype=int)
    return xs[idx], ys[idx]


def main(files, factored_dof=10):
    all_curves = []
    max_dim = 0
    max_step = 0

    for path in files:
        steps, dims, label = load_metrics(path)
        max_dim = max(max_dim, int(dims.max()))
        max_step = max(max_step, int(steps.max()))
        all_curves.append((steps, dims, label))

    ymax = max(max_dim + 5, factored_dof + 10)

    # --- Log-scale plot (main Figure 4b) ---
    log_layers = []
    # reference line at factored DoF
    ref_x = np.linspace(1, 6, 300)
    ref_y = np.full(300, factored_dof)
    log_layers.append((ref_x, ref_y, 'white'))

    for i, (steps, dims, label) in enumerate(all_curves):
        s, d = downsample(steps, dims)
        log_x = np.log10(np.maximum(s, 1))
        color = COLORS[i % len(COLORS)]
        log_layers.append((log_x, d.astype(float), color))

    log_plot = mp.axes(
        mp.scatter(*log_layers, height=15, width=40, xrange=(1, 6), yrange=(0, ymax)),
        title=" dims for 95% variance (log scale) ",
        ylabel="dims",
        xlabel="steps",
        xfmt="10^{x:.0f}",
        yfmt="{y:.0f}",
    )

    # --- Linear-scale plot (early training zoom) ---
    lin_layers = []
    # reference line
    ref_x_lin = np.linspace(0, max_step, 300)
    ref_y_lin = np.full(300, factored_dof)
    lin_layers.append((ref_x_lin, ref_y_lin, 'white'))

    for i, (steps, dims, label) in enumerate(all_curves):
        mask = steps <= max_step
        s, d = downsample(steps[mask], dims[mask])
        color = COLORS[i % len(COLORS)]
        lin_layers.append((s.astype(float), d.astype(float), color))

    lin_plot = mp.axes(
        mp.scatter(*lin_layers, height=15, width=40, xrange=(0, max_step), yrange=(0, ymax)),
        title=" dims for 95% variance (linear) ",
        ylabel="dims",
        xlabel="steps",
        xfmt="{x:.0f}",
        yfmt="{y:.0f}",
    )

    # --- Layout ---
    combined = mp.hstack(log_plot, lin_plot)
    print(combined)

    # legend
    legend_parts = []
    for i, (_, _, label) in enumerate(all_curves):
        color = COLORS[i % len(COLORS)]
        legend_parts.append(f"  {color}={label}")
    print("  " + "  ".join(legend_parts) + f"  white=factored({factored_dof})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} metrics1.jsonl [metrics2.jsonl ...]")
        print(f"  Optional: --dof N  (factored DoF, default 10)")
        sys.exit(1)

    # parse --dof flag
    args = sys.argv[1:]
    dof = 10
    if "--dof" in args:
        idx = args.index("--dof")
        dof = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    main(args, factored_dof=dof)
