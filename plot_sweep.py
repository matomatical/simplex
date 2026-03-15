"""Plot factored vs joint probe MSE by epsilon from sweep results (mean over seeds)."""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
import matthewplotlib as mp


def main(results_file="sweep_results.jsonl"):
    with open(results_file) as f:
        results = [json.loads(line) for line in f if line.strip()]

    # group by epsilon
    by_eps = defaultdict(list)
    for r in results:
        by_eps[r["epsilon"]].append(r)

    epsilons = sorted(by_eps.keys())

    # compute means
    mean_factored = []
    mean_joint = []
    for eps in epsilons:
        runs = by_eps[eps]
        mean_factored.append(np.mean([r["factored_mse"] for r in runs]))
        mean_joint.append(np.mean([r["joint_mse"] for r in runs]))

    # interleave: factored, joint, spacer for each epsilon
    values = []
    colors = []
    labels = []
    for i, eps in enumerate(epsilons):
        values.append(mean_factored[i])
        colors.append("magenta")
        values.append(mean_joint[i])
        colors.append("yellow")
        values.append(0)
        colors.append("black")
        labels.append(str(eps))

    # remove trailing spacer
    values.pop()
    colors.pop()

    chart = mp.columns(
        np.array(values),
        height=12,
        column_width=2,
        column_spacing=0,
        colors=colors,
    )
    chart = mp.border(chart, title=" factored vs joint probe MSE by epsilon (mean) ")
    print(chart)

    # labels: each group is 6 cols wide (2+2+2 spacer), last is 4
    label_parts = []
    for i, label in enumerate(labels):
        w = 6 if i < len(labels) - 1 else 4
        label_parts.append(label.center(w))
    print("  " + "".join(label_parts))
    print("  magenta=factored  yellow=joint")
    print()

    # table with std error and significance
    print(f"  {'epsilon':>10s}  {'factored':>10s}  {'± se':>8s}  "
          f"{'joint':>10s}  {'± se':>8s}  {'p-value':>10s}  {'sig?':>5s}  "
          f"{'n':>3s}")
    for eps in epsilons:
        runs = by_eps[eps]
        f_vals = np.array([r["factored_mse"] for r in runs])
        j_vals = np.array([r["joint_mse"] for r in runs])
        n = len(runs)
        f_mean = f_vals.mean()
        j_mean = j_vals.mean()
        f_se = f_vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        j_se = j_vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        # paired t-test: is factored different from joint?
        if n > 1:
            t_stat, p_val = stats.ttest_rel(f_vals, j_vals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        else:
            p_val = float('nan')
            sig = ""
        print(f"  {eps:10.4f}  {f_mean:10.6f}  {f_se:8.6f}  "
              f"{j_mean:10.6f}  {j_se:8.6f}  {p_val:10.4f}  {sig:>5s}  {n:3d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sweep_results.jsonl")
