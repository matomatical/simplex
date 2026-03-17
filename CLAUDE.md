# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Simplex Replication — a JAX-based research project replicating ML papers on how transformers learn representations, starting with Shai et al. (2024) "Transformers represent belief state geometry" and Shai et al. (2026) "Transformers learn factored representations."

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# simplex1: single HMM belief geometry (original paper)
python simplex1.py --generator mess3 --num_steps 100000 --learning_rate 0.01

# simplex2: two-factor experiments with factored vs joint probes
python simplex2.py --generator tetra --num_steps 100000

# simplex2b: multi-factor experiments replicating Figure 4b (factored reps paper)
# Uses pmap for multi-device training. Default: 5 Mess3 factors, paper's architecture.
python simplex2b.py --epsilon 0.5 --batch-size-per-device 6250 --metrics-file metrics.jsonl

# Plot metrics from unattended runs (no TPU needed)
python plot_metrics.py metrics_eps0.0.jsonl metrics_eps0.5.jsonl

# Epsilon sweep analysis
python plot_sweep.py sweep_results_r2.jsonl
```

No test suite, linter, or CI exists — this is early-stage research code.

## Experiments

### simplex1 — Single HMM belief geometry
Trains a transformer on sequences from a single HMM (Mess3/ZOR/ALT) and probes whether activations encode the belief simplex geometry. Live terminal visualization with matthewplotlib.

### simplex2 — Two-factor factored representations
Trains on product of two HMMs. Compares factored vs joint linear probes using R² and MSE. Includes simplex triangle visualizations per factor.

### simplex2b — Multi-factor replication of Shai et al. (2026) Figure 4b
Trains on product of N Mess3 factors with optional noisy channel (epsilon). Tracks PCA dimensionality (dims for 95% variance) over training. Uses `jax.pmap` for data-parallel training across multiple TPU devices. Key hyperparameters match the paper: 4 layers, 3 heads, d_model=120, d_ff=480, d_head=40, batch=25000, lr=5e-4, seq_len=8. See `simplex-papers/hyperparameters.md` for full comparison.

## Architecture

Python modules, all built on JAX with `jaxtyping` annotations:

- **`simplex1.py`** — Single-HMM training pipeline with live visualization.
- **`simplex2.py`** — Two-factor experiment with factored/joint probes and R².
- **`simplex2b.py`** — N-factor experiment with pmap, PCA dimensionality tracking, metrics logging.
- **`transformer.py`** — Decode-only transformer. Pre-LayerNorm, causal self-attention, residual connections, separate QKV projections per head. `SequenceTransformer` wraps one-hot encoding. `forward_batch_with_activations` returns per-layer residual stream.
- **`generators.py`** — HMM sequence generators. `SequenceGenerator` with `sample()`, `sample_batch()`, `belief_states()`. Supports product (`*`) of generators and `noisy_channel(gen, epsilon)`.
- **`strux.py`** — `@struct` decorator for immutable JAX PyTree dataclasses.
- **`baselines.py`** — Computes Bayes-optimal loss for generators.
- **`plot_metrics.py`** — Plots dims_95 over training from JSONL metrics files (log + linear scale).
- **`plot_sweep.py`** — Plots R²/MSE comparison across epsilon values from sweep results.
- **`logs/`** — Experiment logs, metrics JSONL files, and sweep results.
- **`simplex-papers/`** — Reference papers, figures, hyperparameter comparison, and paper's code repo.

## Key Patterns

- All model components use `@strux.struct` for immutability and JAX pytree compatibility.
- Heavy use of `jax.vmap()` for vectorization and `@jax.jit` with `static_argnames` for compilation.
- `jax.pmap` with `jax.lax.pmean` for multi-device data parallelism (simplex2b).
- PRNG keys are explicitly split and threaded for reproducibility.
- `matthewplotlib` (custom plotting library) is a dependency installed from GitHub.
- Metrics logged as JSONL for unattended runs (`--metrics-file` flag).
