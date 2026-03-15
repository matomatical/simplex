# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Simplex Replication — a JAX-based research project replicating ML papers on how transformers learn representations, starting with Shai et al. (2024) "Transformers represent belief state geometry."

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Train (main entry point, uses tyro for CLI args)
python simplex1.py --generator mess3 --num_steps 100000 --learning_rate 0.01

# Key CLI flags: --generator (mess3/zor/alt), --num_blocks, --embed_size,
#   --num_heads, --head_size, --mlp_size, --opt (sgd/adam), --batch_size, --seed
```

No test suite, linter, or CI exists — this is early-stage research code.

## Architecture

Four Python modules, all built on JAX with `jaxtyping` annotations:

- **`simplex1.py`** — Training pipeline. Configures hyperparameters via `tyro.cli()`, runs the training loop with JIT-compiled steps, computes softmax cross-entropy loss for next-token prediction.
- **`transformer.py`** — Decode-only transformer. Pre-LayerNorm architecture with causal self-attention, residual connections, separate QKV projections per head. `SequenceTransformer` is the top-level wrapper handling one-hot encoding.
- **`generators.py`** — HMM sequence generators (`MESS3`, `ZOR`, `ALT`). `SequenceGenerator` is the core engine with `sample()` and `sample_batch()` methods.
- **`strux.py`** — `@struct` decorator that makes classes into immutable dataclasses + JAX PyTree nodes. Also provides `split_treelike()` for PRNG key splitting and `size()` for parameter counting.

## Key Patterns

- All model components use `@strux.struct` for immutability and JAX pytree compatibility.
- Heavy use of `jax.vmap()` for vectorization and `@jax.jit` with `static_argnames` for compilation.
- PRNG keys are explicitly split and threaded for reproducibility.
- `matthewplotlib` (custom plotting library) is a dependency installed from GitHub.
