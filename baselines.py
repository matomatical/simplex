"""Compute theoretical loss baselines for HMM sequence generators."""

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from typing import Literal
from generators import MESS3, ZOR, ALT, SequenceGenerator, mess3


GENERATORS = {
    "mess3": MESS3,
    "zor": ZOR,
    "alt": ALT,
    "union": mess3(0.6, 0.15) + mess3(0.85, 0.05),
}


def compute_baselines(
    gen: SequenceGenerator,
    num_sequences: int = 1000,
    sequence_length: int = 1000,
    seed: int = 0,
):
    T = np.array(gen.transition_distributions)

    # stationary state distribution (left eigenvector of transition matrix)
    T_marginal = T.sum(axis=0)  # sum over symbols -> states x states
    eigenvalues, eigenvectors = np.linalg.eig(T_marginal.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()

    # marginal symbol distribution
    emission_probs = T.sum(axis=2)  # P(symbol | state)
    symbol_probs = stationary @ emission_probs.T
    marginal_entropy = -np.sum(symbol_probs * np.log(symbol_probs + 1e-10))
    max_entropy = np.log(gen.num_symbols)

    # bayes-optimal loss (empirical, via belief states)
    key = jax.random.key(seed)
    batch = gen.sample_batch(key, sequence_length, num_sequences)
    all_beliefs = jax.vmap(gen.belief_states)(batch.symbols)
    beliefs = np.array(all_beliefs[:, :-1, :]).reshape(-1, gen.num_states)
    symbols = np.array(batch.symbols.reshape(-1))

    pred_dists = beliefs @ emission_probs.T
    actual_probs = pred_dists[np.arange(len(symbols)), symbols]
    bayes_optimal_loss = -np.log(actual_probs).mean()
    bayes_optimal_entropy = -np.sum(
        pred_dists * np.log(pred_dists + 1e-10), axis=1,
    ).mean()

    return {
        "max_entropy": max_entropy,
        "marginal_entropy": marginal_entropy,
        "bayes_optimal_loss": bayes_optimal_loss,
        "bayes_optimal_entropy": bayes_optimal_entropy,
    }


def main(
    generator: Literal["mess3", "zor", "alt", "union"] = "mess3",
    num_sequences: int = 1000,
    sequence_length: int = 1000,
    seed: int = 0,
):
    gen = GENERATORS[generator]
    results = compute_baselines(gen, num_sequences, sequence_length, seed)

    print(f"baselines for {generator}:")
    print(f"  max entropy (uniform):    {results['max_entropy']:.4f} nats")
    print(f"  marginal entropy:         {results['marginal_entropy']:.4f} nats")
    print(f"  bayes-optimal loss:       {results['bayes_optimal_loss']:.4f} nats")
    print(f"  bayes-optimal entropy:    {results['bayes_optimal_entropy']:.4f} nats")


if __name__ == "__main__":
    tyro.cli(main)
