# Hyperparameter Comparison: Our Runs vs Paper (Figure 4b)

Paper: Shai et al. (2025) "Transformers learn factored representations"

## Generative Process

| Parameter | Paper | Ours | Match? |
|---|---|---|---|
| Num factors | 5 | 5 | Y |
| Factor types | 3 Mess3 + 2 Bloch Walk | 5 Mess3 | N |
| Mess3 params | all alpha=0.6, x=0.15 | all alpha=0.6, x=0.15 | Y |
| Bloch Walk params | alpha=1.0, beta=3.0 | N/A | N |
| Vocab size | 3^3 x 4^2 = 432 | 3^5 = 243 | N |
| Joint states | 3^5 = 243 | 3^5 = 243 | Y |
| Factored DoF | 5 x 2 = 10 | 5 x 2 = 10 | Y |
| Noise epsilon values | {0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5} | {0, 0.5} | partial |

## Architecture

| Parameter | Paper | Ours | Match? |
|---|---|---|---|
| Type | GPT-2 decoder-only | decoder-only | Y |
| Layers | 4 | 4 | Y |
| Heads | 3 | 3 | Y |
| d_model | 120 | 120 | Y |
| d_head | 40 | 40 | Y |
| d_ff (MLP) | 480 | 480 | Y |
| Normalization | Pre-LayerNorm | Pre-LayerNorm | Y |
| MLP activation | GELU (GPT-2) | ReLU | N |
| BOS token | yes | no | N |
| Positional embeddings | learned | learned | Y |

## Training

| Parameter | Paper | Ours | Match? |
|---|---|---|---|
| Batch size | 25,000 | 25,000 | Y |
| Optimizer | Adam | Adam | Y |
| Learning rate | 5e-4 | 5e-4 | Y |
| Weight decay | none | none (optax default) | Y |
| Sequence length | 8 | 8 | Y |
| Loss | next-token cross-entropy | next-token cross-entropy | Y |

## Probing / PCA

| Parameter | Paper | Ours | Match? |
|---|---|---|---|
| Activation layer | blocks.3.hook_resid_post (last block, post-MLP) | probe_layer=-1 (same) | Y |
| PCA positions | all positions | all positions | Y |
| PCA threshold | 95% variance | 95% variance | Y |
| Noise channel | per joint token, uniform replacement | per joint token, uniform replacement | Y |

## Remaining Differences

1. **No Bloch Walk**: We use 5 Mess3 instead of 3 Mess3 + 2 Bloch Walk. Vocab is 243 not 432, and we miss Bloch sphere geometry. Joint state count is the same (243).
2. **ReLU vs GELU**: Our MLP uses ReLU; GPT-2 style uses GELU.
3. **No BOS token**: Paper prepends a BOS token to each sequence.
