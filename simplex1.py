import datetime
import sys

import jax
import jax.numpy as jnp
import einops
import optax

import tyro
import tqdm

import strux
from transformer import SequenceTransformer
from generators import Sequence, MESS3, ZOR, ALT

from typing import Self, Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray


# HYPER-PARAMETER   Shai+2024   Notes
# ----------------------------------------------------------------------------
# max_examples      10
# num_blocks        4
# embed_size        64
# num_heads         1
# head_size         8           num_heads x head_size NOT equal to embed size!
# mlp_size          256
# learning_rate     0.01
# batch_size        64
# optimiser         SGD
# num_steps         1 million   seems like overkill


def main(
    # data config
    sequence_length: int  = 10,
    generator: Literal[
        'mess3',
        'zor',
        'alt',
    ]                     = 'mess3',
    # model config
    num_blocks: int       = 4,
    embed_size: int       = 64,
    num_heads: int        = 1,
    head_size: int        = 8,
    mlp_size: int         = 256,
    # training config
    learning_rate: float  = 0.01,
    batch_size: int       = 64,
    num_steps: int        = 1024 * 1024,
    opt: Literal[
        'sgd',
        'adam',
    ]                     = 'sgd',
    # evals config
    eval_batch_size: int  = 1024,
    eval_period: int      = 64,
    # experiment config
    seed: int             = 42,
    train: bool           = True,
):
    args = ' '.join(sys.argv)
    start_time = datetime.datetime.now()
    print("configuration:")
    config = locals()
    for config_key, config_value in config.items():
        print(f"* {config_key:30s}: {config_value!r}")
    key = jax.random.key(seed=seed)


    print("initialising training distribution...")
    key_tasks, key = jax.random.split(key)
    if generator == 'zor':
        sequence_generator = ZOR
    elif generator == 'mess3':
        sequence_generator = MESS3
    elif generator == 'alt':
        sequence_generator = ALT
    
    print("example sequences")
    key_examples, key = jax.random.split(key)
    for i in range(10):
        key_gen, key_examples = jax.random.split(key_examples)
        seq = sequence_generator.sample(
            key=key_gen,
            sequence_length=sequence_length,
        )
        print("*", seq)

    
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
    print(strux.size(model), "parameters")
    

    print("initialising optimiser")
    if opt == 'sgd':
        optimiser = optax.sgd(learning_rate=learning_rate)
    elif opt == 'adam':
        optimiser = optax.adam(learning_rate=learning_rate)
    opt_state = optimiser.init(model)


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
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss
    

    if not train: return

    print("starting training loop...")
    # training loop
    for t in tqdm.trange(num_steps):
        # train model for one step
        key_sgd, key = jax.random.split(key)
        model, opt_state, loss = train_step(
            key_sgd,
            model,
            opt_state,
        )

        # evals
        if (t + 1) % eval_period == 0:
            tqdm.tqdm.write(f"step {t+1} loss {loss:.6f} nats per tok")


    print("done!")
    end_time = datetime.datetime.now()


    # if run_name is not None:
    #     run_path = f"logs/{run_name}.json"
    #     print(f"saving results to {run_path!r}...")
    #     data = {
    #         'times': {
    #             'start': start_time.timestamp(),
    #             'end': end_time.timestamp(),
    #         },
    #         'config': config,
    #     }
    #     os.makedirs(os.path.dirname(run_path), exist_ok=True)
    #     with open(run_path, 'w') as outfile:
    #         json.dump(obj=data, fp=outfile, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
