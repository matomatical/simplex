#!/bin/bash
# Sweep epsilon values across TPU devices, 10 seeds each.
# Usage: bash sweep_epsilon.sh

RESULTS=sweep_results_r2.jsonl
STEPS=10000
VIS=1000000
PYTHON=venv/bin/python
SCRIPT=simplex2.py
FLAGS="--num-steps $STEPS --vis-period $VIS --results-file $RESULTS"

rm -f "$RESULTS"

for SEED in $(seq 0 9); do
    echo "=== seed $SEED ==="

    tpu-device 0 $PYTHON $SCRIPT --epsilon 0.0    --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 1 $PYTHON $SCRIPT --epsilon 0.0003 --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 2 $PYTHON $SCRIPT --epsilon 0.001  --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 3 $PYTHON $SCRIPT --epsilon 0.003  --seed $SEED $FLAGS >/dev/null 2>&1 &
    wait

    tpu-device 0 $PYTHON $SCRIPT --epsilon 0.01   --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 1 $PYTHON $SCRIPT --epsilon 0.03   --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 2 $PYTHON $SCRIPT --epsilon 0.1    --seed $SEED $FLAGS >/dev/null 2>&1 &
    tpu-device 3 $PYTHON $SCRIPT --epsilon 0.3    --seed $SEED $FLAGS >/dev/null 2>&1 &
    wait
done

echo "All done. Results in $RESULTS"
