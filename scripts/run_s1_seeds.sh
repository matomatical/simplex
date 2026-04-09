#!/bin/bash
# Run simplex1.py for seeds 4-13 in series on one device (10M steps each)
# Total: ~37 hours

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=10485760  # 2^20 * 10

for SEED in $(seq 4 13); do
    METRICS="logs/s1_long_seed${SEED}_${TIMESTAMP}.jsonl"
    echo "$(date): Starting seed=$SEED -> $METRICS"
    tpu-device 0 python simplex1.py \
        --seed $SEED \
        --num-steps $NUM_STEPS \
        --no-vis \
        --metrics-file "$METRICS"
    echo "$(date): Finished seed=$SEED"
done

echo "$(date): All seeds complete."
