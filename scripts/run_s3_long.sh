#!/bin/bash
# Launch simplex3.py long runs (10M steps) on 4 devices, no visualisation
# Seeds 0-3, same hyperparameters

SEEDS=(0 1 2 3)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=10485760  # 2^20 * 10

# Use tpu0 devices 0-3
for i in 0 1 2 3; do
    SEED=${SEEDS[$i]}
    LOG="logs/s3_long_seed${SEED}_${TIMESTAMP}.log"
    METRICS="logs/s3_long_seed${SEED}_${TIMESTAMP}.jsonl"

    echo "Launching seed=$SEED on tpu0/dev$i -> $LOG"
    nohup tpu-device $i python simplex3.py \
        --seed $SEED \
        --num-steps $NUM_STEPS \
        --no-vis \
        --metrics-file "$METRICS" \
        > "$LOG" 2>&1 &
done

echo "All launched. PIDs:"
jobs -p
echo "Monitor with: tail -f logs/s3_long_seed*_${TIMESTAMP}.log"
