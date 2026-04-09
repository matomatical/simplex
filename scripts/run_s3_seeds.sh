#!/bin/bash
# Launch simplex3.py on 4 TPU devices with different seeds
# Default params, one run per device

SEEDS=(0 1 2 3)
DEVICES=(0 1 2 3)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for i in 0 1 2 3; do
    SEED=${SEEDS[$i]}
    DEV=${DEVICES[$i]}
    LOG="logs/s3_seed${SEED}_${TIMESTAMP}.log"
    METRICS="logs/s3_seed${SEED}_${TIMESTAMP}.jsonl"

    echo "Launching seed=$SEED on device $DEV -> $LOG"
    nohup tpu-device $DEV venv/bin/python simplex3.py \
        --seed $SEED \
        --metrics-file "$METRICS" \
        > "$LOG" 2>&1 &
done

echo "All launched. PIDs:"
jobs -p
echo "Monitor with: tail -f logs/s3_seed*_${TIMESTAMP}.log"
