#!/bin/bash
# Launch simplex1.py for 100M steps with checkpointing
# Single seed, ~36 hours at 780 it/s

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=104857600  # 2^20 * 100
SEED=0
CKPT_DIR="checkpoints/s1_100m_seed${SEED}_${TIMESTAMP}"
METRICS="logs/s1_100m_seed${SEED}_${TIMESTAMP}.jsonl"
LOG="logs/s1_100m_seed${SEED}_${TIMESTAMP}.log"

echo "Launching 100M step run: seed=$SEED -> $LOG"
echo "  checkpoints: $CKPT_DIR"
echo "  metrics: $METRICS"

nohup tpu-device 1 python simplex1.py \
    --seed $SEED \
    --num-steps $NUM_STEPS \
    --no-vis \
    --metrics-file "$METRICS" \
    --checkpoint-dir "$CKPT_DIR" \
    > "$LOG" 2>&1 &

echo "PID: $!"
echo "Monitor with: tail -f $LOG"
