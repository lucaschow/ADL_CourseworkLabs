#!/bin/bash
# Grid Search Script 2 of 3
# Weight decay: 1e-3 (test both Adam and AdamW)
# Batch sizes: 32, 40 (ordered smallest to largest)
# Learning rate: 1e-4 (fixed)
# Total: 4 configurations

cd "$(dirname "$0")"

echo "Starting Grid Search Script 1 with smoothing"

python3 src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0001 --batch-size 32 --weight-decay 0.001 --dropout 0.5 --label-smoothing 0.1 --epochs 33
python3 src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0001 --batch-size 32 --weight-decay 0.0001 --dropout 0.5 --label-smoothing 0.1 --epochs 33

python3 src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0001 --batch-size 32 --weight-decay 0.005 --dropout 0.5 --label-smoothing 0.1 --epochs 33

echo "Grid Search Script 1 with smoothing completed!"

