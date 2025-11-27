#!/bin/bash
# Grid Search Script 2 of 3
# Weight decay: 1e-3 (test both Adam and AdamW)
# Batch sizes: 32, 40 (ordered smallest to largest)
# Learning rate: 1e-4 (fixed)
# Total: 4 configurations

cd "$(dirname "$0")"

echo "Starting Grid Search Script 2 (weight_decay=1e-3)..."

# Adam (smaller batch first)
python src/train_siamese.py --optimizer adam --scheduler none --learning-rate 0.0001 --batch-size 32 --weight-decay 0.001 --dropout 0.5
python src/train_siamese.py --optimizer adam --scheduler none --learning-rate 0.0001 --batch-size 40 --weight-decay 0.001 --dropout 0.5

# AdamW (smaller batch first)
python src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0001 --batch-size 32 --weight-decay 0.001 --dropout 0.5
python src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0001 --batch-size 40 --weight-decay 0.001 --dropout 0.5

echo "Grid Search Script 2 completed!"

