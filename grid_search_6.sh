#!/bin/bash
# Grid Search Script 3 of 3
# Weight decay: 5e-3 (test both Adam and AdamW)
# Batch sizes: 32, 40 (ordered smallest to largest)
# Learning rate: 1e-4 (fixed)
# Total: 4 configurations

cd "$(dirname "$0")"

echo "Starting Grid Search Script 3 (weight_decay=5e-3), Learning Rate 5e-4..."

# Adam (smaller batch first)
python3 src/train_siamese.py --optimizer adam --scheduler none --learning-rate 0.0005 --batch-size 32 --weight-decay 0.005 --dropout 0.5
python3 src/train_siamese.py --optimizer adam --scheduler none --learning-rate 0.0005 --batch-size 40 --weight-decay 0.005 --dropout 0.5

# AdamW (smaller batch first)
python3 src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0005 --batch-size 32 --weight-decay 0.005 --dropout 0.5
python3 src/train_siamese.py --optimizer adamw --scheduler none --learning-rate 0.0005 --batch-size 40 --weight-decay 0.005 --dropout 0.5

echo "Grid Search Script 3 completed!"
