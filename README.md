# ADL Coursework - Recipe Progression Classification

## Overview
This repository contains a Siamese network implementation for classifying recipe progression pairs into three categories:
- **Category 0**: Forward progression (same recipe, progressing)
- **Category 1**: Reverse progression (same recipe, regressing)
- **Category 2**: Unrelated (different recipes)

## File Structure

### Main Scripts
- **`src/train_siamese.py`**: Main training script for the Siamese network. Supports multiple architectures and hyperparameter configurations.
- **`src/dataloader.py`**: Dataset loader for the ProgressionDataset class. Handles training pair generation and test/validation data loading.
- **`test_single_model.py`**: Test a single model on the test set. Automatically detects model architecture.
- **`evaluate_all_models.py`**: Evaluate all saved models from grid search experiments.
- **`finetune_model1.py`**: Fine-tune an existing model with weighted sampling (useful for addressing category 2 performance).
- **`analyze_category2_predictions.py`**: Analyze and visualize category 2 predictions to understand model behavior.

### Best Models
- **`best_models/base.pth`**: Best model for the base task
- **`best_models/extension.pth`**: Best model for the extension task

## Running the Best Models

### For Markers: Testing Our Best Models

To test our best models on the test set, use the `test_single_model.py` script:

**Base Task:**
```bash
python test_single_model.py best_models/base.pth
```

**Extension Task:**
```bash
python test_single_model.py best_models/extension.pth
```

The script will:
- Automatically detect the model architecture
- Load the model weights
- Evaluate on the test set
- Display accuracy, loss, and confusion matrix

```

## Dataset Structure
- `dataset/train/`: Training recipe folders with step images
- `dataset/val/`: Validation images with `val_labels.txt`
- `dataset/test/`: Test images with `test_labels.txt`


