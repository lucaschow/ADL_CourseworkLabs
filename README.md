Files within this repo and their descriptions:
- **`src/train_siamese.py`**: Our implementation of SiameseNet.
- **`src/dataloader.py`**: Dataset loader as provided from brief.
- **`test_single_model.py`**: To test a model/.pth file on the test dataset.
- **`evaluate_all_models.py`**: Same as above but runs more than one.
- **`finetune_model1.py`**: Fine-tune an existing model - used for our extension.
- **`analyze_category2_predictions.py`**: Analyze and visualize category 2 predictions to understand model behavior.

Our best models we submit which acieve accuracies as reported can be found in the following files
- **`best_models/base.pth`**: Best model for the base task
- **`best_models/extension.pth`**: Best model for the extension task

To test our best models on the test set, use the `test_single_model.py` script running the commands as laid our below:

**Base Task:**
```bash
python test_single_model.py best_models/base.pth
```

**Extension Task:**
```bash
python test_single_model.py best_models/extension.pth
```



