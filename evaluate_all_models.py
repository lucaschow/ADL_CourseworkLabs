#!/usr/bin/env python3
"""
Evaluate all saved models from grid search on the test set.
Finds all best_model.pth files, loads them, evaluates on test set, and reports results.
"""
import torch
import numpy as np
from pathlib import Path
import glob
import re
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import ProgressionDataset
from train_siamese import Siamese, compute_accuracy, DEVICE

def parse_config_from_dir(dir_name):
    """Extract hyperparameters from directory name like 'opt=adam_sched=none_bs=32_lr=1e-4_wd=1e-4_run_0'"""
    config = {}
    # Parse optimizer
    opt_match = re.search(r'opt=(\w+)', dir_name)
    if opt_match:
        config['optimizer'] = opt_match.group(1)
    
    # Parse scheduler
    sched_match = re.search(r'sched=(\w+)', dir_name)
    if sched_match:
        config['scheduler'] = sched_match.group(1)
    
    # Parse batch size
    bs_match = re.search(r'bs=(\d+)', dir_name)
    if bs_match:
        config['batch_size'] = int(bs_match.group(1))
    
    # Parse learning rate
    lr_match = re.search(r'lr=([\d.e-]+)', dir_name)
    if lr_match:
        lr_str = lr_match.group(1)
        # Convert "1e-4" to 0.0001
        if 'e' in lr_str:
            base, exp = lr_str.split('e-')
            config['learning_rate'] = float(base) * (10 ** -int(exp))
        else:
            config['learning_rate'] = float(lr_str)
    
    # Parse weight decay
    wd_match = re.search(r'wd=([\d.e-]+)', dir_name)
    if wd_match:
        wd_str = wd_match.group(1)
        if wd_str == '0':
            config['weight_decay'] = 0.0
        elif 'e' in wd_str:
            base, exp = wd_str.split('e-')
            config['weight_decay'] = float(base) * (10 ** -int(exp))
        else:
            config['weight_decay'] = float(wd_str)
    
    return config

def evaluate_model(model_path, test_loader, device):
    """Load model and evaluate on test set"""
    # Create model (dropout doesn't matter for eval, but we need to match architecture)
    model = Siamese(in_channels=3, dropout=0.5)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    results = {"preds": [], "labels": []}
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for img_a, img_b, labels in test_loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))
            results["labels"].extend(list(labels.cpu().numpy()))
    
    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / len(test_loader)
    
    return accuracy, average_loss

def main():
    # Setup test dataset - MUST match training code exactly
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = ProgressionDataset(
        root_dir='dataset/test', 
        transform=eval_tf, 
        mode='test', 
        label_file='dataset/test_labels.txt'
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    )
    
    # Find all best_model.pth files
    log_dir = Path("logs")
    model_paths = list(log_dir.glob("*/best_model.pth"))
    
    if not model_paths:
        print("No best_model.pth files found in logs/ directory!")
        return
    
    print(f"Found {len(model_paths)} models to evaluate\n")
    print("=" * 100)
    
    results = []
    
    # Evaluate each model
    for model_path in sorted(model_paths):
        dir_name = model_path.parent.name
        config = parse_config_from_dir(dir_name)
        
        print(f"\nEvaluating: {dir_name}")
        print(f"  Config: {config}")
        
        try:
            accuracy, loss = evaluate_model(model_path, test_loader, DEVICE)
            results.append({
                'path': str(model_path),
                'dir': dir_name,
                'config': config,
                'test_accuracy': accuracy,
                'test_loss': loss
            })
            print(f"  Test Accuracy: {accuracy * 100:.2f}%")
            print(f"  Test Loss: {loss:.5f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 100)
    print("\nSUMMARY OF ALL RESULTS:")
    print("=" * 100)
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        print(f"\n{i}. Test Accuracy: {result['test_accuracy'] * 100:.2f}% | Test Loss: {result['test_loss']:.5f}")
        print(f"   Optimizer: {config.get('optimizer', 'N/A')}")
        print(f"   Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Weight Decay: {config.get('weight_decay', 'N/A')}")
        print(f"   Directory: {result['dir']}")
    
    # Best model
    if results:
        best = results[0]
        print("\n" + "=" * 100)
        print("BEST MODEL:")
        print("=" * 100)
        print(f"Test Accuracy: {best['test_accuracy'] * 100:.2f}%")
        print(f"Test Loss: {best['test_loss']:.5f}")
        print(f"\nConfiguration:")
        for key, value in best['config'].items():
            print(f"  {key}: {value}")
        print(f"\nModel Path: {best['path']}")
        print(f"Directory: {best['dir']}")

if __name__ == "__main__":
    main()

