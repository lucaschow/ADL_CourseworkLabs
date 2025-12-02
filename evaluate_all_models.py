#!/usr/bin/env python3
"""
Evaluate all saved models from grid search on the test set.
Automatically detects architecture (old/new/variant) and uses the correct model class.
Usage: python evaluate_all_models.py
"""
import torch
import numpy as np
from pathlib import Path
import re
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataloader import ProgressionDataset
from train_siamese import Branch, compute_accuracy, DEVICE

# Define all three architectures
class SiameseOld(nn.Module):
    """Old architecture: 1024 -> 512 -> 3"""
    def __init__(self, in_channels=3, dropout=0.5):
        super().__init__()
        self.branch = Branch(channels=in_channels)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 3)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, anchor, comparator):
        b1 = self.branch(anchor)
        b2 = self.branch(comparator)
        b1 = b1.view(b1.size(0), -1)
        b2 = b2.view(b2.size(0), -1) 
        x = torch.cat((b1, b2), dim=1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SiameseVariant(nn.Module):
    """Variant architecture: 1024 -> 256 -> 3"""
    def __init__(self, in_channels=3, dropout=0.5):
        super().__init__()
        self.branch = Branch(channels=in_channels)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 3)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, anchor, comparator):
        b1 = self.branch(anchor)
        b2 = self.branch(comparator)
        b1 = b1.view(b1.size(0), -1)
        b2 = b2.view(b2.size(0), -1) 
        x = torch.cat((b1, b2), dim=1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SiameseNew(nn.Module):
    """New architecture: 1024 -> 256 -> 128 -> 3"""
    def __init__(self, in_channels=3, dropout=0.5):
        super().__init__()
        self.branch = Branch(channels=in_channels)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, anchor, comparator):
        b1 = self.branch(anchor)
        b2 = self.branch(comparator)
        b1 = b1.view(b1.size(0), -1)
        b2 = b2.view(b2.size(0), -1) 
        x = torch.cat((b1, b2), dim=1)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def detect_architecture(state_dict):
    """
    Detect architecture from state_dict keys.
    Returns: 'old', 'variant', or 'new'
    """
    keys = list(state_dict.keys())
    
    # Check if fc3 exists -> new architecture
    if 'fc3.weight' in keys:
        return 'new'  # 1024 -> 256 -> 128 -> 3
    
    # Check fc2 weight shape to distinguish old vs variant
    if 'fc2.weight' in keys:
        fc2_shape = state_dict['fc2.weight'].shape
        if fc2_shape[0] == 3:  # Output is 3 classes
            if fc2_shape[1] == 256:
                return 'variant'  # 1024 -> 256 -> 3
            elif fc2_shape[1] == 512:
                return 'old'  # 1024 -> 512 -> 3
    
    # Default to old if we can't determine
    return 'old'

def get_model_class(architecture):
    """Get the appropriate model class for the architecture"""
    if architecture == 'old':
        return SiameseOld
    elif architecture == 'variant':
        return SiameseVariant
    elif architecture == 'new':
        return SiameseNew
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def parse_config_from_dir(dir_name):
    """Extract hyperparameters from directory name"""
    config = {}
    opt_match = re.search(r'opt=(\w+)', dir_name)
    if opt_match:
        config['optimizer'] = opt_match.group(1)
    
    sched_match = re.search(r'sched=(\w+)', dir_name)
    if sched_match:
        config['scheduler'] = sched_match.group(1)
    
    bs_match = re.search(r'bs=(\d+)', dir_name)
    if bs_match:
        config['batch_size'] = int(bs_match.group(1))
    
    lr_match = re.search(r'lr=([\d.e-]+)', dir_name)
    if lr_match:
        lr_str = lr_match.group(1)
        if 'e' in lr_str:
            base, exp = lr_str.split('e-')
            config['learning_rate'] = float(base) * (10 ** -int(exp))
        else:
            config['learning_rate'] = float(lr_str)
    
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
    
    # Parse architecture from directory name if present
    arch_match = re.search(r'arch=(\w+)', dir_name)
    if arch_match:
        config['architecture'] = arch_match.group(1)
    
    return config

def evaluate_model(model_path, test_loader, device, dropout=0.5):
    """Load model and evaluate on test set - automatically detects architecture"""
    # Load state dict to detect architecture
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    architecture = detect_architecture(state_dict)
    
    # Get appropriate model class
    ModelClass = get_model_class(architecture)
    model = ModelClass(in_channels=3, dropout=dropout)
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    # Evaluate
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
    
    return accuracy, average_loss, architecture

def main():
    # Setup test dataset
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dataset = ProgressionDataset(
        root_dir='dataset/test', 
        transform=eval_transform, 
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
            accuracy, loss, architecture = evaluate_model(model_path, test_loader, DEVICE, dropout=0.5)
            results.append({
                'path': str(model_path),
                'dir': dir_name,
                'config': config,
                'test_accuracy': accuracy,
                'test_loss': loss,
                'architecture': architecture
            })
            print(f"  Architecture: {architecture}")
            print(f"  Test Accuracy: {accuracy * 100:.2f}%")
            print(f"  Test Loss: {loss:.5f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 100)
    print("\nSUMMARY OF ALL RESULTS:")
    print("=" * 100)
    
    # Sort by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    for i, result in enumerate(results, 1):
        config = result['config']
        arch = result['architecture']
        arch_desc = {
            'old': '1024->512->3',
            'variant': '1024->256->3',
            'new': '1024->256->128->3'
        }.get(arch, arch)
        
        print(f"\n{i}. Test Accuracy: {result['test_accuracy'] * 100:.2f}% | Test Loss: {result['test_loss']:.5f}")
        print(f"   Architecture: {arch_desc}")
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
        arch_desc = {
            'old': '1024->512->3',
            'variant': '1024->256->3',
            'new': '1024->256->128->3'
        }.get(best['architecture'], best['architecture'])
        print(f"Architecture: {arch_desc}")
        print(f"\nConfiguration:")
        for key, value in best['config'].items():
            print(f"  {key}: {value}")
        print(f"\nModel Path: {best['path']}")
        print(f"Directory: {best['dir']}")

if __name__ == "__main__":
    main()
