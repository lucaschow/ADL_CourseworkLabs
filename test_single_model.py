#!/usr/bin/env python3
"""
Test a single model on the test set.
Automatically detects architecture (old/variant/new) and uses the correct model class.
Usage: python test_single_model.py path/to/model.pth [--dropout 0.5]
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
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

def main():
    parser = argparse.ArgumentParser(description="Test a single model on the test set")
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value (default: 0.5)")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    print("="*80)
    print(f"Testing model: {model_path}")
    print(f"Device: {DEVICE}")
    print(f"Dropout: {args.dropout}")
    print("="*80)
    
    # Setup test dataset - MUST match training code exactly
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    print("\nLoading test dataset...")
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
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Load state dict to detect architecture
    print(f"\nLoading model...")
    state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
    architecture = detect_architecture(state_dict)
    
    # Get appropriate model class
    ModelClass = get_model_class(architecture)
    model = ModelClass(in_channels=3, dropout=args.dropout)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    # Display architecture info
    arch_desc = {
        'old': '1024->512->3',
        'variant': '1024->256->3',
        'new': '1024->256->128->3'
    }.get(architecture, architecture)
    print(f"Detected architecture: {arch_desc}")
    
    print("\nEvaluating on test set...")
    results = {"preds": [], "labels": []}
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (img_a, img_b, labels) in enumerate(test_loader):
            img_a = img_a.to(DEVICE)
            img_b = img_b.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(img_a, img_b)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))
            results["labels"].extend(list(labels.cpu().numpy()))

    num_classes = 3
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(results["labels"], results["preds"]):
        conf_matrix[t,p] += 1
    print("Confusion Matrix:")
    print(conf_matrix)
    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / len(test_loader)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Architecture: {arch_desc}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {average_loss:.5f}")
    print("="*80)

if __name__ == "__main__":
    main()

