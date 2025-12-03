#!/usr/bin/env python3
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

# Note we didnt implement the following for switching architectures so ignore code below
class SiameseOld(nn.Module):
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
    keys = list(state_dict.keys())
    if 'fc3.weight' in keys:
        return 'new' 
    
    
    if 'fc2.weight' in keys:
        fc2_shape = state_dict['fc2.weight'].shape
        if fc2_shape[0] == 3:  
            if fc2_shape[1] == 256:
                return 'variant' 
            elif fc2_shape[1] == 512:
                return 'old'  
    
    return 'old'

def get_model_class(architecture):
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
    print(f"\nLoading model...")
    state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
    architecture = detect_architecture(state_dict)
    ModelClass = get_model_class(architecture)
    model = ModelClass(in_channels=3, dropout=args.dropout)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    # not needed
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
    print("Running model on test...:")
    print(f"Architecture: {arch_desc}") 
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {average_loss:.5f}")

if __name__ == "__main__":
    main()

