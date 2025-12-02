#!/usr/bin/env python3
"""
Test a single model on the test set (OLD FC architecture: 1024->512->3).
Usage: python test_single_model_old.py path/to/model.pth [--dropout 0.5]
"""
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataloader import ProgressionDataset
from train_siamese_old import Siamese, compute_accuracy, DEVICE

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
    
    # Create and load model
    print(f"\nLoading model...")
    model = Siamese(in_channels=3, dropout=args.dropout)
    state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE)
    model.eval()
    
    print("Evaluating on test set...")
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
    
    accuracy = compute_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    average_loss = total_loss / len(test_loader)
    
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {average_loss:.5f}")
    print("="*80)

if __name__ == "__main__":
    main()

