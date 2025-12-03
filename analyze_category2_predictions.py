#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataloader import ProgressionDataset

from torch import nn

class Branch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.chunk1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.chunk2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.chunk3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.chunk4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.apply(self.initialise_layer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.chunk1(images)           
        x = self.chunk2(x)           
        x = self.chunk3(x)           
        x = self.chunk4(x) 
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias") and layer.bias is not None:  
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight") and layer.weight is not None:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)

class Siamese(nn.Module):
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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


LABEL_NAMES = {
    0: "Forward (progressing)",
    1: "Reverse (regressing)", 
    2: "Unrelated (different recipe)"
}

def analyze_category2_predictions(model, test_loader, device, num_examples=10):
    model.eval()
    correct_category2 = []  
    incorrect_category2 = []  
    
    with torch.no_grad():
        for batch_idx, (img_a, img_b, labels) in enumerate(test_loader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)
            logits = model(img_a, img_b)
            preds = logits.argmax(dim=-1)
            dataset = test_loader.dataset
            batch_start_idx = batch_idx * test_loader.batch_size
            for i in range(len(labels)):
                idx = batch_start_idx + i
                if idx >= len(dataset.fixed_pairs):
                    break
                    
                img_a_path, img_b_path, true_label = dataset.fixed_pairs[idx]
                pred = preds[i].item()
                true_label_val = labels[i].item()
                
                # Only consider pairs that ARE category 2 (different recipe)
                if true_label_val == 2:
                    pair_info = {
                        'img_a_path': img_a_path,
                        'img_b_path': img_b_path,
                        'true_label': true_label_val,
                        'pred_label': pred,
                        'index': Path(img_a_path).stem.replace('_1', ''), 
                    }
                    if pred == 2:
                        correct_category2.append(pair_info)
                    else:
                        incorrect_category2.append(pair_info)
    
    print(f" No. cat 2 correct predictions: {len(correct_category2)}")
    print(f"  No. incorrect cat 2 predictions: {len(incorrect_category2)}")
    if len(correct_category2) + len(incorrect_category2) > 0:
        print(f"  Accuracy on cat 2: {len(correct_category2) / (len(correct_category2) + len(incorrect_category2)) * 100:.2f}%")
    
    return correct_category2, incorrect_category2

def visualize_category2_pairs(pairs, title, num_examples=10):
    n_pairs = min(num_examples, len(pairs))
    if n_pairs == 0:
        print(f"No pairs to visualize for {title}")
        return
    
    base_filename = f"category2_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
    metadata_file = f"{base_filename}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"{title}\n")
        f.write("="*80 + "\n\n")
        for pair_idx, pair_info in enumerate(pairs[:n_pairs]):
            img_a_path = pair_info['img_a_path']
            img_b_path = pair_info['img_b_path']
            true_label = pair_info['true_label']
            pred_label = pair_info['pred_label']
            index = pair_info['index']
            
            f.write(f"Pair {pair_idx + 1}:\n")
            f.write(f"  Index: {index}\n")
            f.write(f"  Image A: {Path(img_a_path).name}\n")
            f.write(f"  Image B: {Path(img_b_path).name}\n")
            f.write(f"  True Label: {LABEL_NAMES[true_label]} ({true_label})\n")
            f.write(f"  Predicted Label: {LABEL_NAMES[pred_label]} ({pred_label})\n")
            f.write(f"  Status: {'CORRECT' if pred_label == true_label else 'WRONG'}\n")
            f.write("\n")
    print(f"Saved metadata to: {metadata_file}")
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 5*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, y=0.995)
    
    for pair_idx, pair_info in enumerate(pairs[:n_pairs]):
        img_a_path = pair_info['img_a_path']
        img_b_path = pair_info['img_b_path']
        true_label = pair_info['true_label']
        pred_label = pair_info['pred_label']
        index = pair_info['index']

        img_a = np.array(Image.open(img_a_path).convert('RGB'))
        img_b = np.array(Image.open(img_b_path).convert('RGB'))
  
        axes[pair_idx, 0].imshow(img_a)
        axes[pair_idx, 0].set_title(f"Image A\n{Path(img_a_path).name}", fontsize=10)
        axes[pair_idx, 0].axis('off')
        
        axes[pair_idx, 1].imshow(img_b)
        axes[pair_idx, 1].set_title(f"Image B\n{Path(img_b_path).name}", fontsize=10)
        axes[pair_idx, 1].axis('off')
        pred_text = f"Pred: {LABEL_NAMES[pred_label]}"
        true_text = f"True: {LABEL_NAMES[true_label]}"
        status = "✓ CORRECT" if pred_label == true_label else f"❌ WRONG (predicted {LABEL_NAMES[pair_info['pred_label']]})"
        info_text = f"Index: {index}\n{true_text}\n{pred_text}\n{status}"
        color = 'lightgreen' if pred_label == true_label else 'lightcoral'
        
        fig.text(0.5, 0.98 - (pair_idx + 1) * (1.0 / (n_pairs + 1)), 
                info_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    output_file_with_labels = f"{base_filename}_with_labels.png"
    plt.savefig(output_file_with_labels, dpi=150, bbox_inches='tight')
    print(f"Saved visualization with labels to: {output_file_with_labels}")
    plt.close()
    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 5*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    for pair_idx, pair_info in enumerate(pairs[:n_pairs]):
        img_a_path = pair_info['img_a_path']
        img_b_path = pair_info['img_b_path']
        img_a = np.array(Image.open(img_a_path).convert('RGB'))
        img_b = np.array(Image.open(img_b_path).convert('RGB'))
        axes[pair_idx, 0].imshow(img_a)
        axes[pair_idx, 0].axis('off')
        axes[pair_idx, 1].imshow(img_b)
        axes[pair_idx, 1].axis('off')
    
    plt.tight_layout()
    output_file_clean = f"{base_filename}_clean.png"
    plt.savefig(output_file_clean, dpi=150, bbox_inches='tight')
    print(f"Saved clean visualization to: {output_file_clean}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Looking into Cat.2 predictions")
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value (default: 0.5)")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to visualize (default: 10)")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return

    model = Siamese(in_channels=3, dropout=args.dropout)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
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
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    correct_cat2, incorrect_cat2 = analyze_category2_predictions(
        model, test_loader, DEVICE, args.num_examples
    )
    print(f"Correct idenitification:")
    visualize_category2_pairs(correct_cat2, "Cat.2: Correctly Identified", args.num_examples)
    
    print(f"\npairs we missed")
    visualize_category2_pairs(incorrect_cat2, "Category 2: Missed (Wrong Guess)", args.num_examples)

    if len(correct_cat2) + len(incorrect_cat2) > 0:
        print(f"  Accuracy: {len(correct_cat2) / (len(correct_cat2) + len(incorrect_cat2)) * 100:.2f}%")
   

if __name__ == "__main__":
    main()

