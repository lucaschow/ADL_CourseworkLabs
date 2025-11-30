#!/usr/bin/env python3
"""
Visualize model predictions: show correct/incorrect pairs with Grad-CAM fixation maps.
Usage: python visualize_predictions.py path/to/model.pth [--num-examples N] [--dropout 0.5]
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataloader import ProgressionDataset

# Copy model classes (same as test_single_model.py)
from torch import nn
from typing import Union

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

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Label names
LABEL_NAMES = {
    0: "Forward (progressing)",
    1: "Reverse (regressing)", 
    2: "Unrelated (different recipe)"
}

class GradCAM:
    """Grad-CAM for generating fixation maps"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, input_img, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model.branch(input_img)
        
        if class_idx is None:
            # Use the class with highest score
            # But we need the full model output, so we'll need to handle this differently
            # For now, we'll use the feature map directly
            pass
        
        # Backward pass
        if self.gradients is not None:
            # Compute weights
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            # Generate CAM
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        return None

def generate_gradcam_for_pair(model, img_a, img_b, device):
    """Generate Grad-CAM maps for both images in a pair using feature activations"""
    model.eval()
    
    img_a = img_a.unsqueeze(0).to(device)
    img_b = img_b.unsqueeze(0).to(device)
    
    # Store activations from the last conv layer (before AdaptiveAvgPool)
    activations_a = []
    activations_b = []
    gradients_a = []
    gradients_b = []
    
    def forward_hook_a(module, input, output):
        activations_a.append(output)
    
    def forward_hook_b(module, input, output):
        activations_b.append(output)
    
    def backward_hook_a(module, grad_input, grad_output):
        gradients_a.append(grad_output[0])
    
    def backward_hook_b(module, grad_input, grad_output):
        gradients_b.append(grad_output[0])
    
    # Register hooks on the last conv layer (chunk4's last conv, before AdaptiveAvgPool)
    # This is the Conv2d at index 3 in chunk4
    hook_a = model.branch.chunk4[3].register_forward_hook(forward_hook_a)
    hook_b = model.branch.chunk4[3].register_forward_hook(forward_hook_b)
    hook_grad_a = model.branch.chunk4[3].register_full_backward_hook(backward_hook_a)
    hook_grad_b = model.branch.chunk4[3].register_full_backward_hook(backward_hook_b)
    
    # Forward pass
    b1 = model.branch(img_a)
    b2 = model.branch(img_b)
    b1_flat = b1.view(b1.size(0), -1)
    b2_flat = b2.view(b2.size(0), -1)
    x = torch.cat((b1_flat, b2_flat), dim=1)
    x = model.fc1(x)
    x = model.ReLU(x)
    x = model.dropout(x)
    output = model.fc2(x)
    
    # Get prediction
    pred = output.argmax(dim=1).item()
    
    # Backward pass
    model.zero_grad()
    output[0, pred].backward()
    
    # Generate CAM
    if len(activations_a) > 0 and len(gradients_a) > 0:
        # For image A: compute weighted sum of feature maps
        weights_a = torch.mean(gradients_a[0], dim=(2, 3), keepdim=True)
        cam_a = torch.sum(weights_a * activations_a[0], dim=1, keepdim=True)
        cam_a = F.relu(cam_a)
        cam_a = cam_a.squeeze().detach().cpu().numpy()
        cam_a = (cam_a - cam_a.min()) / (cam_a.max() - cam_a.min() + 1e-8)
    else:
        # Fallback: use simple gradient magnitude
        cam_a = np.ones((14, 14)) * 0.5  # Default heatmap
    
    if len(activations_b) > 0 and len(gradients_b) > 0:
        # For image B
        weights_b = torch.mean(gradients_b[0], dim=(2, 3), keepdim=True)
        cam_b = torch.sum(weights_b * activations_b[0], dim=1, keepdim=True)
        cam_b = F.relu(cam_b)
        cam_b = cam_b.squeeze().detach().cpu().numpy()
        cam_b = (cam_b - cam_b.min()) / (cam_b.max() - cam_b.min() + 1e-8)
    else:
        # Fallback
        cam_b = np.ones((14, 14)) * 0.5
    
    # Remove hooks
    hook_a.remove()
    hook_b.remove()
    hook_grad_a.remove()
    hook_grad_b.remove()
    
    return cam_a, cam_b, pred

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image using matplotlib/PIL (no cv2 dependency)"""
    # Resize heatmap to match image dimensions using PIL
    if heatmap.shape != (img.shape[0], img.shape[1]):
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((img.shape[1], img.shape[0]), Image.LANCZOS)
        heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0
    else:
        heatmap_resized = heatmap
    
    # Normalize heatmap to [0, 1]
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Apply colormap (JET-like using matplotlib)
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Get RGB, ignore alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Ensure image is in correct format
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = img.astype(np.uint8)
    else:
        img_rgb = img.astype(np.uint8)
    
    # Blend images: overlayed = (1-alpha) * img + alpha * heatmap
    overlayed = (1 - alpha) * img_rgb + alpha * heatmap_colored
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    
    return overlayed

def visualize_pairs(model, test_loader, device, num_examples=5):
    """Collect and visualize correct/incorrect pairs"""
    model.eval()
    
    correct_pairs = []
    incorrect_pairs = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for batch_idx, (img_a, img_b, labels) in enumerate(test_loader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)
            
            logits = model(img_a, img_b)
            preds = logits.argmax(dim=-1)
            
            # Get dataset to access image paths
            dataset = test_loader.dataset
            batch_start_idx = batch_idx * test_loader.batch_size
            
            for i in range(len(labels)):
                idx = batch_start_idx + i
                if idx >= len(dataset.fixed_pairs):
                    break
                    
                img_a_path, img_b_path, true_label = dataset.fixed_pairs[idx]
                pred = preds[i].item()
                true_label_val = labels[i].item()
                
                pair_info = {
                    'img_a_path': img_a_path,
                    'img_b_path': img_b_path,
                    'true_label': true_label_val,
                    'pred_label': pred,
                    'img_a_tensor': img_a[i].cpu(),
                    'img_b_tensor': img_b[i].cpu(),
                }
                
                if pred == true_label_val:
                    correct_pairs.append(pair_info)
                else:
                    incorrect_pairs.append(pair_info)
    
    print(f"Found {len(correct_pairs)} correct pairs, {len(incorrect_pairs)} incorrect pairs")
    
    # Select examples to visualize
    num_correct = min(num_examples, len(correct_pairs))
    num_incorrect = min(num_examples, len(incorrect_pairs))
    
    # Visualize correct pairs
    if num_correct > 0:
        print(f"\nVisualizing {num_correct} correct pairs...")
        visualize_pair_group(correct_pairs[:num_correct], model, device, "Correct Predictions")
    
    # Visualize incorrect pairs
    if num_incorrect > 0:
        print(f"\nVisualizing {num_incorrect} incorrect pairs...")
        visualize_pair_group(incorrect_pairs[:num_incorrect], model, device, "Incorrect Predictions")
    
    return correct_pairs, incorrect_pairs

def visualize_pair_group(pairs, model, device, title):
    """Visualize a group of pairs with Grad-CAM"""
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 4, figsize=(16, 4*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, y=1.0)
    
    for pair_idx, pair_info in enumerate(pairs):
        img_a_path = pair_info['img_a_path']
        img_b_path = pair_info['img_b_path']
        true_label = pair_info['true_label']
        pred_label = pair_info['pred_label']
        img_a_tensor = pair_info['img_a_tensor']
        img_b_tensor = pair_info['img_b_tensor']
        
        # Load original images
        img_a_orig = np.array(Image.open(img_a_path).convert('RGB'))
        img_b_orig = np.array(Image.open(img_b_path).convert('RGB'))
        
        # Generate Grad-CAM
        cam_a, cam_b, _ = generate_gradcam_for_pair(model, img_a_tensor, img_b_tensor, device)
        
        # Overlay heatmaps
        img_a_heat = overlay_heatmap(img_a_orig, cam_a)
        img_b_heat = overlay_heatmap(img_b_orig, cam_b)
        
        # Plot
        axes[pair_idx, 0].imshow(img_a_orig)
        axes[pair_idx, 0].set_title(f"Image A\n{Path(img_a_path).name}")
        axes[pair_idx, 0].axis('off')
        
        axes[pair_idx, 1].imshow(img_a_heat)
        axes[pair_idx, 1].set_title("Image A + Fixation Map")
        axes[pair_idx, 1].axis('off')
        
        axes[pair_idx, 2].imshow(img_b_orig)
        axes[pair_idx, 2].set_title(f"Image B\n{Path(img_b_path).name}")
        axes[pair_idx, 2].axis('off')
        
        axes[pair_idx, 3].imshow(img_b_heat)
        axes[pair_idx, 3].set_title("Image B + Fixation Map")
        axes[pair_idx, 3].axis('off')
        
        # Add label info
        label_text = f"True: {LABEL_NAMES[true_label]}\nPred: {LABEL_NAMES[pred_label]}"
        if true_label != pred_label:
            label_text += "\n❌ WRONG"
        else:
            label_text += "\n✓ CORRECT"
        
        fig.text(0.5, 0.98 - (pair_idx + 1) * (1.0 / (n_pairs + 1)), 
                label_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat' if true_label == pred_label else 'lightcoral', alpha=0.5))
    
    plt.tight_layout()
    output_file = f"visualization_{title.lower().replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions with Grad-CAM")
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value (default: 0.5)")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples per category (default: 5)")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    print("="*80)
    print(f"Loading model: {model_path}")
    print("="*80)
    
    # Load model
    model = Siamese(in_channels=3, dropout=args.dropout)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    # Load test dataset
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
        batch_size=1,  # Use batch_size=1 for easier indexing
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Visualize
    correct_pairs, incorrect_pairs = visualize_pairs(model, test_loader, DEVICE, args.num_examples)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(f"Total correct: {len(correct_pairs)}")
    print(f"Total incorrect: {len(incorrect_pairs)}")
    print(f"Accuracy: {len(correct_pairs) / (len(correct_pairs) + len(incorrect_pairs)) * 100:.2f}%")

if __name__ == "__main__":
    main()

