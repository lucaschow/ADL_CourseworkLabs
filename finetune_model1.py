#!/usr/bin/env python3
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
import json
from pathlib import Path
import sys
import time
import random
from typing import Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dataloader import ProgressionDataset
from train_siamese import Branch, compute_accuracy, DEVICE, Trainer

torch.backends.cudnn.benchmark = True

class WeightedProgressionDataset(ProgressionDataset):
    def __init__(self, root_dir, transform=None, mode='train', 
                 recipe_ids_list=None, epoch_size=None,
                 label_file=None, pair_weights=[0.1, 0.1, 0.8]):
       
        super().__init__(root_dir, transform, mode, recipe_ids_list, epoch_size, label_file)
        self.pair_weights = pair_weights
        print(f"Model fine tuned w/ config: Forward={pair_weights[0]}, Reverse={pair_weights[1]}, Unrelated={pair_weights[2]}")
    
    def _generate_pair(self):
        pair_type = random.choices([0, 1, 2], weights=self.pair_weights, k=1)[0]

        if pair_type == 2: 
            recipe_id_a, recipe_id_b = random.sample(self.recipe_ids, 2)
            img_a_path = random.choice(self.recipes[recipe_id_a])
            img_b_path = random.choice(self.recipes[recipe_id_b])
            label = 2

        else:  
            recipe_id = random.choice(self.recipe_ids)
            steps = self.recipes[recipe_id]

            if len(steps) < 2:
                recipe_id_a, recipe_id_b = random.sample(self.recipe_ids, 2)
                img_a_path = random.choice(self.recipes[recipe_id_a])
                img_b_path = random.choice(self.recipes[recipe_id_b])
                label = 2
            else:
                idx_1, idx_2 = random.sample(range(len(steps)), 2)

                if pair_type == 0:
                    idx_a = min(idx_1, idx_2)
                    idx_b = max(idx_1, idx_2)
                    label = 0
                else:
                    idx_a = max(idx_1, idx_2)
                    idx_b = min(idx_1, idx_2)
                    label = 1

                img_a_path = steps[idx_a]
                img_b_path = steps[idx_b]

        return img_a_path, img_b_path, label

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

def load_model(model_path, device, dropout=0.5):
    from train_siamese import Siamese
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    architecture = detect_architecture(state_dict)
    model = Siamese(in_channels=3, dropout=dropout, architecture=architecture)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    
    arch_desc = {
        'old': '1024->512->3',
        'variant': '1024->256->3',
        'new': '1024->256->128->3'
    }.get(architecture, architecture)
    
    print(f"Loaded model with architecture: {arch_desc}")
    return model, architecture

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", type=Path, help="Path to the existing model .pth file")
    parser.add_argument("--epochs", type=int, default=5, help="No. epochs to train")
    parser.add_argument("--category2-weight", type=float, default=0.8, 
        help="Catagory 2 weight")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
        help="Learning rate for fine-tuning (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epoch-size", type=int, default=5000, help="Number of pairs per epoch")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout value")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory for logs")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="Directory to save fine-tuned model ")
    parser.add_argument("--save-all-epochs", action="store_true", default=True,
        help="Save model after each epoch ")
    
    args = parser.parse_args()
    
    if not args.model_path.exists():
        print(f"Error file not found: {args.model_path}")
        return
    
    remaining_weight = 1.0 - args.category2_weight
    forward_weight = remaining_weight / 2.0
    reverse_weight = remaining_weight / 2.0
    pair_weights = [forward_weight, reverse_weight, args.category2_weight]
 
    print("Model Config:")
    print(f"Model: {args.model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Pair weights: Class 0={pair_weights[0]:.2f}, Class 1={pair_weights[1]:.2f}, Class 2={pair_weights[2]:.2f}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save all epochs: {args.save_all_epochs}")

    # Load model
    model, architecture = load_model(args.model_path, DEVICE, args.dropout)
    
    # Setup datasets
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    recipe_ids_list = ["P01_R01_instance_0","P01_R01_instance_1","P01_R01_instance_2","P01_R03","P01_R04","P01_R05","P01_R07","P01_R09","P02_R01","P02_R02","P02_R03_instance_0","P02_R03_instance_1","P02_R03_instance_2","P02_R03_instance_3","P02_R05","P02_R07","P02_R08","P02_R10","P02_R11","P03_R01","P03_R02","P03_R03_instance_0","P03_R03_instance_2","P03_R03_instance_3","P03_R04","P03_R05","P03_R06","P03_R07","P03_R08","P03_R09","P03_R10","P04_R01","P04_R02","P04_R03","P04_R04","P04_R05","P04_R06","P05_R01","P05_R02_instance_0","P05_R02_instance_2","P05_R03","P05_R04","P05_R05","P05_R06","P07_R01","P07_R02","P07_R03","P07_R04","P07_R05","P07_R06","P07_R07","P08_R01","P08_R02","P08_R04","P08_R05","P08_R06","P08_R07","P08_R08","P08_R09","P09_R01","P09_R02","P09_R03_instance_0","P09_R03_instance_1","P09_R04","P09_R05","P09_R06"]
    
    # Use weighted dataset for training
    train_dataset = WeightedProgressionDataset(
        root_dir='dataset/train',
        transform=train_tf,
        mode='train',
        recipe_ids_list=recipe_ids_list,
        epoch_size=args.epoch_size,
        pair_weights=pair_weights
    )
    
    test_dataset = ProgressionDataset(
        root_dir='dataset/test',
        transform=eval_tf,
        mode='test',
        label_file='dataset/test_labels.txt'
    )
    val_dataset = ProgressionDataset(
        root_dir='dataset/val',
        transform=eval_tf,
        mode='val',
        label_file='dataset/val_labels.txt'
    )
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    if args.output_dir is None:
        original_dir = args.model_path.parent.name
        cat2_str = f"cat2_{args.category2_weight}".replace(".", "_")
        args.output_dir = args.log_dir / f"finetune_{original_dir}_{cat2_str}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_writer = SummaryWriter(
        str(args.output_dir),
        flush_secs=5
    )
    
    print(f"\nWriting logs to: {args.output_dir}")
    
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, criterion, optimizer, None, summary_writer, DEVICE
    )
    
   
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print("="*80)
    
    best_val_accuracy = trainer.best_val_accuracy  
    best_test_accuracy = 0.0
    epoch_results = []

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        trainer.model.train()
        data_load_start_time = time.time()
        for batch_idx, (img_a, img_b, labels) in enumerate(train_loader):
            img_a = img_a.to(DEVICE)
            img_b = img_b.to(DEVICE)
            labels = labels.to(DEVICE)
            data_load_end_time = time.time()
            
            logits = trainer.model(img_a, img_b)
            loss = trainer.criterion(logits, labels)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = compute_accuracy(labels, preds)

            data_load_time = data_load_end_time - data_load_start_time
            step_time = time.time() - data_load_end_time
            
            if ((trainer.step + 1) % 10) == 0:
                val_loss, val_accuracy = trainer.validate()
                train_eval_loss = trainer.eval_criterion(logits, labels).item()
                trainer.summary_writer.add_scalars("accuracy", {"train":accuracy, "val": val_accuracy}, trainer.step)
                trainer.summary_writer.add_scalars("loss", {"train":train_eval_loss, "val": val_loss}, trainer.step)
                trainer.model.train()
            if ((trainer.step + 1) % 10) == 0:
                epoch_step = trainer.step % len(train_loader)
                print(
                    f"epoch: [{epoch}], "
                    f"step: [{epoch_step}/{len(train_loader)}], "
                    f"batch loss: {loss:.5f}, "
                    f"batch accuracy: {accuracy * 100:2.2f}, "
                    f"data load time: {data_load_time:.5f}, "
                    f"step time: {step_time:.5f}"
                )

            trainer.step += 1
            data_load_start_time = time.time()
        trainer.summary_writer.add_scalar("epoch", epoch, trainer.step)
        val_loss, val_accuracy = trainer.validate()
        test_loss, test_accuracy = trainer.test(test_loader)
 
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Validation - Accuracy: {val_accuracy * 100:.2f}% | Loss: {val_loss:.5f}")
        print(f"  Test       - Accuracy: {test_accuracy * 100:.2f}% | Loss: {test_loss:.5f}")
  
        trainer.summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)
        trainer.summary_writer.add_scalar("test_loss", test_loss, epoch)
     
        if args.save_all_epochs:
            epoch_model_path = args.output_dir / f"epoch_{epoch + 1}.pth"
            try:
                torch.save(model.state_dict(), epoch_model_path)
                print(f"  ✓ Epoch {epoch + 1} model saved: {epoch_model_path.name}")
            except OSError as e:
                if "No space left" in str(e) or "Disk quota" in str(e) or e.errno == 122:
                    print(f"  ERROR: Disk full! Cannot save model.")
                    raise
                else:
                    raise
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_test_accuracy = test_accuracy
            best_model_path = args.output_dir / "best_model.pth"
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved (val acc: {val_accuracy * 100:.2f}%): best_model.pth")
            except OSError as e:
                if "No space left" in str(e) or "Disk quota" in str(e) or e.errno == 122:
                    print(f"  ERROR: Disk full! Cannot save model.")
                    raise
                else:
                    raise
        
        epoch_results.append({
            'epoch': epoch + 1,
            'val_accuracy': float(val_accuracy),
            'val_loss': float(val_loss),
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
        })
        
    
        trainer.model.train()

    print("Fine tuning results...")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Best Test Accuracy: {best_test_accuracy * 100:.2f}%")

    results_json = {
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_accuracy_percent": float(best_val_accuracy * 100),
        "best_test_accuracy": float(best_test_accuracy),
        "best_test_accuracy_percent": float(best_test_accuracy * 100),
        "final_test_accuracy": float(epoch_results[-1]['test_accuracy']),
        "final_test_accuracy_percent": float(epoch_results[-1]['test_accuracy'] * 100),
        "final_test_loss": float(epoch_results[-1]['test_loss']),
        "epoch_results": epoch_results,
        "fine_tuning": {
            "original_model": str(args.model_path),
            "architecture": architecture,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "pair_weights": pair_weights,
            "category2_weight": args.category2_weight,
        },
    }
    results_file = args.output_dir / "finetune_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nFine-tuning results saved to: {results_file}")
    
    summary_writer.close()
    print(f"Fine-tuned model saved to: {args.output_dir}/best_model.pth")

if __name__ == "__main__":
    main()