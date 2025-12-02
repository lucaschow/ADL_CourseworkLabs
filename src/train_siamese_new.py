#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
from dataloader import ProgressionDataset

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
import json
from pathlib import Path

torch.backends.cudnn.benchmark = True #Just improves training speeds, can see later if we want to keep

parser = argparse.ArgumentParser(
    description="Train a Siamese network on ProgressionDataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path("dataset")
parser.add_argument(
    "--dataset-root",
    default=default_dataset_dir,
    type=Path, 
    help="Path to the dataset directory, contains train/va/test splits",
)
parser.add_argument(
    "--log-dir", 
    default=Path("logs"), 
    type=Path,
    help="Directory to store TensorBoard logs",
)
parser.add_argument(
    "--learning-rate", 
    default=1e-4, 
    type=float, 
    help="Base Adam Learning rate"
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of pairs within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--epoch-size",
    default=5000,
    type=int,
    help="Number of training pairs per epoch (default: 5000)",
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to update the best model save",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=0, #ONLY FOR MAC FIX - cpu_count() is not supported on Mac
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0,
    help="L2 weight decay "
)
parser.add_argument("--beta1", type=float, default=0.9,help="Adam Beta1.")
parser.add_argument("--beta2", type=float, default=0.999,help="Adam Beta2.")

# Optimizer selection
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"],
    help="Optimizer type: adam or adamw")

# Scheduler selection
parser.add_argument("--scheduler", type=str, default="none", 
    choices=["none", "cosine", "reduce_on_plateau"],
    help="Learning rate scheduler type")

# Cosine annealing parameters
parser.add_argument("--T-max", type=int, default=None,
    help="T_max for CosineAnnealingLR (default: epochs, only used if scheduler=cosine)")

# Dropout
parser.add_argument("--dropout", type=float, default=0.5,
    help="Dropout probability (0.0 to 1.0)")

# Label smoothing
parser.add_argument("--label-smoothing", type=float, default=0.0,
    help="Label smoothing factor (0.0 = no smoothing, 0.1 = common value)")

# Run ID for multi-machine runs
parser.add_argument("--run-id", type=str, default=None,
    help="Unique run identifier (e.g., random hash) to prevent conflicts across machines")


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    # Training transforms: with augmentation
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()
    ])
    # Val/Test transforms: no augmentation
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    epoch_size = args.epoch_size
    recepie_ids_list = ["P01_R01_instance_0","P01_R01_instance_1","P01_R01_instance_2","P01_R03","P01_R04","P01_R05","P01_R07","P01_R09","P02_R01","P02_R02","P02_R03_instance_0","P02_R03_instance_1","P02_R03_instance_2","P02_R03_instance_3","P02_R05","P02_R07","P02_R08","P02_R10","P02_R11","P03_R01","P03_R02","P03_R03_instance_0","P03_R03_instance_2","P03_R03_instance_3","P03_R04","P03_R05","P03_R06","P03_R07","P03_R08","P03_R09","P03_R10","P04_R01","P04_R02","P04_R03","P04_R04","P04_R05","P04_R06","P05_R01","P05_R02_instance_0","P05_R02_instance_2","P05_R03","P05_R04","P05_R05","P05_R06","P07_R01","P07_R02","P07_R03","P07_R04","P07_R05","P07_R06","P07_R07","P08_R01","P08_R02","P08_R04","P08_R05","P08_R06","P08_R07","P08_R08","P08_R09","P09_R01","P09_R02","P09_R03_instance_0","P09_R03_instance_1","P09_R04","P09_R05","P09_R06"]

    
    train_dataset = ProgressionDataset(root_dir='dataset/train', transform=train_tf, mode='train', recipe_ids_list=recepie_ids_list, epoch_size=epoch_size)
    #you didnt load in the data correctly you melon - we only have train
    test_dataset = ProgressionDataset(root_dir='dataset/test', transform=eval_tf, mode='test', label_file='dataset/test_labels.txt')
    val_dataset = ProgressionDataset(root_dir='dataset/val', transform=eval_tf, mode='val', label_file='dataset/val_labels.txt')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = Siamese(in_channels=3, dropout=args.dropout) #was CNN

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    # Use label smoothing if specified
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Create optimizer
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                               betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                              betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Create scheduler
    scheduler = None
    if args.scheduler == "cosine":
        T_max = args.T_max if args.T_max else args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0)
    elif args.scheduler == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    print(f"Training with: optimizer={args.optimizer}, lr={args.learning_rate}, batch_size={args.batch_size}, weight_decay={args.weight_decay}, dropout={args.dropout}, label_smoothing={args.label_smoothing}, scheduler={args.scheduler}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )
    test_loss, test_accuracy = trainer.test(test_loader)
    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Final test loss: {test_loss:.5f}")
    
    # Save test results to JSON file in log directory
    results_json = {
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "test_loss": float(test_loss),
        "hyperparameters": {
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "epochs": args.epochs,
            "epoch_size": args.epoch_size,
        },
        "augmentation": True,  # Always True now since we hardcoded it
    }
    results_file = Path(log_dir) / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Test results saved to: {results_file}")

    summary_writer.close()

#This is one branch of the cnn before concat
class Branch(nn.Module):
    def __init__(self,  channels):
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
        
       
        #Note - we could think about having a static batch norm initialiser for better learning

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
        self.branch = Branch(channels=in_channels) #initialise once so we get shared weights
        # New FC architecture: 1024 -> 256 -> 128 -> 3
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



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion  # Training criterion (may have label smoothing)
        self.eval_criterion = nn.CrossEntropyLoss()  # Evaluation criterion (no smoothing for fair comparison)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0
        self.best_val_accuracy = 0.0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for img_a, img_b, labels in self.train_loader:
                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                logits =self.model(img_a, img_b)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    val_loss, val_accuracy = self.validate()
                    # Use unsmoothed loss for training loss logging too, for fair comparison
                    train_eval_loss = self.eval_criterion(logits, labels).item()
                    self.summary_writer.add_scalars("accuracy", {"train":accuracy, "val": val_accuracy}, self.step)
                    self.summary_writer.add_scalars("loss", {"train":train_eval_loss, "val": val_loss}, self.step)
                    self.model.train()

                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                val_loss, val_accuracy = self.validate()
                if val_accuracy > self.best_val_accuracy:
                    best_model_path = Path(self.summary_writer.log_dir) / "best_model.pth"
                    try:
                        torch.save(self.model.state_dict(), best_model_path)
                        self.best_val_accuracy = val_accuracy
                        print(f"best model saved with validation accuracy", val_accuracy)
                    except OSError as e:
                        if "No space left" in str(e) or "Disk quota" in str(e) or e.errno == 122:
                            print(f"ERROR: Disk full! Cannot save model. Stopping training.")
                            print(f"Free up disk space and resume training.")
                            raise  # Stop training - can't continue without saving
                        else:
                            raise  # Re-raise other OSErrors
               
                # Step scheduler based on validation metric (for ReduceLROnPlateau)
                if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_accuracy)
                
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
            # Step scheduler after each epoch (for cosine annealing)
            if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            if ((epoch + 1) % 5) == 0:
                best_model_path = Path(self.summary_writer.log_dir) / "best_model.pth"
                if best_model_path.exists():
                    try:
                        self.model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=self.device))
                    except (RuntimeError, EOFError) as e:
                        print(f"Warning: Could not load best_model.pth (may be corrupted): {e}")
                        print("Continuing with current model state...")
                
                test_loss, test_accuracy = self.test(self.test_loader)
                print(f"Epoch {epoch+1}: Test accuracy: {test_accuracy * 100:.2f}%, Test loss: {test_loss:.5f}")
                
                self.summary_writer.add_scalar("test_accuracy", test_accuracy, epoch)
                self.summary_writer.add_scalar("test_loss", test_loss, epoch)
                
                self.model.train()
        best_model_path = Path(self.summary_writer.log_dir) / "best_model.pth"
        if best_model_path.exists():
            try:
                self.model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=self.device))
            except (RuntimeError, EOFError) as e:
                print(f"Warning: Could not load best_model.pth (may be corrupted): {e}")
                print("Using final model state instead of best model...")

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
                
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for img_a,img_b, labels in self.val_loader:
                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(img_a,img_b)
                loss = self.eval_criterion(logits, labels)  # Use unsmoothed loss for fair comparison
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)
        return average_loss, accuracy
    def test(self, test_loader):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for img_a, img_b, labels in test_loader:
                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(img_a, img_b)
                loss = self.eval_criterion(logits, labels)  # Use unsmoothed loss for fair comparison
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(test_loader)
        return average_loss, accuracy


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    # Format learning rate, weight decay, and label smoothing for cleaner directory names
    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-")
    wd_str = f"{args.weight_decay:.0e}".replace("e-0", "e-") if args.weight_decay > 0 else "0"
    ls_str = f"ls={args.label_smoothing}" if args.label_smoothing > 0 else ""
    
    # Include run-id if provided (for multi-machine runs)
    run_id_suffix = f"_{args.run_id}" if args.run_id else ""
    # Add _aug suffix to distinguish augmented runs
    # Include label smoothing in name if used
    smoothing_part = f"_{ls_str}" if ls_str else ""
    tb_log_dir_prefix = f'opt={args.optimizer}_sched={args.scheduler}_bs={args.batch_size}_lr={lr_str}_wd={wd_str}{smoothing_part}_aug{run_id_suffix}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())