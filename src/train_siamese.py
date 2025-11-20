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
    default=1e-3, 
    type=float, 
    help="Base Adam Learning rate"
)
parser.add_argument(
    "--batch-size",
    default=8,
    type=int,
    help="Number of pairs within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--epoch-size",
    default=5000,
    type=int,
    help="number of pairs generated per epoch in training mode",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
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


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    epoch_size = 5000 #change to 5000
    recepie_ids_list = ["P01_R01_instance_0","P01_R01_instance_1","P01_R01_instance_2","P01_R03","P01_R04","P01_R05","P01_R07","P01_R09","P02_R01","P02_R02","P02_R03_instance_0","P02_R03_instance_1","P02_R03_instance_2","P02_R03_instance_3","P02_R05","P02_R07","P02_R08","P02_R10","P02_R11","P03_R01","P03_R02","P03_R03_instance_0","P03_R03_instance_2","P03_R03_instance_3","P03_R04","P03_R05","P03_R06","P03_R07","P03_R08","P03_R09","P03_R10","P04_R01","P04_R02","P04_R03","P04_R04","P04_R05","P04_R06","P05_R01","P05_R02_instance_0","P05_R02_instance_2","P05_R03","P05_R04","P05_R05","P05_R06","P07_R01","P07_R02","P07_R03","P07_R04","P07_R05","P07_R06","P07_R07","P08_R01","P08_R02","P08_R04","P08_R05","P08_R06","P08_R07","P08_R08","P08_R09","P09_R01","P09_R02","P09_R03_instance_0","P09_R03_instance_1","P09_R04","P09_R05","P09_R06"]

    
    train_dataset = ProgressionDataset(root_dir='dataset/train', transform=transform, mode='train', recipe_ids_list=recepie_ids_list, epoch_size=epoch_size)
    #you didnt load in the data correctly you melon - we only have train
    test_dataset = ProgressionDataset(root_dir='dataset/test', transform=transform, mode='test', label_file='dataset/test_labels.txt')
    val_dataset = ProgressionDataset(root_dir='dataset/val', transform=transform, mode='val', label_file='dataset/val_labels.txt')
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

    model = Siamese(in_channels=3) #was CNN

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

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
    def __init__(self, in_channels=3):
        super().__init__()
        self.branch = Branch(channels=in_channels) #initialise once so we get shared weights
        self.fc1 =nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,3)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

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



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

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
            print("Starting training loop...")
            data_load_start_time = time.time()
            for img_a, img_b, labels in self.train_loader:
                print(f"Loaded first batch, shape: {img_a.shape}, {img_b.shape}")
                img_a = img_a.to(self.device)
                img_b = img_b.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                logits =self.model(img_a, img_b)
                print("Starting forward pass...")
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                ## TASK 10: Compute the backward pass

                ## TASK 12: Step the optimizer and then zero out the gradient buffers.

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

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
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


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
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())