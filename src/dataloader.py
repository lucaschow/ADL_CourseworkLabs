import os
import random
import re
import torch
from torch.utils.data import Dataset
from PIL import Image

class ProgressionDataset(Dataset):
    """
    A PyTorch `Dataset` class for the Recipe Progress Classification task.

    This dataset supports three modes of operation:
    - **'train'**: Dynamically generates image pairs from recipe folders, 
      creating three types of pairs:
        * forward (label = 0): (i,j) where j > 1
        * reverse (label = 1): (i,j) where j < 1)
        * unrelated (label = 2): steps from different recipes
    - **'val'** / **'test'**: Loads pre-defined pairs and their labels 
      from a given label file.

    Attributes
    ----------
    root_dir : str
        Path to the directory containing recipe image folders.
    transform : callable, optional
        A torchvision-compatible transform applied to each image.
    mode : str
        One of {'train', 'val', 'test'} indicating dataset behavior.
    recipes : dict
        Dictionary mapping recipe IDs to lists of image file paths (used in 'train' mode).
    fixed_pairs : list
        List of (img_a_path, img_b_path, label) tuples loaded from a label file 
        (used in 'val'/'test' mode).
    epoch_size : int
        Number of samples per epoch in training mode.
    """

    def __init__(self, root_dir, transform=None, mode='train', 
                 recipe_ids_list=None, epoch_size=None,
                 label_file=None):
        """
        Initialize the ProgressionDataset.

        Parameters
        ----------
        root_dir : str
            Root directory containing recipe folders or image pairs.
        transform : callable, optional
            Optional torchvision transform for image preprocessing.
        mode : str, default='train'
            Operation mode: 'train', 'val', or 'test'.
        recipe_ids_list : list of str, optional
            List of recipe folder names (required for 'train' mode).
        epoch_size : int, optional
            Number of samples per epoch (required for 'train' mode).
        label_file : str, optional
            Path to text file containing image pair indices and labels 
            (required for 'val'/'test' mode).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        print(f"Initializing dataset in '{self.mode}' mode.")

        # ---------------------------
        # TRAINING MODE
        # ---------------------------
        if self.mode == 'train':
            if recipe_ids_list is None or epoch_size is None:
                raise ValueError("In 'train' mode, 'recipe_ids_list' and 'epoch_size' must be provided.")
            
            self.epoch_size = epoch_size
            self.recipes = {}
            # Iterate through recipe folders and collect ordered step images
            for recipe_id in recipe_ids_list:
                recipe_path = os.path.join(self.root_dir, recipe_id)
                image_files = [f for f in os.listdir(recipe_path) if f.lower().endswith('.jpg')]
                
                def get_step_number(filename):
                    match = re.search(r'_S(\d+)\.jpg', filename, re.IGNORECASE)
                    return int(match.group(1)) if match else -1
                
                steps = sorted(image_files, key=get_step_number)
                # Only include recipes with more than one step
                if len(steps) > 1:
                    self.recipes[recipe_id] = [os.path.join(recipe_path, s) for s in steps]

            self.recipe_ids = list(self.recipes.keys())
            if len(self.recipe_ids) < 2:
                 raise ValueError("Training mode needs at least two recipes for 'unrelated' pairs.")
            print(f"Found {len(self.recipe_ids)} recipes for training.")

        # ---------------------------
        # VALIDATION / TEST MODE
        # ---------------------------
        elif self.mode in ['val', 'test']:
            if label_file is None:
                raise ValueError(f"In '{self.mode}' mode, 'label_file' must be provided.")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")

            # Load pre-defined image pairs and labels from text file
            self.fixed_pairs = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        index, label_str = parts
                        label = int(label_str)
                        
                        # Construct expected file paths for each pair
                        img_a_path = os.path.join(self.root_dir, f"{index}_1.jpg")
                        img_b_path = os.path.join(self.root_dir, f"{index}_2.jpg")
                        
                        if os.path.exists(img_a_path) and os.path.exists(img_b_path):
                            self.fixed_pairs.append((img_a_path, img_b_path, label))
                        else:
                            print(f"Warning: Could not find pair for index {index}. Skipping.")
            
            print(f"Loaded {len(self.fixed_pairs)} fixed pairs for {self.mode}.")

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'train', 'val', 'test'.")

    def _generate_pair(self):
        """
        Randomly generate a training pair and its label.

        Returns
        -------
        tuple
            (img_a_path, img_b_path, label)
            where:
            - label = 0 for forward pairs (i,j) where j > 1
            - label = 1 for reverse pairs (i,j) where j < 1)
            - label = 2 for unrelated pairs (different recipes)
        """
        # 0: forward, 1: reverse, 2: unrelated
        pair_type = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2], k=1)[0]

        if pair_type == 2:  # Unrelated pair (from different recipes)
            recipe_id_a, recipe_id_b = random.sample(self.recipe_ids, 2)
            img_a_path = random.choice(self.recipes[recipe_id_a])
            img_b_path = random.choice(self.recipes[recipe_id_b])
            label = 2

        else:  # Forward or reverse pair (from same recipe)
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

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            `epoch_size` in training mode, or number of fixed pairs otherwise.
        """
        return self.epoch_size if self.mode == 'train' else len(self.fixed_pairs)

    def __getitem__(self, idx=None):
        """
        Retrieve a dataset sample at the given index.

        Parameters
        ----------
        idx : int, optional
            Sample index (used only in 'val'/'test' mode), defaults to None
            assuming a 'train' mode.

        Returns
        -------
        tuple
            (img_a, img_b, label)
            where `img_a` and `img_b` are transformed tensors,
            and `label` is a torch.LongTensor indicating the pair type.
        """
        if self.mode == 'train':
            img_a_path, img_b_path, label = self._generate_pair()
        else:
            if idx is None:
                raise ValueError("In 'val'/'test' mode, 'idx' must be provided.")
            img_a_path, img_b_path, label = self.fixed_pairs[idx]

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            
        return img_a, img_b, torch.tensor(label, dtype=torch.long)
