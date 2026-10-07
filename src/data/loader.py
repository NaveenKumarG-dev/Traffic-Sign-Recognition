"""
PyTorch Dataset and DataLoader for GTSRB.
Implements custom Dataset class and provides train/val/test DataLoader creation.
"""

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from src.logger import get_logger
from src.config import load_config
from src.data.preprocessing import get_train_transforms, get_val_transforms

logger = get_logger()


class GTSRBDataset(Dataset):
    """Custom PyTorch Dataset for GTSRB traffic sign images.
    
    Attributes:
        image_paths: List of image file paths
        labels: List of integer class labels (0-42)
        transform: torchvision transform pipeline
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding class labels
            transform: Optional torchvision transform
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load and return a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def collect_train_samples(train_dir):
    """Collect all training image paths and labels from the directory structure.
    
    The GTSRB training data is organized as:
        Train/
            0/  (class 0 images)
            1/  (class 1 images)
            ...
            42/ (class 42 images)
    
    Args:
        train_dir: Path to the training directory
        
    Returns:
        tuple: (list of image paths, list of labels)
    """
    image_paths = []
    labels = []
    
    # Each subdirectory name is the class ID
    valid_extensions = ('.png', '.ppm', '.jpg', '.jpeg')
    
    for class_dir in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        try:
            class_id = int(class_dir)
        except ValueError:
            continue
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(class_path, img_file))
                labels.append(class_id)
    
    logger.info(f"Collected {len(image_paths)} training images across {len(set(labels))} classes")
    return image_paths, labels


def collect_test_samples(test_dir, test_csv):
    """Collect test image paths and labels from the test directory and CSV.
    
    Args:
        test_dir: Path to the test images directory
        test_csv: Path to the test CSV with labels
        
    Returns:
        tuple: (list of image paths, list of labels)
    """
    image_paths = []
    labels = []
    
    if test_csv and os.path.exists(test_csv):
        try:
            # Kaggle Test.csv uses commas, some others use ';'
            df = pd.read_csv(test_csv, sep=None, engine='python')
        except Exception as e:
            logger.error(f"Failed to read test CSV {test_csv}: {e}")
            return image_paths, labels
            
        # Handle different CSV column name conventions
        path_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ("path", "filename", "file", "image"):
                path_col = col
            elif col_lower in ("classid", "class", "label", "classlabel"):
                label_col = col
        
        if path_col is None or label_col is None:
            # Fallback: assume last column is label, first or 'Path' column is path
            if "Path" in df.columns:
                path_col = "Path"
            else:
                path_col = df.columns[0]
            
            if "ClassId" in df.columns:
                label_col = "ClassId"
            else:
                label_col = df.columns[-1]
        
        for _, row in df.iterrows():
            img_filename = str(row[path_col]).strip()
            label = int(row[label_col])
            
            # Build full path - the CSV might have relative paths
            if os.path.isabs(img_filename):
                img_path = img_filename
            else:
                # Try relative to test_dir first, then relative to CSV directory
                img_path = os.path.join(test_dir, img_filename)
                if not os.path.exists(img_path):
                    csv_dir = os.path.dirname(test_csv)
                    img_path = os.path.join(csv_dir, img_filename)
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(label)
        
        logger.info(f"Collected {len(image_paths)} test images from CSV")
    else:
        logger.warning("No test CSV found. Test evaluation will be skipped.")
    
    return image_paths, labels


def create_data_loaders(dataset_paths, config=None):
    """Create train, validation, and test DataLoaders.
    
    Performs stratified train/val split on the training data.
    
    Args:
        dataset_paths: Dict with 'train_dir', 'test_dir', 'test_csv' keys
        config: Configuration dictionary. If None, loads from config.yaml.
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders and metadata
    """
    if config is None:
        config = load_config()
    
    image_size = config["data"]["image_size"]
    val_split = config["data"]["validation_split"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    pin_memory = config["training"]["pin_memory"]
    seed = config["training"]["random_seed"]
    
    # Collect training samples
    all_paths, all_labels = collect_train_samples(dataset_paths["train_dir"])
    
    # Stratified train/validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=val_split,
        random_state=seed,
        stratify=all_labels,
    )
    
    logger.info(f"Train/Val split: {len(train_paths)} train, {len(val_paths)} validation")
    
    # Create transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = GTSRBDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = GTSRBDataset(val_paths, val_labels, transform=val_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    result = {
        "train": train_loader,
        "val": val_loader,
        "test": None,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "num_test": 0,
    }
    
    # Create test DataLoader if test data exists
    if dataset_paths.get("test_dir") and dataset_paths.get("test_csv"):
        test_paths, test_labels = collect_test_samples(
            dataset_paths["test_dir"], dataset_paths["test_csv"]
        )
        if test_paths:
            test_dataset = GTSRBDataset(test_paths, test_labels, transform=val_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
            result["test"] = test_loader
            result["test_labels"] = test_labels
            result["num_test"] = len(test_paths)
            logger.info(f"Test set: {len(test_paths)} images")
    
    return result
