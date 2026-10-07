"""
Dataset download module using KaggleHub.
Downloads the GTSRB dataset and discovers its structure.
"""

import os
import kagglehub
from src.logger import get_logger

logger = get_logger()


def download_dataset():
    """Download the GTSRB dataset using KaggleHub.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    logger.info("=" * 60)
    logger.info("DOWNLOADING GTSRB DATASET")
    logger.info("=" * 60)
    
    # Download latest version using KaggleHub
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    
    logger.info(f"Dataset downloaded to: {path}")
    
    # Discover and log the directory structure
    _inspect_dataset_structure(path)
    
    return path


def _inspect_dataset_structure(path, max_depth=3):
    """Inspect and log the dataset directory structure.
    
    Args:
        path: Root path of the downloaded dataset
        max_depth: Maximum depth to traverse
    """
    logger.info("-" * 40)
    logger.info("Dataset structure:")
    
    for root, dirs, files in os.walk(path):
        # Calculate depth relative to the base path
        depth = root.replace(path, "").count(os.sep)
        if depth >= max_depth:
            continue
        
        indent = "  " * depth
        folder_name = os.path.basename(root) or os.path.basename(path)
        logger.info(f"{indent}{folder_name}/")
        
        # Log files (limit to first 5 per directory)
        sub_indent = "  " * (depth + 1)
        file_list = sorted(files)
        for f in file_list[:5]:
            logger.info(f"{sub_indent}{f}")
        if len(file_list) > 5:
            logger.info(f"{sub_indent}... and {len(file_list) - 5} more files")
    
    logger.info("-" * 40)


def find_dataset_paths(base_path):
    """Discover the actual train and test data paths within the dataset.
    
    The GTSRB dataset from Kaggle may have varying directory structures.
    This function discovers the actual paths dynamically.
    
    Args:
        base_path: Root path of the downloaded dataset
        
    Returns:
        dict: Dictionary with 'train_dir', 'test_dir', 'test_csv' paths
    """
    result = {
        "train_dir": None,
        "test_dir": None,
        "test_csv": None,
    }
    
    # Walk the directory to find Train and Test folders and CSV files
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            dir_lower = d.lower()
            full_path = os.path.join(root, d)
            if dir_lower == "train":
                result["train_dir"] = full_path
                logger.info(f"Found training directory: {full_path}")
            elif dir_lower in ("test", "meta-test"):
                # Only set if we haven't found one yet, prefer "Test"
                if result["test_dir"] is None or d.lower() == "test":
                    result["test_dir"] = full_path
                    logger.info(f"Found test directory: {full_path}")
        
        for f in files:
            if f.lower() in ("test.csv", "gt-final_test.csv"):
                result["test_csv"] = os.path.join(root, f)
                logger.info(f"Found test CSV: {result['test_csv']}")
    
    # Validate
    if result["train_dir"] is None:
        raise FileNotFoundError(
            f"Could not find training directory in {base_path}. "
            "Please check the dataset structure."
        )
    
    # Count training samples
    train_count = 0
    for root, dirs, files in os.walk(result["train_dir"]):
        train_count += len([f for f in files if f.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg'))])
    logger.info(f"Total training images found: {train_count}")
    
    if result["test_dir"]:
        test_count = 0
        for root, dirs, files in os.walk(result["test_dir"]):
            test_count += len([f for f in files if f.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg'))])
        logger.info(f"Total test images found: {test_count}")
    
    return result
