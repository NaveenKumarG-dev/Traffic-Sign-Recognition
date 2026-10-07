"""
Exploratory Data Analysis for GTSRB dataset.
Generates class distribution and sample image plots.
"""

import os
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, load_config, GTSRB_CLASSES
from src.logger import get_logger

logger = get_logger()


def run_eda(dataset_paths, config=None):
    """Run exploratory data analysis on the GTSRB dataset.
    
    Generates:
        - class_distribution.png
        - sample_images.png
    
    Args:
        dataset_paths: Dict with 'train_dir' key
        config: Configuration dictionary
    """
    if config is None:
        config = load_config()
    
    train_dir = dataset_paths["train_dir"]
    plots_dir = os.path.join(PROJECT_ROOT, config["paths"]["plots_dir"])
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    # Collect all image paths and labels
    image_paths_by_class = {}
    all_labels = []
    sample_dims = []
    
    valid_ext = ('.png', '.ppm', '.jpg', '.jpeg')
    
    for class_dir in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        try:
            class_id = int(class_dir)
        except ValueError:
            continue
        
        class_images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(valid_ext):
                class_images.append(os.path.join(class_path, img_file))
                all_labels.append(class_id)
        
        image_paths_by_class[class_id] = class_images
    
    # Basic statistics
    total_images = len(all_labels)
    num_classes = len(image_paths_by_class)
    class_counts = Counter(all_labels)
    
    logger.info(f"Total images: {total_images}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Min class size: {min(class_counts.values())} (Class {min(class_counts, key=class_counts.get)})")
    logger.info(f"Max class size: {max(class_counts.values())} (Class {max(class_counts, key=class_counts.get)})")
    logger.info(f"Mean class size: {np.mean(list(class_counts.values())):.1f}")
    
    # Sample image dimensions
    for class_id in list(image_paths_by_class.keys())[:10]:
        paths = image_paths_by_class[class_id]
        if paths:
            img = Image.open(paths[0])
            sample_dims.append(img.size)
    
    if sample_dims:
        widths = [d[0] for d in sample_dims]
        heights = [d[1] for d in sample_dims]
        logger.info(f"Sample image dimensions: width {min(widths)}-{max(widths)}, height {min(heights)}-{max(heights)}")
    
    # Plot 1: Class distribution
    _plot_class_distribution(class_counts, num_classes, plots_dir)
    
    # Plot 2: Sample images from different classes
    _plot_sample_images(image_paths_by_class, plots_dir)
    
    logger.info("EDA completed. Plots saved to: " + plots_dir)


def _plot_class_distribution(class_counts, num_classes, plots_dir):
    """Plot class distribution bar chart.
    
    Args:
        class_counts: Counter of class frequencies
        num_classes: Total number of classes
        plots_dir: Directory to save the plot
    """
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(classes)))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.5)
    
    ax.set_xlabel("Class ID", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("GTSRB Dataset - Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, num_classes, 2))
    ax.set_xticklabels(range(0, num_classes, 2), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        if count > 100:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                str(count), ha="center", va="bottom", fontsize=5, rotation=90,
            )
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Class distribution plot saved to: {save_path}")


def _plot_sample_images(image_paths_by_class, plots_dir, num_classes_to_show=15):
    """Plot sample images from multiple classes.
    
    Args:
        image_paths_by_class: Dict mapping class_id -> list of image paths
        plots_dir: Directory to save the plot
        num_classes_to_show: Number of classes to display
    """
    # Pick evenly spaced classes
    all_classes = sorted(image_paths_by_class.keys())
    step = max(1, len(all_classes) // num_classes_to_show)
    selected_classes = all_classes[::step][:num_classes_to_show]
    
    cols = 5
    rows = (len(selected_classes) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows * cols == 1 else axes.flatten()
    
    for idx, class_id in enumerate(selected_classes):
        ax = axes[idx]
        paths = image_paths_by_class[class_id]
        if paths:
            img = Image.open(paths[0]).convert("RGB")
            ax.imshow(img)
        
        class_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")
        # Truncate long names
        display_name = class_name[:25] + "..." if len(class_name) > 25 else class_name
        ax.set_title(f"[{class_id}] {display_name}", fontsize=8)
        ax.axis("off")
    
    # Hide unused axes
    for idx in range(len(selected_classes), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("GTSRB - Sample Traffic Signs by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, "sample_images.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sample images plot saved to: {save_path}")
