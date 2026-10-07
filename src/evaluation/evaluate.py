"""
Evaluation module for GTSRB Traffic Sign Recognition.
Loads the best checkpoint and evaluates on the test set.
Generates confusion matrix, classification report, and saves metrics.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import PROJECT_ROOT, load_config, GTSRB_CLASSES
from src.logger import get_logger
from src.models.cnn import TrafficSignCNN

logger = get_logger()


def load_checkpoint(checkpoint_path=None):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file.
                        If None, loads from models/best_model.pth
    
    Returns:
        tuple: (model, checkpoint_dict, device)
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct model
    num_classes = checkpoint.get("num_classes", 43)
    dropout = checkpoint.get("dropout", 0.3)
    image_size = checkpoint.get("image_size", 32)
    
    model = TrafficSignCNN(
        num_classes=num_classes,
        dropout=dropout,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Checkpoint val accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    return model, checkpoint, device


def evaluate_model(data_loaders, checkpoint_path=None, config=None):
    """Evaluate the model on test/validation data.
    
    Args:
        data_loaders: Dict with DataLoaders ('test' and/or 'val')
        checkpoint_path: Path to checkpoint. If None, uses default.
        config: Configuration dictionary
        
    Returns:
        dict: Evaluation metrics
    """
    if config is None:
        config = load_config()
    
    model, checkpoint, device = load_checkpoint(checkpoint_path)
    use_amp = config["training"].get("use_amp", True)
    
    # Use test loader if available, otherwise use validation loader
    eval_loader = data_loaders.get("test") or data_loaders.get("val")
    eval_name = "Test" if data_loaders.get("test") else "Validation"
    
    if eval_loader is None:
        logger.error("No evaluation data available!")
        return {}
    
    logger.info("=" * 60)
    logger.info(f"EVALUATING ON {eval_name.upper()} SET")
    logger.info("=" * 60)
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs.float(), dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = running_loss / len(all_labels)
    
    metrics = {
        "eval_set": eval_name.lower(),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "loss": round(avg_loss, 4),
        "num_samples": len(all_labels),
        "num_classes": len(set(all_labels)),
    }
    
    logger.info(f"{eval_name} Results:")
    logger.info(f"  Accuracy:  {accuracy:.2f}%")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  Loss:      {avg_loss:.4f}")
    
    # Save metrics
    metrics_dir = os.path.join(PROJECT_ROOT, config["paths"]["metrics_dir"])
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Generate confusion matrix plot
    plots_dir = os.path.join(PROJECT_ROOT, config["paths"]["plots_dir"])
    os.makedirs(plots_dir, exist_ok=True)
    _plot_confusion_matrix(all_labels, all_preds, plots_dir)
    
    # Print classification report
    unique_labels = sorted(set(all_labels) | set(all_preds))
    target_names = [GTSRB_CLASSES.get(i, f"Class {i}") for i in unique_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save classification report
    report_path = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    return metrics


def _plot_confusion_matrix(y_true, y_pred, plots_dir):
    """Generate and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        plots_dir: Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix - GTSRB Traffic Sign Classification", fontsize=14)
    
    # Add tick labels for every 5th class to avoid clutter
    tick_marks = list(range(0, cm.shape[0], 5))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks, fontsize=8)
    ax.set_yticklabels(tick_marks, fontsize=8)
    
    plt.tight_layout()
    
    save_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_training_history(config=None):
    """Plot training history from saved JSON.
    
    Args:
        config: Configuration dictionary
    """
    if config is None:
        config = load_config()
    
    metrics_dir = os.path.join(PROJECT_ROOT, config["paths"]["metrics_dir"])
    history_path = os.path.join(metrics_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        logger.warning(f"Training history not found: {history_path}")
        return
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history["train_accuracy"], "b-o", markersize=4, label="Training Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], "r-o", markersize=4, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("GTSRB Traffic Sign Recognition - Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plots_dir = os.path.join(PROJECT_ROOT, config["paths"]["plots_dir"])
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved to: {save_path}")
