"""
Training module for GTSRB Traffic Sign Recognition.
Implements the complete PyTorch training loop with validation,
mixed precision training, and checkpointing.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

from src.config import PROJECT_ROOT, load_config, GTSRB_CLASSES
from src.logger import get_logger
from src.models.cnn import TrafficSignCNN

logger = get_logger()


def set_seeds(seed=42):
    """Set random seeds for reproducibility.
    
    Note: GPU operations may still have minor nondeterminism
    depending on the CUDA/PyTorch configuration.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Note: torch.backends.cudnn.deterministic = True can slow down training
    # For an assignment, we accept minor nondeterminism for speed
    logger.info(f"Random seeds set to {seed}")


def get_device():
    """Detect and return the best available device.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"Device: cuda")
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {vram:.1f} GB")
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is NOT available. Training will use CPU (much slower).")
        logger.info(f"Device: cpu")
    
    return device


def train_model(data_loaders, config=None):
    """Train the CNN model on GTSRB dataset.
    
    Implements:
        - Mixed precision training (AMP) for RTX 4050
        - Validation after each epoch
        - Best model checkpointing based on validation accuracy
        - Training history tracking
    
    Args:
        data_loaders: Dict with 'train' and 'val' DataLoaders
        config: Configuration dictionary
        
    Returns:
        dict: Training history with loss and accuracy curves
    """
    if config is None:
        config = load_config()
    
    # Setup
    seed = config["training"]["random_seed"]
    set_seeds(seed)
    device = get_device()
    
    # Model
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]
    image_size = config["data"]["image_size"]
    
    model = TrafficSignCNN(
        num_classes=num_classes,
        dropout=dropout,
        image_size=image_size,
    )
    model.to(device)
    
    # Log model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: TrafficSignCNN")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0001)
    use_amp = config["training"].get("use_amp", True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rates": [],
    }
    
    # Best model tracking
    best_val_acc = 0.0
    models_dir = os.path.join(PROJECT_ROOT, config["paths"]["models_dir"])
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pth")
    
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    
    logger.info("=" * 60)
    logger.info("TRAINING STARTED")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Mixed precision: {use_amp and device.type == 'cuda'}")
    logger.info("=" * 60)
    
    training_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        # ---- Validation Phase ----
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["learning_rates"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "class_names": GTSRB_CLASSES,
                "image_size": image_size,
                "num_classes": num_classes,
                "dropout": dropout,
                "config": config,
            }, best_model_path)
            logger.info(f"  ★ New best model saved! Val Acc: {val_acc:.2f}%")
    
    total_time = time.time() - training_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info("=" * 60)
    
    # Save training history
    metrics_dir = os.path.join(PROJECT_ROOT, config["paths"]["metrics_dir"])
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    return history


def validate(model, val_loader, criterion, device, use_amp=True):
    """Run validation on the validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: torch.device
        use_amp: Whether to use mixed precision
        
    Returns:
        tuple: (val_loss, val_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc
