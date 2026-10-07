"""
Traffic Sign Recognition - Main Entry Point
============================================
GTSRB-based PyTorch CNN Classifier for Driver Assistance Systems

Usage:
    python main.py download    - Download GTSRB dataset
    python main.py eda         - Run exploratory data analysis
    python main.py train       - Train the CNN model
    python main.py evaluate    - Evaluate on test set
    python main.py predict <image_path>  - Predict single image
    python main.py all         - Run full pipeline (download → train → evaluate)
"""

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.logger import setup_logger


def cmd_download():
    """Download the GTSRB dataset."""
    from src.data.download import download_dataset, find_dataset_paths
    
    path = download_dataset()
    dataset_paths = find_dataset_paths(path)
    return dataset_paths


def cmd_eda():
    """Run exploratory data analysis."""
    from src.data.download import download_dataset, find_dataset_paths
    from src.data.eda import run_eda
    
    path = download_dataset()
    dataset_paths = find_dataset_paths(path)
    run_eda(dataset_paths)


def cmd_train():
    """Train the model."""
    from src.data.download import download_dataset, find_dataset_paths
    from src.data.loader import create_data_loaders
    from src.training.train import train_model
    from src.evaluation.evaluate import plot_training_history
    
    config = load_config()
    
    path = download_dataset()
    dataset_paths = find_dataset_paths(path)
    data_loaders = create_data_loaders(dataset_paths, config)
    
    history = train_model(data_loaders, config)
    plot_training_history(config)
    
    return history


def cmd_evaluate():
    """Evaluate the trained model."""
    from src.data.download import download_dataset, find_dataset_paths
    from src.data.loader import create_data_loaders
    from src.evaluation.evaluate import evaluate_model, plot_training_history
    
    config = load_config()
    
    path = download_dataset()
    dataset_paths = find_dataset_paths(path)
    data_loaders = create_data_loaders(dataset_paths, config)
    
    metrics = evaluate_model(data_loaders, config=config)
    plot_training_history(config)
    
    return metrics


def cmd_predict(image_path):
    """Run inference on a single image."""
    from src.inference.predict import predict
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    result = predict(image_path)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Traffic Sign: {result['class_name']}")
    print(f"Confidence:   {result['confidence']:.2f}%")
    print(f"Class ID:     {result['class_id']}")
    print("\nTop Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"  {i}. {pred['class_name']} — {pred['confidence']:.2f}%")
    print("=" * 50)


def cmd_all():
    """Run the complete pipeline."""
    logger = setup_logger()
    logger.info("Running complete ML pipeline...")
    
    # Download
    dataset_paths = cmd_download()
    
    # EDA
    from src.data.eda import run_eda
    run_eda(dataset_paths)
    
    # Train
    cmd_train()
    
    # Evaluate
    cmd_evaluate()
    
    logger.info("Complete pipeline finished successfully!")


def main():
    """Parse command-line arguments and dispatch."""
    logger = setup_logger()
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "download":
        cmd_download()
    elif command == "eda":
        cmd_eda()
    elif command == "train":
        cmd_train()
    elif command == "evaluate":
        cmd_evaluate()
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image_path>")
            sys.exit(1)
        cmd_predict(sys.argv[2])
    elif command == "all":
        cmd_all()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
