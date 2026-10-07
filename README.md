# Traffic Sign Recognition for Driver Assistance Systems

Build a clean, professional, end-to-end machine learning project using the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## Project Overview

This project implements a complete Machine Learning lifecycle for classifying traffic signs from the GTSRB dataset. It features an end-to-end pipeline from data downloading and Exploratory Data Analysis (EDA) to building a Convolutional Neural Network (CNN) in PyTorch, training with CUDA acceleration, evaluating the model, and serving it via a Streamlit web application.

## Problem Statement

Driver assistance systems (and autonomous vehicles) require reliable and fast identification of traffic signs to make safe driving decisions. Accurately recognizing signs under varying lighting conditions, angles, and occlusions is a critical computer vision task.

## Objective

The objective is to build a professional, easily reproducible PyTorch CNN classifier capable of categorizing images into one of 43 traffic sign classes using a lightweight architecture suitable for inference on standard hardware.

## Dataset

- **Name**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Classes**: 43 distinct traffic sign categories
- **Format**: RGB images of varying sizes (standardized to 32x32 in this project)
- **Source**: Kaggle

## Technology Stack

- **Language**: Python
- **Deep Learning Framework**: PyTorch (with CUDA support)
- **Data Manipulation**: NumPy, Pandas
- **Image Processing**: Pillow, torchvision
- **Machine Learning Utilities**: scikit-learn
- **Visualization**: Matplotlib
- **Configuration**: PyYAML
- **Web Interface**: Streamlit

## Project Architecture

```text
traffic-sign-recognition/
├── app/                  # Streamlit application
├── artifacts/            # Generated files (logs, metrics, plots)
├── configs/              # YAML configuration files
├── data/                 # Raw and processed data
├── models/               # Saved PyTorch checkpoints
├── notebooks/            # Jupyter notebooks for interactive EDA
├── src/                  # Core source code
│   ├── data/             # Data loading and preprocessing
│   ├── evaluation/       # Evaluation metrics and reporting
│   ├── inference/        # Single-image prediction pipeline
│   ├── models/           # PyTorch CNN definition
│   └── training/         # Training loop and checkpointing
├── tests/                # Lightweight pytest suite
├── main.py               # CLI entry point
└── requirements.txt      # Project dependencies
```

## ML Pipeline

1. **Dataset Download**: Automatically fetches the dataset using `kagglehub`.
2. **Dataset Inspection & EDA**: Discovers directory structure, plots class distributions, and samples images.
3. **Preprocessing**: Resizes images to 32x32, applies conservative augmentation (rotation, translation, scaling, color jitter) to training data, and normalizes tensors.
4. **CNN Model**: A lightweight 3-block Convolutional Neural Network optimized for standard GPUs.
5. **Training**: PyTorch training loop with Adam optimizer, ReduceLROnPlateau scheduler, Mixed Precision Training (AMP), and validation-based checkpointing.
6. **Evaluation**: Computes Accuracy, Precision, Recall, F1-Score, and generates a confusion matrix on the test set.
7. **Inference & App**: A clean `predict()` interface served via a responsive Streamlit web app.

## CNN Architecture

The model uses a straightforward, lightweight architecture designed for efficiency on a 6 GB VRAM GPU (like the RTX 4050):

- **Block 1**: Conv2D(3→32) → BatchNorm → ReLU → MaxPool
- **Block 2**: Conv2D(32→64) → BatchNorm → ReLU → MaxPool
- **Block 3**: Conv2D(64→128) → BatchNorm → ReLU → MaxPool
- **Classifier**: Flatten → Linear(256) → ReLU → Dropout(0.3) → Linear(43)

## GPU/CUDA Setup

This project is optimized for:
- **OS**: Windows
- **GPU**: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)
- **Framework**: PyTorch with CUDA support

### Installing PyTorch with CUDA (Required)

Because PyTorch CUDA wheels can be specific to your environment, they are not strictly versioned in `requirements.txt`. Instead, install PyTorch separately before the other dependencies.

For Windows with Python 3.10+ (using CUDA 12.6 as an example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify your installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Installation

1. Create and activate a virtual environment (do not use Conda as per requirements):
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install PyTorch with CUDA (see above).

3. Install the remaining project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Commands

The project provides a clean CLI interface via `main.py`.

### 1. Dataset Download
Downloads the dataset and discovers the directory structure:
```bash
python main.py download
```

### 2. Exploratory Data Analysis (EDA)
Generates `class_distribution.png` and `sample_images.png` in the `artifacts/plots/` directory:
```bash
python main.py eda
```

### 3. Training
Trains the CNN model and saves the best checkpoint to `models/best_model.pth`:
```bash
python main.py train
```

### 4. Evaluation
Evaluates the best model on the test set and generates a confusion matrix:
```bash
python main.py evaluate
```

### 5. Run Entire Pipeline
Runs download, EDA, training, and evaluation sequentially:
```bash
python main.py all
```

## Streamlit Application

To start the interactive web application for real-time traffic sign classification:

```bash
streamlit run app/streamlit_app.py
```

Upload any PNG/JPG image to receive an instant prediction along with confidence scores and top-5 alternatives.

## Results

*(Results will be populated in `artifacts/metrics/test_metrics.json` after running the training pipeline.)*

- Check `artifacts/plots/training_history.png` for loss and accuracy curves.
- Check `artifacts/plots/confusion_matrix.png` for per-class performance details.
