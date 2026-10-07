"""
Image preprocessing and augmentation pipelines for GTSRB dataset.
Defines separate transforms for training and validation/test.
"""

from torchvision import transforms


def get_train_transforms(image_size=32):
    """Get training image transformations with data augmentation.
    
    Augmentations are kept conservative to avoid changing the meaning
    of traffic signs (e.g., no horizontal flip since that changes sign meaning).
    
    Args:
        image_size: Target image size (square)
        
    Returns:
        torchvision.transforms.Compose: Training transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],  # GTSRB dataset statistics
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])


def get_val_transforms(image_size=32):
    """Get validation/test image transformations (no augmentation).
    
    Args:
        image_size: Target image size (square)
        
    Returns:
        torchvision.transforms.Compose: Validation transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3337, 0.3064, 0.3171],
            std=[0.2672, 0.2564, 0.2629],
        ),
    ])


def get_inverse_normalize():
    """Get inverse normalization transform for visualization.
    
    Returns:
        torchvision.transforms.Normalize: Inverse normalization
    """
    return transforms.Normalize(
        mean=[-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629],
        std=[1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629],
    )
