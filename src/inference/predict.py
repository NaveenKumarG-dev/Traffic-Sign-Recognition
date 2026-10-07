"""
Inference module for GTSRB Traffic Sign Recognition.
Provides a clean predict() interface for single image classification.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import PROJECT_ROOT, GTSRB_CLASSES
from src.logger import get_logger
from src.models.cnn import TrafficSignCNN
from src.data.preprocessing import get_val_transforms

logger = get_logger()


class TrafficSignPredictor:
    """Encapsulates model loading and prediction for traffic sign images.
    
    Usage:
        predictor = TrafficSignPredictor()
        result = predictor.predict(image)
    """
    
    def __init__(self, checkpoint_path=None):
        """Initialize the predictor by loading the trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint.
                           Defaults to models/best_model.pth
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                "Please train the model first using: python main.py train"
            )
        
        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.image_size = checkpoint.get("image_size", 32)
        self.num_classes = checkpoint.get("num_classes", 43)
        self.class_names = checkpoint.get("class_names", GTSRB_CLASSES)
        
        # Reconstruct model
        self.model = TrafficSignCNN(
            num_classes=self.num_classes,
            dropout=checkpoint.get("dropout", 0.3),
            image_size=self.image_size,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform (same as validation)
        self.transform = get_val_transforms(self.image_size)
        
        logger.info(f"Model loaded successfully from: {checkpoint_path}")
        logger.info(f"Input size: {self.image_size}x{self.image_size}")
    
    def predict(self, image):
        """Predict the traffic sign class for a given image.
        
        Args:
            image: PIL Image, file path (str), or preprocessed tensor
            
        Returns:
            dict: Prediction result with keys:
                - class_id: Predicted class index (0-42)
                - class_name: Human-readable class name
                - confidence: Confidence percentage
                - top_predictions: List of top 5 predictions with
                                  (class_id, class_name, confidence)
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Already a tensor, skip transform
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            return self._predict_tensor(image)
        
        # Apply preprocessing
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return self._predict_tensor(img_tensor)
    
    def _predict_tensor(self, img_tensor):
        """Run prediction on a preprocessed tensor.
        
        Args:
            img_tensor: Preprocessed tensor of shape (1, 3, H, W)
            
        Returns:
            dict: Prediction result
        """
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, k=min(5, self.num_classes), dim=1)
        
        top5_prob = top5_prob.squeeze().cpu().numpy()
        top5_idx = top5_idx.squeeze().cpu().numpy()
        
        # Build result
        class_id = int(top5_idx[0])
        confidence = float(top5_prob[0]) * 100
        
        top_predictions = []
        for i in range(len(top5_idx)):
            pred_id = int(top5_idx[i])
            pred_conf = float(top5_prob[i]) * 100
            pred_name = self.class_names.get(pred_id, f"Class {pred_id}")
            top_predictions.append({
                "class_id": pred_id,
                "class_name": pred_name,
                "confidence": round(pred_conf, 2),
            })
        
        result = {
            "class_id": class_id,
            "class_name": self.class_names.get(class_id, f"Class {class_id}"),
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
        }
        
        logger.info(
            f"Prediction: {result['class_name']} "
            f"(ID: {result['class_id']}, Confidence: {result['confidence']:.2f}%)"
        )
        
        return result


def predict(image, checkpoint_path=None):
    """Convenience function for single image prediction.
    
    Args:
        image: PIL Image, file path, or tensor
        checkpoint_path: Optional path to checkpoint
        
    Returns:
        dict: Prediction result
    """
    predictor = TrafficSignPredictor(checkpoint_path)
    return predictor.predict(image)
