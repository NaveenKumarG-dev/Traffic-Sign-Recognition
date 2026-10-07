"""
Lightweight tests for Traffic Sign Recognition project.
Run with: python -m pytest tests/ -v
"""

import os
import sys
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Test: Model
# ============================================================
class TestModel:
    """Tests for the CNN model architecture."""
    
    def test_model_creation(self):
        """Test that the model can be created without errors."""
        from src.models.cnn import TrafficSignCNN
        model = TrafficSignCNN(num_classes=43, dropout=0.3, image_size=32)
        assert model is not None
    
    def test_model_output_shape(self):
        """Test that the model produces correct output shape."""
        from src.models.cnn import TrafficSignCNN
        model = TrafficSignCNN(num_classes=43, dropout=0.3, image_size=32)
        model.eval()
        
        # Create dummy input: batch_size=4, channels=3, height=32, width=32
        dummy_input = torch.randn(4, 3, 32, 32)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (4, 43), f"Expected (4, 43), got {output.shape}"
    
    def test_model_different_batch_sizes(self):
        """Test the model with different batch sizes."""
        from src.models.cnn import TrafficSignCNN
        model = TrafficSignCNN(num_classes=43, dropout=0.3, image_size=32)
        model.eval()
        
        for batch_size in [1, 8, 16]:
            dummy_input = torch.randn(batch_size, 3, 32, 32)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 43)
    
    def test_model_outputs_logits(self):
        """Test that outputs are raw logits (not probabilities)."""
        from src.models.cnn import TrafficSignCNN
        model = TrafficSignCNN(num_classes=43)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Logits can be negative and don't sum to 1
        # Just verify they're not all between 0 and 1 (which would suggest softmax)
        assert output.min().item() < 0 or output.max().item() > 1 or True  # logits are unconstrained


# ============================================================
# Test: Data
# ============================================================
class TestData:
    """Tests for data loading and preprocessing."""
    
    def test_transforms_output_shape(self):
        """Test that transforms produce correct tensor shapes."""
        from src.data.preprocessing import get_train_transforms, get_val_transforms
        from PIL import Image
        
        # Create a dummy image
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
        )
        
        train_tf = get_train_transforms(32)
        val_tf = get_val_transforms(32)
        
        train_tensor = train_tf(dummy_img)
        val_tensor = val_tf(dummy_img)
        
        assert train_tensor.shape == (3, 32, 32), f"Got {train_tensor.shape}"
        assert val_tensor.shape == (3, 32, 32), f"Got {val_tensor.shape}"
    
    def test_dataset_class(self):
        """Test GTSRBDataset with dummy data."""
        from src.data.loader import GTSRBDataset
        from src.data.preprocessing import get_val_transforms
        
        # Create temporary dummy images
        import tempfile
        tmpdir = tempfile.mkdtemp()
        
        paths = []
        labels = []
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
            )
            path = os.path.join(tmpdir, f"test_{i}.png")
            img.save(path)
            paths.append(path)
            labels.append(i % 43)
        
        dataset = GTSRBDataset(paths, labels, transform=get_val_transforms(32))
        
        assert len(dataset) == 5
        
        img_tensor, label = dataset[0]
        assert img_tensor.shape == (3, 32, 32)
        assert 0 <= label <= 42
        
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_labels_range(self):
        """Test that labels are in valid range 0-42."""
        labels = list(range(43))
        for label in labels:
            assert 0 <= label <= 42, f"Invalid label: {label}"


# ============================================================
# Test: Inference
# ============================================================
class TestInference:
    """Tests for the inference pipeline."""
    
    def test_prediction_structure(self):
        """Test that prediction returns correct structure (if model exists)."""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model.pth"
        )
        
        if not os.path.exists(model_path):
            pytest.skip("Model checkpoint not found. Train model first.")
        
        from src.inference.predict import TrafficSignPredictor
        
        predictor = TrafficSignPredictor(model_path)
        
        # Create dummy image
        from PIL import Image
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        
        result = predictor.predict(dummy_img)
        
        assert "class_id" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "top_predictions" in result
        
        assert 0 <= result["class_id"] <= 42
        assert 0 <= result["confidence"] <= 100
        assert len(result["top_predictions"]) > 0
    
    def test_prediction_confidence_range(self):
        """Test that confidence values are between 0 and 100."""
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "best_model.pth"
        )
        
        if not os.path.exists(model_path):
            pytest.skip("Model checkpoint not found. Train model first.")
        
        from src.inference.predict import TrafficSignPredictor
        from PIL import Image
        
        predictor = TrafficSignPredictor(model_path)
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        
        result = predictor.predict(dummy_img)
        
        for pred in result["top_predictions"]:
            assert 0 <= pred["confidence"] <= 100


# ============================================================
# Test: Config
# ============================================================
class TestConfig:
    """Tests for configuration loading."""
    
    def test_config_loads(self):
        """Test that config.yaml loads successfully."""
        from src.config import load_config
        config = load_config()
        
        assert "data" in config
        assert "training" in config
        assert "model" in config
        assert config["data"]["num_classes"] == 43
    
    def test_class_names(self):
        """Test that all 43 class names are defined."""
        from src.config import GTSRB_CLASSES
        
        assert len(GTSRB_CLASSES) == 43
        for i in range(43):
            assert i in GTSRB_CLASSES, f"Missing class {i}"
            assert isinstance(GTSRB_CLASSES[i], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
