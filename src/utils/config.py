"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str = "yolov8n.pt"
    input_size: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"

@dataclass
class TrackingConfig:
    """Tracking configuration."""
    max_disappeared: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    max_distance: float = 100.0

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    epochs: int = 100
    num_workers: int = 4

@dataclass
class PerformanceConfig:
    """Performance configuration."""
    target_fps: int = 30
    max_latency_ms: int = 50
    memory_limit_gb: int = 8

class ConfigManager:
    """Configuration manager for the perception system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/detection_config.yaml"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            # Return default configuration
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults
        default_config = self._get_default_config()
        return self._merge_configs(default_config, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "architecture": "yolov8n.pt",
                "input_size": 640,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "auto"
            },
            "tracking": {
                "max_disappeared": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "max_distance": 100.0
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0005,
                "epochs": 100,
                "num_workers": 4
            },
            "performance": {
                "target_fps": 30,
                "max_latency_ms": 50,
                "memory_limit_gb": 8
            }
        }
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Merge custom config with default config."""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self.config.get("model", {})
        return ModelConfig(**model_config)
    
    def get_tracking_config(self) -> TrackingConfig:
        """Get tracking configuration."""
        tracking_config = self.config.get("tracking", {})
        return TrackingConfig(**tracking_config)
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        training_config = self.config.get("training", {})
        return TrainingConfig(**training_config)
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        performance_config = self.config.get("performance", {})
        return PerformanceConfig(**performance_config)
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        output_file = Path(output_path or self.config_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)