"""Automotive-specific metrics for model evaluation."""

import numpy as np
from typing import Dict, List
from sklearn.metrics import average_precision_score


class AutomotiveMetrics:
    """Metrics calculator for automotive perception models."""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.safety_critical_classes = ['person', 'bicycle', 'motorcycle']
        self.vehicle_classes = ['car', 'truck', 'bus']
        self.infrastructure_classes = ['traffic_light', 'stop_sign']
    
    def calculate_automotive_ap(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate automotive-specific Average Precision metrics."""
        
        # Standard mAP calculation
        ap_scores = []
        for i in range(len(self.class_names)):
            if len(np.unique(y_true[:, i])) > 1:  # Check if class exists
                ap = average_precision_score(y_true[:, i], y_scores[:, i])
                ap_scores.append(ap)
        
        metrics = {
            'mAP': np.mean(ap_scores) if ap_scores else 0.0,
            'safety_critical_mAP': self._calculate_category_map(ap_scores, self.safety_critical_classes),
            'vehicle_mAP': self._calculate_category_map(ap_scores, self.vehicle_classes),
            'infrastructure_mAP': self._calculate_category_map(ap_scores, self.infrastructure_classes)
        }
        
        return metrics
    
    def _calculate_category_map(self, ap_scores: List[float], category_classes: List[str]) -> float:
        """Calculate mAP for specific category of classes."""
        category_indices = [i for i, name in enumerate(self.class_names) if name in category_classes]
        if not category_indices or not ap_scores:
            return 0.0
        
        category_aps = [ap_scores[i] for i in category_indices if i < len(ap_scores)]
        return np.mean(category_aps) if category_aps else 0.0
    
    def calculate_distance_accuracy(self, pred_distances: np.ndarray, true_distances: np.ndarray) -> Dict[str, float]:
        """Calculate distance estimation accuracy metrics."""
        
        # Mean Absolute Error
        mae = np.mean(np.abs(pred_distances - true_distances))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((pred_distances - true_distances) ** 2))
        
        # Percentage within threshold
        threshold_1m = np.mean(np.abs(pred_distances - true_distances) < 1.0) * 100
        threshold_2m = np.mean(np.abs(pred_distances - true_distances) < 2.0) * 100
        
        return {
            'distance_mae': mae,
            'distance_rmse': rmse,
            'accuracy_1m': threshold_1m,
            'accuracy_2m': threshold_2m
        }