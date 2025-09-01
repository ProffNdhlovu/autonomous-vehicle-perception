"""Advanced visualization utilities for detection and tracking results."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

class AdvancedVisualizer:
    """Advanced visualizer with trajectory plotting and performance metrics."""
    
    def __init__(self):
        self.colors = {
            'person': (255, 0, 0),
            'bicycle': (0, 255, 0),
            'car': (0, 0, 255),
            'motorcycle': (255, 255, 0),
            'bus': (255, 0, 255),
            'truck': (0, 255, 255),
            'traffic_light': (128, 0, 128),
            'stop_sign': (255, 165, 0)
        }
        
        self.class_names = list(self.colors.keys())
        self.trajectory_history = {}
    
    def draw_detections_with_metrics(
        self, 
        image: np.ndarray, 
        detections: Dict[str, np.ndarray],
        distances: Optional[np.ndarray] = None,
        inference_time: Optional[float] = None
    ) -> np.ndarray:
        """Draw detections with performance metrics overlay."""
        
        result_image = image.copy()
        
        # Draw performance metrics
        if inference_time:
            fps = 1.0 / inference_time
            cv2.putText(result_image, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_image, f"Latency: {inference_time*1000:.1f}ms", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detection count
        num_detections = len(detections['boxes'])
        cv2.putText(result_image, f"Objects: {num_detections}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detections
        boxes = detections['boxes']
        classes = detections['classes']
        confidences = detections.get('confidences', np.ones(len(boxes)))
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get class info
            class_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
            color = self.colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(conf * 4))
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare enhanced label
            label = f'{class_name}: {conf:.2f}'
            if distances is not None and i < len(distances):
                label += f' ({distances[i]:.1f}m)'
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw distance warning for close objects
            if distances is not None and i < len(distances) and distances[i] < 5.0:
                cv2.putText(result_image, "WARNING: CLOSE OBJECT", 
                           (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image
    
    def create_performance_dashboard(
        self, 
        metrics_history: List[Dict],
        save_path: Optional[str] = None
    ) -> None:
        """Create performance dashboard with metrics over time."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Autonomous Vehicle Perception - Performance Dashboard', fontsize=16)
        
        # Extract metrics
        timestamps = range(len(metrics_history))
        fps_values = [m.get('fps', 0) for m in metrics_history]
        latency_values = [m.get('latency_ms', 0) for m in metrics_history]
        detection_counts = [m.get('num_detections', 0) for m in metrics_history]
        confidence_avg = [m.get('avg_confidence', 0) for m in metrics_history]
        
        # FPS over time
        axes[0, 0].plot(timestamps, fps_values, 'b-', linewidth=2)
        axes[0, 0].set_title('Frames Per Second (FPS)')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True)
        axes[0, 0].axhline(y=30, color='r', linestyle='--', label='Target FPS')
        axes[0, 0].legend()
        
        # Latency over time
        axes[0, 1].plot(timestamps, latency_values, 'r-', linewidth=2)
        axes[0, 1].set_title('Inference Latency')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=50, color='r', linestyle='--', label='Max Latency')
        axes[0, 1].legend()
        
        # Detection count over time
        axes[1, 0].plot(timestamps, detection_counts, 'g-', linewidth=2)
        axes[1, 0].set_title('Number of Detections')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].grid(True)
        
        # Average confidence over time
        axes[1, 1].plot(timestamps, confidence_avg, 'm-', linewidth=2)
        axes[1, 1].set_title('Average Detection Confidence')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()