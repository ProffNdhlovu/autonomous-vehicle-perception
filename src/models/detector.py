"""YOLO-based object detection model for autonomous vehicles."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class AutonomousVehicleDetector:
    """Real-time object detection system optimized for autonomous vehicles.
    
    This class implements a YOLO-based detection system specifically tuned
    for detecting vehicles, pedestrians, cyclists, and traffic signs.
    """
    
    # Classes relevant for autonomous driving
    AV_CLASSES = {
        0: 'person',
        1: 'bicycle', 
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        9: 'traffic_light',
        11: 'stop_sign'
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """Initialize the detector.
        
        Args:
            model_path: Path to custom trained model. If None, uses YOLOv8n.
            device: Device to run inference on ('cpu', 'cuda', or 'auto').
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
        """
        self.device = self._setup_device(device)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')  # Use pretrained YOLOv8 nano
            
        self.model.to(self.device)
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def detect(
        self, 
        image: np.ndarray,
        filter_classes: bool = True
    ) -> List[Dict]:
        """Detect objects in image.
        
        Args:
            image: Input image as numpy array (BGR format).
            filter_classes: Whether to filter for AV-relevant classes only.
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: float
            - class_id: int
            - class_name: str
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for AV-relevant classes if requested
                    if filter_classes and class_id not in self.AV_CLASSES:
                        continue
                        
                    class_name = self.AV_CLASSES.get(
                        class_id, 
                        self.model.names[class_id]
                    )
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
                    
        return detections
    
    def detect_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[List[Dict]]:
        """Detect objects in batch of images.
        
        Args:
            images: List of input images.
            
        Returns:
            List of detection lists for each image.
        """
        batch_detections = []
        
        for image in images:
            detections = self.detect(image)
            batch_detections.append(detections)
            
        return batch_detections
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2
    ) -> np.ndarray:
        """Visualize detections on image.
        
        Args:
            image: Input image.
            detections: List of detections from detect().
            thickness: Line thickness for bounding boxes.
            
        Returns:
            Image with visualized detections.
        """
        vis_image = image.copy()
        
        # Color map for different classes
        colors = {
            'person': (0, 255, 0),      # Green
            'bicycle': (255, 255, 0),   # Cyan
            'car': (255, 0, 0),         # Blue
            'motorcycle': (255, 0, 255), # Magenta
            'bus': (0, 165, 255),       # Orange
            'truck': (0, 0, 255),       # Red
            'traffic_light': (0, 255, 255), # Yellow
            'stop_sign': (0, 0, 128)    # Maroon
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for class
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
        return vis_image