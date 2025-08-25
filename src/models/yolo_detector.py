import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from pathlib import Path


class AutonomousVehicleDetector:
    """
    Real-time object detection system optimized for autonomous vehicle perception.
    
    Supports detection of vehicles, pedestrians, cyclists, traffic signs, and lane markings
    with specialized post-processing for automotive applications.
    """
    
    # Automotive-specific class mappings
    AUTOMOTIVE_CLASSES = {
        'car': 2, 'truck': 7, 'bus': 5, 'motorcycle': 3,
        'bicycle': 1, 'person': 0, 'traffic_light': 9,
        'stop_sign': 11, 'speed_limit': 12
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        """
        Initialize the detector with automotive-optimized parameters.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device for inference ('auto', 'cpu', 'cuda')
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # Performance optimization for real-time inference
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
    
    def detect(
        self, 
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Perform object detection on input image.
        
        Args:
            image: Input image as numpy array (BGR format)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing bounding boxes, classes, and optionally confidences
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detections
        detections = results[0].boxes
        
        if detections is None or len(detections) == 0:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'classes': np.array([]),
                'confidences': np.array([]) if return_confidence else None
            }
        
        # Convert to numpy arrays
        boxes = detections.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = detections.cls.cpu().numpy().astype(int)
        confidences = detections.conf.cpu().numpy() if return_confidence else None
        
        result = {
            'boxes': boxes,
            'classes': classes
        }
        
        if return_confidence:
            result['confidences'] = confidences
            
        return result
    
    def detect_automotive_objects(
        self, 
        image: np.ndarray,
        filter_automotive: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Detect objects relevant for autonomous driving.
        
        Args:
            image: Input image
            filter_automotive: Whether to filter for automotive-relevant classes
            
        Returns:
            Filtered detections for autonomous vehicle perception
        """
        detections = self.detect(image)
        
        if not filter_automotive or len(detections['classes']) == 0:
            return detections
        
        # Filter for automotive-relevant classes
        automotive_mask = np.isin(detections['classes'], list(self.AUTOMOTIVE_CLASSES.values()))
        
        filtered_detections = {
            'boxes': detections['boxes'][automotive_mask],
            'classes': detections['classes'][automotive_mask],
            'confidences': detections['confidences'][automotive_mask] if detections['confidences'] is not None else None
        }
        
        return filtered_detections
    
    def calculate_distance_estimation(
        self, 
        boxes: np.ndarray, 
        classes: np.ndarray,
        camera_height: float = 1.5,  # meters
        focal_length: float = 800.0   # pixels
    ) -> np.ndarray:
        """
        Estimate distance to detected objects using monocular depth estimation.
        
        Args:
            boxes: Bounding boxes (x1, y1, x2, y2)
            classes: Object classes
            camera_height: Height of camera from ground (meters)
            focal_length: Camera focal length in pixels
            
        Returns:
            Estimated distances in meters
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Simplified distance estimation based on object height in image
        # This is a basic implementation - in practice, you'd use stereo vision or LiDAR
        
        # Typical object heights (meters)
        object_heights = {
            0: 1.7,   # person
            1: 1.5,   # bicycle
            2: 1.5,   # car
            3: 1.3,   # motorcycle
            5: 3.0,   # bus
            7: 2.5,   # truck
        }
        
        distances = []
        for box, cls in zip(boxes, classes):
            if cls in object_heights:
                # Calculate object height in pixels
                object_height_pixels = box[3] - box[1]
                
                # Estimate distance using similar triangles
                real_height = object_heights[cls]
                distance = (real_height * focal_length) / object_height_pixels
                distances.append(max(distance, 1.0))  # Minimum 1 meter
            else:
                distances.append(10.0)  # Default distance
        
        return np.array(distances)
    
    def benchmark_performance(self, image: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark detection performance for real-time applications.
        
        Args:
            image: Test image
            num_runs: Number of inference runs
            
        Returns:
            Performance metrics
        """
        import time
        
        # Warmup
        for _ in range(10):
            self.detect(image)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.detect(image)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_inference_time = total_time / num_runs
        fps = 1.0 / avg_inference_time
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'total_time_s': total_time
        }