"""Unit tests for YOLO detector."""

import pytest
import numpy as np
import cv2
from src.models.yolo_detector import AutonomousVehicleDetector

class TestAutonomousVehicleDetector:
    
    @pytest.fixture
    def detector(self):
        return AutonomousVehicleDetector()
    
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        assert detector.confidence_threshold == 0.5
        assert detector.iou_threshold == 0.45
        assert detector.model is not None
    
    def test_detection_output_format(self, detector, sample_image):
        detections = detector.detect(sample_image)
        
        assert 'boxes' in detections
        assert 'classes' in detections
        assert 'confidences' in detections
        assert isinstance(detections['boxes'], np.ndarray)
        assert isinstance(detections['classes'], np.ndarray)
    
    def test_automotive_filtering(self, detector, sample_image):
        detections = detector.detect_automotive_objects(sample_image)
        
        # Should return same format as regular detection
        assert 'boxes' in detections
        assert 'classes' in detections
    
    def test_distance_estimation(self, detector):
        # Mock detection data
        boxes = np.array([[100, 100, 200, 300], [300, 150, 400, 250]])
        classes = np.array([2, 0])  # car, person
        
        distances = detector.calculate_distance_estimation(boxes, classes)
        
        assert len(distances) == 2
        assert all(d > 0 for d in distances)
    
    def test_performance_benchmark(self, detector, sample_image):
        metrics = detector.benchmark_performance(sample_image, num_runs=5)
        
        assert 'avg_inference_time_ms' in metrics
        assert 'fps' in metrics
        assert metrics['fps'] > 0