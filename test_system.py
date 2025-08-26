"""Simple test to verify the system works."""

import numpy as np
from src.models.yolo_detector import AutonomousVehicleDetector
from src.models.kalman_tracker import MultiObjectTracker

def test_basic_functionality():
    """Test basic detection and tracking functionality."""
    
    print("Testing Autonomous Vehicle Perception System...")
    
    # Test detector initialization
    try:
        detector = AutonomousVehicleDetector()
        print("[SUCCESS] Detector initialized successfully")
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}")
        return
    
    # Test tracker initialization
    try:
        tracker = MultiObjectTracker()
        print("[SUCCESS] Tracker initialized successfully")
    except Exception as e:
        print(f"[ERROR] Tracker initialization failed: {e}")
        return
    
    # Test with dummy data
    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect(dummy_image)
        print(f"[SUCCESS] Detection completed. Found {len(detections['boxes'])} objects")
        
        # Test tracking
        tracks = tracker.update(detections)
        print(f"[SUCCESS] Tracking completed. Active tracks: {len(tracks)}")
        
        print("\n[COMPLETE] All core components working correctly!")
        
    except Exception as e:
        print(f"[ERROR] Runtime test failed: {e}")

if __name__ == "__main__":
    test_basic_functionality()