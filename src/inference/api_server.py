"""FastAPI server for autonomous vehicle perception."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from typing import List, Dict
import io
from PIL import Image
import time

from src.models.yolo_detector import AutonomousVehicleDetector
from src.models.kalman_tracker import MultiObjectTracker
from src.utils.logger import setup_logger

app = FastAPI(
    title="Autonomous Vehicle Perception API",
    description="Real-time object detection and tracking for autonomous vehicles",
    version="1.0.0"
)

# Initialize models
detector = AutonomousVehicleDetector()
tracker = MultiObjectTracker()
logger = setup_logger("api_server")

# Class names mapping
CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck', 9: 'traffic_light', 11: 'stop_sign'
}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in uploaded image."""
    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        detections = detector.detect_automotive_objects(image_np)
        distances = detector.calculate_distance_estimation(
            detections['boxes'], detections['classes']
        )
        
        inference_time = time.time() - start_time
        
        # Format response
        results = []
        for i, (box, cls, conf) in enumerate(zip(
            detections['boxes'], 
            detections['classes'], 
            detections['confidences']
        )):
            results.append({
                "class_id": int(cls),
                "class_name": CLASS_NAMES.get(int(cls), f"class_{cls}"),
                "confidence": float(conf),
                "bbox": box.tolist(),
                "distance_m": float(distances[i]) if i < len(distances) else None
            })
        
        return JSONResponse({
            "status": "success",
            "detections": results,
            "num_objects": len(results),
            "inference_time_ms": inference_time * 1000,
            "fps": 1.0 / inference_time if inference_time > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track")
async def track_objects(file: UploadFile = File(...)):
    """Detect and track objects in uploaded image."""
    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run detection
        detections = detector.detect_automotive_objects(image_np)
        
        # Run tracking
        tracks = tracker.update(detections)
        
        inference_time = time.time() - start_time
        
        # Format tracking results
        track_results = []
        for track in tracks:
            track_results.append({
                "track_id": track.id,
                "class_id": int(track.class_id),
                "class_name": CLASS_NAMES.get(int(track.class_id), f"class_{track.class_id}"),
                "confidence": float(track.confidence),
                "bbox": track.bbox.tolist(),
                "hits": track.hits,
                "age": track.age
            })
        
        return JSONResponse({
            "status": "success",
            "tracks": track_results,
            "num_tracks": len(track_results),
            "inference_time_ms": inference_time * 1000,
            "fps": 1.0 / inference_time if inference_time > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Tracking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "autonomous-vehicle-perception"}

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    # Create dummy image for benchmark
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    metrics = detector.benchmark_performance(test_image, num_runs=10)
    
    return {
        "performance": metrics,
        "model_info": {
            "architecture": "YOLOv8",
            "input_size": 640,
            "num_classes": len(detector.AUTOMOTIVE_CLASSES),
            "device": detector.device
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)