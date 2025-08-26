"""Real-time detection and tracking pipeline."""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import time
import click
from pathlib import Path
import json

from src.models.detector import AutonomousVehicleDetector
from src.tracking.kalman_tracker import KalmanTracker
from src.utils.visualization import DetectionVisualizer
from src.utils.logger import setup_logger


class RealTimeDetectionPipeline:
    """Real-time detection and tracking pipeline for autonomous vehicles."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        tracking_enabled: bool = True,
        save_results: bool = False,
        output_dir: str = "results"
    ):
        """Initialize the pipeline.
        
        Args:
            model_path: Path to trained model weights.
            confidence_threshold: Detection confidence threshold.
            iou_threshold: IoU threshold for NMS.
            tracking_enabled: Whether to enable object tracking.
            save_results: Whether to save detection results.
            output_dir: Directory to save results.
        """
        self.logger = setup_logger('detection_pipeline')
        
        # Initialize detector
        self.detector = AutonomousVehicleDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Initialize tracker if enabled
        self.tracking_enabled = tracking_enabled
        if tracking_enabled:
            self.tracker = KalmanTracker(
                max_disappeared=30,
                max_distance=100.0,
                min_hits=3
            )
        
        # Initialize visualizer
        self.visualizer = DetectionVisualizer()
        
        # Setup result saving
        self.save_results = save_results
        if save_results:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.results = []
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
    def process_frame(
        self, 
        frame: np.ndarray, 
        frame_id: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process a single frame.
        
        Args:
            frame: Input frame (BGR format).
            frame_id: Optional frame identifier.
            
        Returns:
            Tuple of (annotated_frame, frame_results).
        """
        start_time = time.time()
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # Run tracking if enabled
        tracks = []
        if self.tracking_enabled and hasattr(self, 'tracker'):
            tracks = self.tracker.update(detections)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Create frame results
        frame_results = {
            'frame_id': frame_id or self.frame_count,
            'timestamp': time.time(),
            'inference_time': inference_time,
            'detections': detections,
            'tracks': [
                {
                    'id': track.id,
                    'class_name': track.class_name,
                    'position': track.state[:2].tolist(),
                    'velocity': track.state[2:].tolist(),
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits
                }
                for track in tracks
            ] if self.tracking_enabled else []
        }
        
        # Visualize results
        annotated_frame = self.visualizer.draw_detections_and_tracks(
            frame, detections, tracks if self.tracking_enabled else []
        )
        
        # Add performance info
        fps = 1.0 / inference_time if inference_time > 0 else 0
        avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        
        info_text = [
            f"FPS: {fps:.1f}",
            f"Avg FPS: {avg_fps:.1f}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracks)}" if self.tracking_enabled else ""
        ]
        
        self.visualizer.draw_info_panel(annotated_frame, info_text)
        
        # Save results if enabled
        if self.save_results:
            self.results.append(frame_results)
        
        return annotated_frame, frame_results
    
    def process_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        display: bool = True
    ) -> List[Dict]:
        """Process video file.
        
        Args:
            video_path: Path to input video.
            output_path: Path to save output video.
            display: Whether to display video during processing.
            
        Returns:
            List of frame results.
        """
        self.logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame, result = self.process_frame(frame, frame_id)
                frame_results.append(result)
                
                # Write frame if output specified
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Autonomous Vehicle Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_id += 1
                
                # Progress update
                if frame_id % 100 == 0:
                    progress = (frame_id / total_frames) * 100
                    self.logger.info(f"Processed {frame_id}/{total_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        self.logger.info(f"Video processing completed. Processed {frame_id} frames.")
        return frame_results
    
    def process_camera(
        self, 
        camera_id: int = 0,
        output_path: Optional[str] = None
    ):
        """Process live camera feed.
        
        Args:
            camera_id: Camera device ID.
            output_path: Path to save output video.
        """
        self.logger.info(f"Starting camera processing (camera {camera_id})")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Setup video writer if output path specified
        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    continue
                
                # Process frame
                annotated_frame, _ = self.process_frame(frame)
                
                # Write frame if output specified
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Autonomous Vehicle Detection - Live', annotated_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and self.save_results:
                    # Save current results
                    self.save_detection_results()
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        self.logger.info("Camera processing stopped.")
    
    def save_detection_results(self, filename: Optional[str] = None):
        """Save detection results to JSON file.
        
        Args:
            filename: Output filename. If None, auto-generate.
        """
        if not self.save_results or not hasattr(self, 'results'):
            self.logger.warning("No results to save")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Add summary statistics
        summary = {
            'total_frames': len(self.results),
            'average_fps': len(self.results) / self.total_inference_time if self.total_inference_time > 0 else 0,
            'total_processing_time': self.total_inference_time,
            'tracking_enabled': self.tracking_enabled
        }
        
        output_data = {
            'summary': summary,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics.
        """
        if self.frame_count == 0:
            return {}
        
        avg_inference_time = self.total_inference_time / self.frame_count
        avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_inference_time,
            'average_fps': avg_fps
        }


@click.command()
@click.option('--model', '-m', help='Path to trained model weights')
@click.option('--source', '-s', required=True, help='Video file, camera ID, or image directory')
@click.option('--output', '-o', help='Output video path')
@click.option('--conf', default=0.5, help='Confidence threshold')
@click.option('--iou', default=0.45, help='IoU threshold')
@click.option('--no-tracking', is_flag=True, help='Disable object tracking')
@click.option('--save-results', is_flag=True, help='Save detection results to JSON')
@click.option('--no-display', is_flag=True, help='Disable video display')
def main(
    model: str,
    source: str,
    output: str,
    conf: float,
    iou: float,
    no_tracking: bool,
    save_results: bool,
    no_display: bool
):
    """Real-time autonomous vehicle detection and tracking.
    
    Examples:
        # Process video file
        python -m src.inference.detector -s video.mp4 -o output.mp4
        
        # Process live camera
        python -m src.inference.detector -s 0
        
        # Use custom model
        python -m src.inference.detector -m models/best.pt -s video.mp4
    """
    # Initialize pipeline
    pipeline = RealTimeDetectionPipeline(
        model_path=model,
        confidence_threshold=conf,
        iou_threshold=iou,
        tracking_enabled=not no_tracking,
        save_results=save_results
    )
    
    try:
        # Determine source type
        if source.isdigit():
            # Camera source
            camera_id = int(source)
            pipeline.process_camera(camera_id, output)
        elif Path(source).is_file():
            # Video file
            pipeline.process_video(source, output, display=not no_display)
        else:
            raise ValueError(f"Invalid source: {source}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    
    finally:
        # Save results if enabled
        if save_results:
            pipeline.save_detection_results()
        
        # Print performance stats
        stats = pipeline.get_performance_stats()
        if stats:
            print

# Fix these import lines (around line 11-14):
from ..models.yolo_detector import AutonomousVehicleDetector
from ..models.kalman_tracker import MultiObjectTracker
from ..utils.visualization import DetectionVisualizer
from ..utils.logger import setup_logger