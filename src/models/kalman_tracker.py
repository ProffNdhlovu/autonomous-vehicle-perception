import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import uuid


@dataclass
class TrackState:
    """Represents the state of a tracked object."""
    id: str
    bbox: np.ndarray  # [x1, y1, x2, y2]
    velocity: np.ndarray  # [vx, vy]
    class_id: int
    confidence: float
    age: int
    hits: int
    time_since_update: int
    predicted_bbox: Optional[np.ndarray] = None


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    State vector: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self):
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=np.float32)
        
        # Observation matrix (we observe x, y, w, h)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01  # Lower process noise for velocities
        
        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 0.1
        
        # Initialize state and covariance
        self.x = np.zeros(8, dtype=np.float32)  # State vector
        self.P = np.eye(8, dtype=np.float32) * 1000  # Covariance matrix
        
    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # Return predicted bbox
    
    def update(self, measurement: np.ndarray):
        """Update with measurement."""
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def initialize(self, bbox: np.ndarray):
        """Initialize filter with first measurement."""
        self.x[:4] = bbox
        self.x[4:] = 0  # Initialize velocities to zero


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman filters and Hungarian algorithm for data association.
    Optimized for autonomous vehicle perception with automotive-specific parameters.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 100.0
    ):
        """
        Initialize the multi-object tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be unmatched before deletion
            min_hits: Minimum hits before a track is considered confirmed
            iou_threshold: IoU threshold for matching detections to tracks
            max_distance: Maximum distance for matching (pixels)
        """
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        
        self.tracks: List[TrackState] = []
        self.frame_count = 0
    
    def update(
        self, 
        detections: Dict[str, np.ndarray]
    ) -> List[TrackState]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Dictionary with 'boxes', 'classes', 'confidences'
            
        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1
        
        boxes = detections['boxes']
        classes = detections['classes']
        confidences = detections.get('confidences', np.ones(len(boxes)))
        
        # Predict existing tracks
        for track in self.tracks:
            if hasattr(track, 'kalman_filter'):
                track.predicted_bbox = track.kalman_filter.predict()
            track.time_since_update += 1
        
        # Match detections to tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            boxes, classes, confidences
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            bbox = self._convert_bbox_format(boxes[det_idx])
            
            # Update Kalman filter
            if hasattr(track, 'kalman_filter'):
                track.kalman_filter.update(bbox)
            
            # Update track state
            track.bbox = boxes[det_idx]
            track.confidence = confidences[det_idx]
            track.hits += 1
            track.time_since_update = 0
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_new_track(
                boxes[det_idx], 
                classes[det_idx], 
                confidences[det_idx]
            )
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks 
            if track.time_since_update < self.max_disappeared
        ]
        
        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks 
            if track.hits >= self.min_hits and track.time_since_update == 0
        ]
        
        return confirmed_tracks
    
    def _associate_detections_to_tracks(
        self, 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        confidences: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using Hungarian algorithm."""
        if len(self.tracks) == 0:
            return [], list(range(len(boxes))), []
        
        if len(boxes) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute cost matrix (IoU-based)
        cost_matrix = np.zeros((len(self.tracks), len(boxes)))
        
        for t, track in enumerate(self.tracks):
            for d, box in enumerate(boxes):
                # Only consider same class matches
                if track.class_id == classes[d]:
                    iou = self._calculate_iou(track.bbox, box)
                    cost_matrix[t, d] = 1 - iou  # Convert IoU to cost
                else:
                    cost_matrix[t, d] = 1.0  # High cost for different classes
        
        # Apply Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on IoU threshold
        matches = []
        unmatched_detections = list(range(len(boxes)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for t, d in zip(track_indices, detection_indices):
            if cost_matrix[t, d] < (1 - self.iou_threshold):
                matches.append((t, d))
                unmatched_detections.remove(d)
                unmatched_tracks.remove(t)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        return 0.0
    
    def _convert_bbox_format(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])
    
    def _create_new_track(
        self, 
        bbox: np.ndarray, 
        class_id: int, 
        confidence: float
    ):
        """Create a new track for unmatched detection."""
        track_id = str(uuid.uuid4())[:8]
        
        # Initialize Kalman filter
        kalman_filter = KalmanFilter()
        kalman_filter.initialize(self._convert_bbox_format(bbox))
        
        track = TrackState(
            id=track_id,
            bbox=bbox,
            velocity=np.array([0.0, 0.0]),
            class_id=class_id,
            confidence=confidence,
            age=1,
            hits=1,
            time_since_update=0
        )
        
        track.kalman_filter = kalman_filter
        self.tracks.append(track)
    
    def get_track_trajectories(self) -> Dict[str, List[np.ndarray]]:
        """Get trajectories for all active tracks."""
        trajectories = {}
        for track in self.tracks:
            if track.hits >= self.min_hits:
                if track.id not in trajectories:
                    trajectories[track.id] = []
                trajectories[track.id].append(track.bbox)
        return trajectories