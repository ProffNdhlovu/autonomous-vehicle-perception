"""Kalman filter-based multi-object tracking for autonomous vehicles."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
import uuid


@dataclass
class Track:
    """Represents a single object track."""
    id: str
    class_name: str
    state: np.ndarray  # [x, y, vx, vy]
    covariance: np.ndarray
    age: int
    hits: int
    time_since_update: int
    confidence: float
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())[:8]


class KalmanTracker:
    """Multi-object tracker using Kalman filters.
    
    Implements tracking with constant velocity motion model,
    optimized for autonomous vehicle scenarios.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        min_hits: int = 3
    ):
        """Initialize tracker.
        
        Args:
            max_disappeared: Max frames a track can be unmatched before deletion.
            max_distance: Maximum distance for association.
            min_hits: Minimum hits before a track is considered confirmed.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.frame_count = 0
        
        # Kalman filter matrices
        self.dt = 1.0  # Time step
        self._setup_kalman_matrices()
        
    def _setup_kalman_matrices(self):
        """Setup Kalman filter matrices for constant velocity model."""
        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = 0.1  # Process noise
        self.Q = np.array([
            [self.dt**4/4, 0, self.dt**3/2, 0],
            [0, self.dt**4/4, 0, self.dt**3/2],
            [self.dt**3/2, 0, self.dt**2, 0],
            [0, self.dt**3/2, 0, self.dt**2]
        ]) * q
        
        # Measurement noise covariance
        r = 10.0  # Measurement noise
        self.R = np.eye(2) * r
        
        # Initial covariance
        self.P_init = np.eye(4) * 100
        
    def _bbox_to_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Convert bounding box to center coordinates."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return cx, cy
    
    def _predict_track(self, track: Track) -> Track:
        """Predict next state for a track."""
        # Predict state
        predicted_state = self.F @ track.state
        predicted_covariance = self.F @ track.covariance @ self.F.T + self.Q
        
        # Update track
        track.state = predicted_state
        track.covariance = predicted_covariance
        track.time_since_update += 1
        track.age += 1
        
        return track
    
    def _update_track(self, track: Track, detection: Dict) -> Track:
        """Update track with new detection."""
        # Get measurement
        cx, cy = self._bbox_to_center(detection['bbox'])
        z = np.array([cx, cy])
        
        # Kalman update
        y = z - self.H @ track.state  # Innovation
        S = self.H @ track.covariance @ self.H.T + self.R  # Innovation covariance
        K = track.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        track.state = track.state + K @ y
        track.covariance = (np.eye(4) - K @ self.H) @ track.covariance
        
        # Update track metadata
        track.hits += 1
        track.time_since_update = 0
        track.confidence = detection['confidence']
        
        return track
    
    def _compute_distance_matrix(
        self, 
        tracks: List[Track], 
        detections: List[Dict]
    ) -> np.ndarray:
        """Compute distance matrix between tracks and detections."""
        if not tracks or not detections:
            return np.array([])
            
        distances = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            track_pos = track.state[:2]  # x, y position
            
            for j, detection in enumerate(detections):
                det_pos = np.array(self._bbox_to_center(detection['bbox']))
                distance = np.linalg.norm(track_pos - det_pos)
                distances[i, j] = distance
                
        return distances
    
    def _associate_detections_to_tracks(
        self,
        tracks: List[Track],
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using Hungarian algorithm."""
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []
            
        # Compute distance matrix
        distances = self._compute_distance_matrix(tracks, detections)
        
        # Apply distance threshold
        distances[distances > self.max_distance] = self.max_distance + 1
        
        # Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(distances)
        
        # Filter out assignments with distance > threshold
        matches = []
        for t_idx, d_idx in zip(track_indices, detection_indices):
            if distances[t_idx, d_idx] <= self.max_distance:
                matches.append((t_idx, d_idx))
        
        # Find unmatched tracks and detections
        matched_track_indices = [m[0] for m in matches]
        matched_detection_indices = [m[1] for m in matches]
        
        unmatched_tracks = [
            i for i in range(len(tracks)) 
            if i not in matched_track_indices
        ]
        unmatched_detections = [
            i for i in range(len(detections)) 
            if i not in matched_detection_indices
        ]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _create_new_track(self, detection: Dict) -> Track:
        """Create new track from detection."""
        cx, cy = self._bbox_to_center(detection['bbox'])
        
        # Initialize state [x, y, vx, vy]
        initial_state = np.array([cx, cy, 0, 0])
        
        track = Track(
            id=None,  # Will be auto-generated
            class_name=detection['class_name'],
            state=initial_state,
            covariance=self.P_init.copy(),
            age=1,
            hits=1,
            time_since_update=0,
            confidence=detection['confidence']
        )
        
        return track
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detections from detector.
            
        Returns:
            List of confirmed tracks.
        """
        self.frame_count += 1
        
        # Predict all existing tracks
        for track in self.tracks:
            self._predict_track(track)
        
        # Associate detections to tracks
        matches, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_tracks(self.tracks, detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            self._update_track(self.tracks[track_idx], detections[detection_idx])
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = self._create_new_track(detections[detection_idx])
            self.tracks.append(new_track)
        
        # Remove old tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_disappeared
        ]
        
        # Return confirmed tracks only
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits
        ]
        
        return confirmed_tracks
    
    def get_track_predictions(self, steps: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Get future position predictions for all tracks.
        
        Args:
            steps: Number of future steps to predict.
            
        Returns:
            Dictionary mapping track IDs to list of predicted positions.
        """
        predictions = {}
        
        for track in self.tracks:
            if track.hits < self.min_hits:
                continue
                
            future_positions = []
            state = track.state.copy()
            
            for _ in range(steps):
                state = self.F @ state
                future_positions.append((state[0], state[1]))
            
            predictions[track.id] = future_positions
            
        return predictions