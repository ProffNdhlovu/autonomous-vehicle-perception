"""Performance monitoring and profiling utilities."""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: float
    fps: float
    latency_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    num_detections: int = 0
    avg_confidence: float = 0.0

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Try to get GPU memory if available
                gpu_memory_mb = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except ImportError:
                    pass
                
                # Create metrics entry
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    fps=0.0,  # Will be updated by inference calls
                    latency_ms=0.0,  # Will be updated by inference calls
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    gpu_memory_mb=gpu_memory_mb
                )
                
                with self.lock:
                    self.metrics_history.append(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(interval)
    
    def record_inference(self, inference_time: float, num_detections: int, avg_confidence: float):
        """Record inference performance metrics."""
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        latency_ms = inference_time * 1000
        
        with self.lock:
            if self.metrics_history:
                # Update the latest metrics entry
                latest = self.metrics_history[-1]
                latest.fps = fps
                latest.latency_ms = latency_ms
                latest.num_detections = num_detections
                latest.avg_confidence = avg_confidence
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get performance metrics history."""
        with self.lock:
            history = list(self.metrics_history)
            if last_n:
                return history[-last_n:]
            return history
    
    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average performance metrics."""
        history = self.get_metrics_history(last_n)
        
        if not history:
            return {}
        
        total_metrics = len(history)
        avg_fps = sum(m.fps for m in history) / total_metrics
        avg_latency = sum(m.latency_ms for m in history) / total_metrics
        avg_cpu = sum(m.cpu_percent for m in history) / total_metrics
        avg_memory = sum(m.memory_mb for m in history) / total_metrics
        
        result = {
            "avg_fps": avg_fps,
            "avg_latency_ms": avg_latency,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory
        }
        
        # Add GPU memory if available
        gpu_metrics = [m.gpu_memory_mb for m in history if m.gpu_memory_mb is not None]
        if gpu_metrics:
            result["avg_gpu_memory_mb"] = sum(gpu_metrics) / len(gpu_metrics)
        
        return result
    
    def save_metrics(self, output_path: str):
        """Save metrics history to file."""
        history = self.get_metrics_history()
        
        # Convert to serializable format
        data = []
        for metrics in history:
            data.append({
                "timestamp": metrics.timestamp,
                "fps": metrics.fps,
                "latency_ms": metrics.latency_ms,
                "cpu_percent": metrics.cpu_percent,
                "memory_mb": metrics.memory_mb,
                "gpu_memory_mb": metrics.gpu_memory_mb,
                "num_detections": metrics.num_detections,
                "avg_confidence": metrics.avg_confidence
            })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def check_performance_thresholds(self, target_fps: float = 30, max_latency_ms: float = 50) -> Dict[str, bool]:
        """Check if performance meets target thresholds."""
        current = self.get_current_metrics()
        
        if not current:
            return {"fps_ok": False, "latency_ok": False}
        
        return {
            "fps_ok": current.fps >= target_fps,
            "latency_ok": current.latency_ms <= max_latency_ms,
            "current_fps": current.fps,
            "current_latency_ms": current.latency_ms
        }