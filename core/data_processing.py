"""
Data processing utilities for HTR Analysis Tool.
Handles H5 file loading, signal processing, and data validation.
"""
import numpy as np
import pandas as pd
import h5py
import os
from typing import Tuple, Optional, Dict, Any
from scipy.interpolate import interp1d
from .config import NodeMapping


def _line_intersection(p1, p2, q1, q2):
    """Calculate intersection point of two lines defined by point pairs."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def point_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line segment."""
    p, a, b = np.array(point), np.array(line_start), np.array(line_end)
    
    if np.all(a == b):
        return np.linalg.norm(p - a)
    
    # Project point onto line
    ap = p - a
    ab = b - a
    
    # Handle zero-length segment
    if np.dot(ab, ab) == 0:
        return np.linalg.norm(ap)
    
    t = np.dot(ap, ab) / np.dot(ab, ab)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


class SleapDataLoader:
    """Loads and validates SLEAP H5 tracking data."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.locations = None
        self.point_scores = None
        self.node_names = None
        self.total_frames = 0
        
    def load_data(self) -> bool:
        """Load tracking data from H5 file."""
        try:
            with h5py.File(self.filename, "r") as f:
                # Load tracking data
                raw_locations = f["tracks"][:]
                self.locations = self._normalize_locations(raw_locations)
                
                # Load confidence scores
                if "point_scores" in f:
                    raw_scores = f["point_scores"][:]
                    self.point_scores = self._normalize_scores(raw_scores)
                else:
                    # Create dummy scores if not available
                    self.point_scores = np.ones((self.locations.shape[0], self.locations.shape[1]))
                
                # Load node names if available
                if "node_names" in f:
                    try:
                        raw_names = f["node_names"][:]
                        self.node_names = [
                            n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) 
                            for n in raw_names
                        ]
                    except Exception:
                        self.node_names = None
                
                self.total_frames = self.locations.shape[0]
                return True
                
        except Exception as e:
            print(f"Error loading SLEAP data: {e}")
            return False
    
    def _normalize_locations(self, raw_locations):
        """Normalize location data to (frames, nodes, xy, instance) format."""
        # Handle different possible SLEAP export formats
        if raw_locations.ndim == 3:
            # Format: (frames, nodes, 2) - single instance
            return raw_locations[:, :, :, np.newaxis]
        elif raw_locations.ndim == 4:
            # Format: (frames, nodes, 2, instances) or transposed
            if raw_locations.shape[2] == 2:
                return raw_locations
            else:
                # Try transposing
                return raw_locations.transpose()
        else:
            raise ValueError(f"Unexpected location data shape: {raw_locations.shape}")
    
    def _normalize_scores(self, raw_scores):
        """Normalize confidence scores to (frames, nodes, instances) format."""
        # Follow legacy pattern: always transpose SLEAP export data
        return raw_scores.T
    
    def get_node_positions(self, node_idx: int, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0):
        """Get positions for a specific node across frames."""
        if self.locations is None:
            raise ValueError("No data loaded")
        
        if end_frame is None:
            end_frame = self.total_frames
        
        return self.locations[start_frame:end_frame, node_idx, :, instance]
    
    def get_node_scores(self, node_idx: int, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0):
        """Get confidence scores for a specific node across frames."""
        if self.point_scores is None:
            raise ValueError("No scores loaded")
        
        if end_frame is None:
            end_frame = self.total_frames
        
        return self.point_scores[start_frame:end_frame, node_idx, instance]


class SignalProcessor:
    """Processes tracking data to generate analysis signals."""
    
    def __init__(self, data_loader: SleapDataLoader, node_mapping: NodeMapping):
        self.data_loader = data_loader
        self.node_mapping = node_mapping
        
    def calculate_ear_distances(self, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate ear distances from head midline."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
        
        # Get node positions
        left_ear_pos = self.data_loader.get_node_positions(self.node_mapping.left_ear, start_frame, end_frame, instance)
        right_ear_pos = self.data_loader.get_node_positions(self.node_mapping.right_ear, start_frame, end_frame, instance)
        back_pos = self.data_loader.get_node_positions(self.node_mapping.back, start_frame, end_frame, instance)
        nose_pos = self.data_loader.get_node_positions(self.node_mapping.nose, start_frame, end_frame, instance)
        
        left_distances = []
        right_distances = []
        
        for i in range(len(back_pos)):
            # Find intersection of back-nose line with left-right ear line
            intersection = _line_intersection(back_pos[i], nose_pos[i], left_ear_pos[i], right_ear_pos[i])
            
            if intersection is not None:
                left_dist = np.linalg.norm(left_ear_pos[i] - intersection)
                right_dist = np.linalg.norm(right_ear_pos[i] - intersection)
            else:
                left_dist = np.nan
                right_dist = np.nan
            
            left_distances.append(left_dist)
            right_distances.append(right_dist)
        
        # Interpolate NaN values
        left_distances = pd.Series(left_distances).interpolate().fillna(0).to_numpy()
        right_distances = pd.Series(right_distances).interpolate().fillna(0).to_numpy()
        
        return left_distances, right_distances
    
    def calculate_head_distance_signal(self, start_frame: int = 0, end_frame: Optional[int] = None, 
                                     instance: int = 0, interpolation_method: str = 'linear') -> np.ndarray:
        """Calculate head distance from back-nose midline."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
        
        # Get node positions
        back_pos = self.data_loader.get_node_positions(self.node_mapping.back, start_frame, end_frame, instance)
        nose_pos = self.data_loader.get_node_positions(self.node_mapping.nose, start_frame, end_frame, instance)
        head_pos = self.data_loader.get_node_positions(self.node_mapping.head, start_frame, end_frame, instance)
        
        # Calculate perpendicular distances
        distances = []
        for i in range(len(head_pos)):
            dist = point_line_distance(head_pos[i], back_pos[i], nose_pos[i])
            distances.append(dist)
        
        signal = np.array(distances)
        
        # Interpolate NaN values if present
        if np.isnan(signal).any():
            signal = pd.Series(signal).interpolate(method=interpolation_method).fillna(0).to_numpy()
        
        return signal
    
    def validate_node_mapping(self) -> Dict[str, bool]:
        """Validate that node indices are valid for the loaded data."""
        if self.data_loader.locations is None:
            return {"error": "No data loaded"}
        
        max_nodes = self.data_loader.locations.shape[1]
        validation = {}
        
        for node_name in ['left_ear', 'right_ear', 'back', 'nose', 'head']:
            node_idx = getattr(self.node_mapping, node_name)
            validation[node_name] = 0 <= node_idx < max_nodes
        
        return validation
    
    def get_signal_quality_metrics(self, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0) -> Dict[str, float]:
        """Calculate signal quality metrics for the specified frame range."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
        
        metrics = {}
        
        # Get confidence scores for key nodes
        for node_name in ['left_ear', 'right_ear', 'back', 'nose', 'head']:
            node_idx = getattr(self.node_mapping, node_name)
            scores = self.data_loader.get_node_scores(node_idx, start_frame, end_frame, instance)
            
            metrics[f'{node_name}_median_confidence'] = np.median(scores)
            metrics[f'{node_name}_min_confidence'] = np.min(scores)
            metrics[f'{node_name}_mean_confidence'] = np.mean(scores)
        
        # Calculate overall quality metrics
        all_scores = []
        for node_name in ['left_ear', 'right_ear', 'back', 'nose', 'head']:
            node_idx = getattr(self.node_mapping, node_name)
            scores = self.data_loader.get_node_scores(node_idx, start_frame, end_frame, instance)
            all_scores.extend(scores)
        
        metrics['overall_median_confidence'] = np.median(all_scores)
        metrics['overall_min_confidence'] = np.min(all_scores)
        metrics['frames_analyzed'] = end_frame - start_frame
        
        return metrics


def validate_h5_file(filename: str) -> Dict[str, Any]:
    """Validate H5 file structure and return file info."""
    try:
        with h5py.File(filename, "r") as f:
            info = {
                'valid': True,
                'filename': os.path.basename(filename),
                'file_size_mb': os.path.getsize(filename) / (1024 * 1024),
                'datasets': list(f.keys()),
                'has_tracks': 'tracks' in f,
                'has_scores': 'point_scores' in f,
                'has_node_names': 'node_names' in f
            }
            
            if info['has_tracks']:
                tracks_shape = f['tracks'].shape
                info['tracks_shape'] = tracks_shape
                info['num_frames'] = tracks_shape[0] if len(tracks_shape) > 0 else 0
                info['num_nodes'] = tracks_shape[1] if len(tracks_shape) > 1 else 0
            
            if info['has_node_names']:
                try:
                    raw_names = f['node_names'][:]
                    info['node_names'] = [
                        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
                        for n in raw_names
                    ]
                except Exception:
                    info['node_names'] = None
            
            return info
            
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'filename': os.path.basename(filename) if filename else 'Unknown'
        }