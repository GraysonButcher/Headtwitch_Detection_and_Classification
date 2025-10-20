"""
HTR Detection algorithms for the HTR Analysis Tool.
Contains EarsDetector, HeadDetector, and CombinedDetector classes.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple, Dict, Any, Optional
from .data_processing import SleapDataLoader, SignalProcessor
from .config import EarDetectorConfig, HeadDetectorConfig, NodeMapping


class EarsDetector:
    """Detects headshake events based on ear movement patterns."""
    
    def __init__(self, data_loader: SleapDataLoader, config: EarDetectorConfig, node_mapping: NodeMapping):
        self.data_loader = data_loader
        self.config = config
        self.node_mapping = node_mapping
        self.signal_processor = SignalProcessor(data_loader, node_mapping)
        
        # Computed signals and events
        self.left_ear_distances = None
        self.right_ear_distances = None
        self.crisscross_events = []
        self.headshake_events = []
        
    def detect_headshakes(self, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0) -> List[Tuple[int, int]]:
        """Run the complete ear-based headshake detection pipeline."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
            
        # Calculate ear distance signals
        self.left_ear_distances, self.right_ear_distances = self.signal_processor.calculate_ear_distances(
            start_frame, end_frame, instance
        )
        
        # Detect crisscross events
        self._detect_crisscross_units(start_frame)
        
        # Group crisscrosses into headshakes
        self._group_into_headshakes()
        
        # Apply filters
        self._filter_headshakes(start_frame, end_frame, instance)
        
        # Merge adjacent headshakes
        self._merge_adjacent_headshakes()
        
        return self.headshake_events
    
    def _detect_crisscross_units(self, frame_offset: int = 0):
        """Detect crisscross patterns in ear distance signals."""
        left_dist = self.left_ear_distances
        right_dist = self.right_ear_distances
        
        # Find peaks and valleys in each ear signal
        left_peaks, _ = find_peaks(left_dist, height=self.config.peak_threshold, distance=self.config.max_gap)
        right_peaks, _ = find_peaks(right_dist, height=self.config.peak_threshold, distance=self.config.max_gap)
        left_valleys, _ = find_peaks(-left_dist, height=-self.config.valley_threshold, distance=self.config.max_gap)
        right_valleys, _ = find_peaks(-right_dist, height=-self.config.valley_threshold, distance=self.config.max_gap)
        
        # Combine all events with their types and values
        all_events = sorted([
            (frame + frame_offset, 'LP', left_dist[frame]) for frame in left_peaks
        ] + [
            (frame + frame_offset, 'RP', right_dist[frame]) for frame in right_peaks
        ] + [
            (frame + frame_offset, 'LV', left_dist[frame]) for frame in left_valleys
        ] + [
            (frame + frame_offset, 'RV', right_dist[frame]) for frame in right_valleys
        ])
        
        # Look for crisscross patterns (LP->RV or RP->LV within quick_gap frames)
        events = []
        i = 0
        while i < len(all_events) - 1:
            frame1, type1, val1 = all_events[i]
            frame2, type2, val2 = all_events[i + 1]
            
            if abs(frame2 - frame1) <= self.config.quick_gap:
                if (type1 == 'LP' and type2 == 'RV') or (type1 == 'RV' and type2 == 'LP'):
                    events.append({
                        'frames': (frame1, frame2),
                        'type': 'LPRV',
                        'left_amp': val1,
                        'right_amp': val2
                    })
                    i += 2
                    continue
                elif (type1 == 'RP' and type2 == 'LV') or (type1 == 'LV' and type2 == 'RP'):
                    events.append({
                        'frames': (frame1, frame2),
                        'type': 'RPLV',
                        'left_amp': val2 if type1 == 'RP' else val1,
                        'right_amp': val1 if type1 == 'RP' else val2
                    })
                    i += 2
                    continue
            i += 1
        
        self.crisscross_events = sorted(events, key=lambda x: x['frames'][0])
        return self.crisscross_events
    
    def _group_into_headshakes(self):
        """Group crisscross events into headshakes based on temporal proximity."""
        if not self.crisscross_events:
            self.headshake_events = []
            return
        
        headshakes = []
        current_shake_units = [self.crisscross_events[0]]
        
        for i in range(1, len(self.crisscross_events)):
            prev_event = current_shake_units[-1]
            curr_event = self.crisscross_events[i]
            
            # Check if events are close enough and alternating types
            if (curr_event['type'] != prev_event['type'] and 
                (min(curr_event['frames']) - max(prev_event['frames'])) <= self.config.between_unit_gap):
                current_shake_units.append(curr_event)
            else:
                # End current headshake if it has enough crisscrosses
                if len(current_shake_units) >= self.config.min_crisscrosses:
                    start_frame = min(current_shake_units[0]['frames'])
                    end_frame = max(current_shake_units[-1]['frames'])
                    headshakes.append((start_frame, end_frame))
                
                # Start new headshake
                current_shake_units = [curr_event]
        
        # Add final headshake if valid
        if len(current_shake_units) >= self.config.min_crisscrosses:
            start_frame = min(current_shake_units[0]['frames'])
            end_frame = max(current_shake_units[-1]['frames'])
            headshakes.append((start_frame, end_frame))
        
        self.headshake_events = headshakes
    
    def _filter_headshakes(self, start_frame: int, end_frame: int, instance: int):
        """Apply confidence and amplitude filters to headshakes."""
        if not self.headshake_events:
            return
        
        filtered_events = []
        
        for shake_start, shake_end in self.headshake_events:
            # Adjust frame indices to data range
            data_start = max(0, shake_start - start_frame)
            data_end = min(self.data_loader.total_frames, shake_end - start_frame)
            
            if data_end <= data_start:
                continue
            
            # Apply median score filter
            if self.config.apply_median_score_filter:
                # Get confidence scores for ear and head nodes
                relevant_nodes = [self.node_mapping.left_ear, self.node_mapping.right_ear, self.node_mapping.head]
                scores = self.data_loader.point_scores[data_start:data_end, relevant_nodes, instance]
                
                if scores.size > 0:
                    median_scores = np.median(scores, axis=0)
                    if not np.all(median_scores >= self.config.median_score_threshold):
                        continue
            
            
            filtered_events.append((shake_start, shake_end))
        
        self.headshake_events = filtered_events
    
    def _merge_adjacent_headshakes(self):
        """Merge headshakes that are very close together."""
        if not self.headshake_events:
            return
        
        self.headshake_events.sort(key=lambda x: x[0])
        merged = [self.headshake_events[0]]
        
        for next_start, next_end in self.headshake_events[1:]:
            last_start, last_end = merged[-1]
            
            if next_start - last_end <= self.config.merge_gap:
                # Merge by extending the end time
                merged[-1] = (last_start, next_end)
            else:
                merged.append((next_start, next_end))
        
        self.headshake_events = merged


class HeadDetector:
    """Detects headshake events based on head oscillation patterns."""
    
    def __init__(self, data_loader: SleapDataLoader, config: HeadDetectorConfig, node_mapping: NodeMapping):
        self.data_loader = data_loader
        self.config = config
        self.node_mapping = node_mapping
        self.signal_processor = SignalProcessor(data_loader, node_mapping)
        
        # Computed signals and events
        self.signal = None
        self.signal_raw = None
        self.oscillation_events = []
        self.cycles = []
        self.headshake_events = []
        
    def detect_headshakes(self, start_frame: int = 0, end_frame: Optional[int] = None, instance: int = 0) -> List[Tuple[int, int]]:
        """Run the complete head-based headshake detection pipeline."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
        
        # Calculate head distance signal
        self.signal_raw = self.signal_processor.calculate_head_distance_signal(
            start_frame, end_frame, instance, self.config.interpolation_method
        )
        
        # Apply smoothing if requested
        self.signal = self.signal_raw.copy()
        if self.config.use_smoothing:
            self._apply_smoothing()
        
        # Detect oscillations
        self._detect_oscillations(start_frame)
        
        # Build cycles from oscillation events
        self._build_cycles_from_events(start_frame)
        
        # Filter cycles
        self._filter_cycles()
        
        # Group cycles into headshakes
        self._group_cycles_into_headshakes()
        
        # Apply additional filters
        self._filter_headshakes_by_confidence(start_frame, end_frame, instance)
        self._filter_headshakes_by_median_amplitude()
        
        return self.headshake_events
    
    def _apply_smoothing(self):
        """Apply Savitzky-Golay smoothing to the signal."""
        win = self.config.smoothing_window
        poly = self.config.smoothing_polyorder
        
        if win > poly and len(self.signal) > win:
            self.signal = savgol_filter(self.signal, window_length=win, polyorder=poly)
    
    def _detect_oscillations(self, frame_offset: int = 0):
        """Detect peaks and valleys in the head signal."""
        prom = self.config.peak_prominence
        dist = self.config.peak_distance
        
        peaks, _ = find_peaks(self.signal, prominence=prom, distance=dist)
        valleys, _ = find_peaks(-self.signal, prominence=prom, distance=dist)
        
        self.oscillation_events = sorted([
            (i + frame_offset, 'peak') for i in peaks
        ] + [
            (i + frame_offset, 'valley') for i in valleys
        ])
    
    def _build_cycles_from_events(self, frame_offset: int = 0):
        """Build oscillation cycles from peak/valley events."""
        self.cycles = []
        
        for i in range(len(self.oscillation_events) - 1):
            t1, l1 = self.oscillation_events[i]
            t2, l2 = self.oscillation_events[i + 1]
            
            # Only create cycles between different event types (peak->valley or valley->peak)
            if l1 == l2:
                continue
            
            # Convert to indices in signal array
            idx1 = t1 - frame_offset
            idx2 = t2 - frame_offset
            
            if 0 <= idx1 < len(self.signal) and 0 <= idx2 < len(self.signal):
                amplitude = abs(self.signal[idx1] - self.signal[idx2])
                self.cycles.append({
                    'start': t1,
                    'end': t2,
                    'amplitude': amplitude
                })
    
    def _filter_cycles(self):
        """Filter cycles based on duration and amplitude criteria."""
        filtered_cycles = []
        
        for cycle in self.cycles:
            duration = cycle['end'] - cycle['start']
            amplitude = cycle['amplitude']
            
            # Check duration constraints
            if not (self.config.min_cycle_duration <= duration <= self.config.max_cycle_duration):
                continue
            
            # Check amplitude constraint
            if (hasattr(self.config, 'amplitude_threshold') and 
                self.config.amplitude_threshold is not None and 
                amplitude < self.config.amplitude_threshold):
                continue
            
            filtered_cycles.append(cycle)
        
        self.cycles = filtered_cycles
    
    def _group_cycles_into_headshakes(self):
        """Group cycles into headshakes based on temporal proximity."""
        if not self.cycles:
            self.headshake_events = []
            return
        
        max_gap = self.config.max_cycle_gap
        min_cycles = self.config.min_oscillations
        
        headshakes = []
        current_group = [self.cycles[0]]
        
        for cycle in self.cycles[1:]:
            # Check if cycle is close enough to the previous group
            if cycle['start'] - current_group[-1]['end'] <= max_gap:
                current_group.append(cycle)
            else:
                # End current headshake if it has enough cycles
                if len(current_group) >= min_cycles:
                    start_frame = current_group[0]['start']
                    end_frame = current_group[-1]['end']
                    headshakes.append((start_frame, end_frame))
                
                # Start new headshake
                current_group = [cycle]
        
        # Add final headshake if valid
        if len(current_group) >= min_cycles:
            start_frame = current_group[0]['start']
            end_frame = current_group[-1]['end']
            headshakes.append((start_frame, end_frame))
        
        self.headshake_events = headshakes
    
    def _filter_headshakes_by_confidence(self, start_frame: int, end_frame: int, instance: int):
        """Filter headshakes based on confidence scores."""
        if not hasattr(self.config, 'median_score_threshold') or self.config.median_score_threshold is None:
            return
        
        threshold = self.config.median_score_threshold
        target_nodes = [self.node_mapping.back, self.node_mapping.nose, self.node_mapping.head]
        
        filtered_events = []
        
        for shake_start, shake_end in self.headshake_events:
            # Adjust frame indices
            data_start = max(0, shake_start - start_frame)
            data_end = min(end_frame - start_frame, shake_end - start_frame)
            
            if data_end <= data_start:
                continue
            
            scores = self.data_loader.point_scores[data_start:data_end, target_nodes, instance]
            
            if scores.size > 0:
                median_scores = np.median(scores, axis=0)
                if np.all(median_scores >= threshold):
                    filtered_events.append((shake_start, shake_end))
        
        self.headshake_events = filtered_events
    
    def _filter_headshakes_by_median_amplitude(self):
        """Filter headshakes based on median cycle amplitude."""
        if not hasattr(self.config, 'amplitude_median') or self.config.amplitude_median is None:
            return
        
        threshold = self.config.amplitude_median
        filtered_events = []
        
        for shake_start, shake_end in self.headshake_events:
            # Find cycles within this headshake
            cycles_in_shake = [
                cycle for cycle in self.cycles 
                if shake_start <= cycle['start'] and cycle['end'] <= shake_end
            ]
            
            if cycles_in_shake:
                amplitudes = [cycle['amplitude'] for cycle in cycles_in_shake]
                median_amplitude = np.median(amplitudes)
                
                if median_amplitude >= threshold:
                    filtered_events.append((shake_start, shake_end))
        
        self.headshake_events = filtered_events


class CombinedDetector:
    """Combines ear and head detection methods and merges results."""
    
    def __init__(self, data_loader: SleapDataLoader, ear_config: EarDetectorConfig, 
                 head_config: HeadDetectorConfig, node_mapping: NodeMapping):
        self.data_loader = data_loader
        self.ear_config = ear_config
        self.head_config = head_config
        self.node_mapping = node_mapping
        
        self.ear_detector = EarsDetector(data_loader, ear_config, node_mapping)
        self.head_detector = HeadDetector(data_loader, head_config, node_mapping)
        
        self.combined_events = []
        
    def detect_headshakes(self, start_frame: int = 0, end_frame: Optional[int] = None, 
                         instance: int = 0, iou_threshold: float = 0.1) -> Tuple[List[Dict], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Run both detectors and combine results."""
        if end_frame is None:
            end_frame = self.data_loader.total_frames
        
        # Run both detectors
        ear_events = self.ear_detector.detect_headshakes(start_frame, end_frame, instance)
        head_events = self.head_detector.detect_headshakes(start_frame, end_frame, instance)
        
        # Combine results with confidence levels
        self.combined_events = self._combine_results(ear_events, head_events, iou_threshold)
        
        return self.combined_events, ear_events, head_events
    
    def _combine_results(self, ear_events: List[Tuple[int, int]], head_events: List[Tuple[int, int]], 
                        iou_threshold: float) -> List[Dict]:
        """Combine detection results with confidence scoring."""
        combined = []
        matched_ear = set()
        matched_head = set()
        
        # Find overlapping detections
        for i, ear_event in enumerate(ear_events):
            for j, head_event in enumerate(head_events):
                if j in matched_head:
                    continue
                
                iou = self._calculate_iou(ear_event, head_event)
                if iou >= iou_threshold:
                    matched_ear.add(i)
                    matched_head.add(j)
                    
                    # Merge the time windows
                    start_frame = min(ear_event[0], head_event[0])
                    end_frame = max(ear_event[1], head_event[1])
                    
                    combined.append({
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "confidence": "High",
                        "detection_method": "Both",
                        "iou": iou
                    })
                    break
        
        # Add unmatched ear detections
        for i, ear_event in enumerate(ear_events):
            if i not in matched_ear:
                combined.append({
                    "start_frame": ear_event[0],
                    "end_frame": ear_event[1],
                    "confidence": "Medium",
                    "detection_method": "Ears-Only",
                    "iou": 0.0
                })
        
        # Add unmatched head detections  
        for j, head_event in enumerate(head_events):
            if j not in matched_head:
                combined.append({
                    "start_frame": head_event[0],
                    "end_frame": head_event[1],
                    "confidence": "Low",
                    "detection_method": "Head-Only",
                    "iou": 0.0
                })
        
        # Sort by start frame
        combined.sort(key=lambda x: x["start_frame"])
        return combined
    
    def _calculate_iou(self, event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
        """Calculate Intersection over Union for temporal windows."""
        start1, end1 = event1
        start2, end2 = event2
        
        # Calculate intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        if intersection == 0:
            return 0.0
        
        # Calculate union
        union = (end1 - start1) + (end2 - start2) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get detection results as a pandas DataFrame."""
        if not self.combined_events:
            return pd.DataFrame(columns=["start_frame", "end_frame", "confidence", "detection_method"])
        
        return pd.DataFrame(self.combined_events)