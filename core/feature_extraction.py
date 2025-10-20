"""
Feature extraction for HTR candidates.
Extracts comprehensive features for machine learning classification.
"""
import numpy as np
import pandas as pd
import warnings
from scipy.signal import find_peaks
from typing import List, Dict, Any, Tuple, Optional
from .data_processing import SleapDataLoader, SignalProcessor
from .detectors import CombinedDetector, EarsDetector, HeadDetector
from .config import EarDetectorConfig, HeadDetectorConfig, NodeMapping


class FeatureExtractor:
    """Extracts features from HTR candidate windows for ML classification."""
    
    def __init__(self, data_loader: SleapDataLoader, ear_config: EarDetectorConfig, 
                 head_config: HeadDetectorConfig, node_mapping: NodeMapping, fps: int = 120):
        self.data_loader = data_loader
        self.ear_config = ear_config
        self.head_config = head_config
        self.node_mapping = node_mapping
        self.fps = fps
        
        self.signal_processor = SignalProcessor(data_loader, node_mapping)
        
    def extract_candidate_features(self, candidate_windows: List[Tuple[int, int]], 
                                 ear_detector: Optional[EarsDetector] = None,
                                 head_detector: Optional[HeadDetector] = None,
                                 instance: int = 0) -> pd.DataFrame:
        """Extract features for all candidate windows."""
        if not candidate_windows:
            return pd.DataFrame()
        
        feature_list = []
        
        for i, (start_frame, end_frame) in enumerate(candidate_windows):
            print(f"  Extracting features for candidate {i+1}/{len(candidate_windows)}: Frames {start_frame}-{end_frame}", end='\r')
            
            features = self._extract_features_for_window(
                start_frame, end_frame, ear_detector, head_detector, instance
            )
            feature_list.append(features)
        
        print()  # New line after progress
        
        if not feature_list:
            return pd.DataFrame()
        
        # Create DataFrame and add ground truth column
        results_df = pd.DataFrame(feature_list)
        results_df['ground_truth'] = '__'  # Placeholder for manual labeling
        
        # Organize columns
        core_cols = ['start_frame', 'end_frame', 'duration_frames', 'is_ear_detected', 'is_head_detected']
        other_cols = [col for col in results_df.columns if col not in core_cols and col != 'ground_truth']
        final_cols = core_cols + sorted(other_cols) + ['ground_truth']
        
        return results_df[final_cols]
    
    def _extract_features_for_window(self, start_frame: int, end_frame: int,
                                   ear_detector: Optional[EarsDetector] = None,
                                   head_detector: Optional[HeadDetector] = None,
                                   instance: int = 0) -> Dict[str, Any]:
        """Extract comprehensive features for a single candidate window."""
        features = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'duration_frames': end_frame - start_frame
        }
        
        # 1. Detection method features
        self._add_detection_features(features, start_frame, end_frame, ear_detector, head_detector)
        
        # 2. Crisscross ear features
        self._add_crisscross_features(features, start_frame, end_frame, ear_detector)
        
        # 3. Oscillatory features for different signals
        self._add_oscillatory_features(features, start_frame, end_frame, ear_detector, head_detector, instance)
        
        # 4. Kinematic features
        self._add_kinematic_features(features, start_frame, end_frame, instance)
        
        # 5. Confidence features
        self._add_confidence_features(features, start_frame, end_frame, instance)
        
        return features
    
    def _add_detection_features(self, features: Dict, start_frame: int, end_frame: int,
                              ear_detector: Optional[EarsDetector], head_detector: Optional[HeadDetector]):
        """Add features indicating which detection methods found this candidate."""
        features['is_ear_detected'] = 0
        features['is_head_detected'] = 0
        
        if ear_detector and ear_detector.headshake_events:
            for s, e in ear_detector.headshake_events:
                if s < end_frame and e > start_frame:  # Overlapping events
                    features['is_ear_detected'] = 1
                    break
        
        if head_detector and head_detector.headshake_events:
            for s, e in head_detector.headshake_events:
                if s < end_frame and e > start_frame:  # Overlapping events
                    features['is_head_detected'] = 1
                    break
    
    def _add_crisscross_features(self, features: Dict, start_frame: int, end_frame: int,
                               ear_detector: Optional[EarsDetector]):
        """Add features based on ear crisscross patterns."""
        features['ear_crisscross_count'] = 0
        features['ear_crisscross_median_amplitude'] = 0
        
        if ear_detector and ear_detector.crisscross_events:
            # Find crisscross events within the candidate window
            units_in_candidate = [
                unit for unit in ear_detector.crisscross_events
                if start_frame <= unit['frames'][0] and unit['frames'][1] <= end_frame
            ]
            
            features['ear_crisscross_count'] = len(units_in_candidate)
            
            if units_in_candidate:
                amplitudes = [abs(unit['left_amp'] - unit['right_amp']) for unit in units_in_candidate]
                features['ear_crisscross_median_amplitude'] = np.median(amplitudes)
    
    def _add_oscillatory_features(self, features: Dict, start_frame: int, end_frame: int, 
                                ear_detector: Optional[EarsDetector], head_detector: Optional[HeadDetector], instance: int):
        """Add oscillatory features for head and ear signals."""
        # Use detector signals if available (like legacy), otherwise calculate fresh
        if head_detector and hasattr(head_detector, 'signal') and head_detector.signal is not None:
            # Use head detector's processed signal (may be smoothed)
            head_signal = head_detector.signal
        else:
            head_signal = self._get_signal_for_window('head', start_frame, end_frame, instance)
        
        if ear_detector and hasattr(ear_detector, 'left_ear_distances') and ear_detector.left_ear_distances is not None:
            # Use ear detector's processed signals
            left_ear_signal = ear_detector.left_ear_distances
            right_ear_signal = ear_detector.right_ear_distances
        else:
            left_ear_signal = self._get_signal_for_window('left_ear', start_frame, end_frame, instance)  
            right_ear_signal = self._get_signal_for_window('right_ear', start_frame, end_frame, instance)
        
        # Extract oscillatory features for each signal
        for signal_name, signal_data in [('head', head_signal), ('leftear', left_ear_signal), ('rightear', right_ear_signal)]:
            osc_features = self._get_oscillatory_features_for_signal(signal_data, start_frame, end_frame)
            
            for feature_name, value in osc_features.items():
                features[f'{signal_name}_{feature_name}'] = value
    
    def _get_signal_for_window(self, signal_type: str, start_frame: int, end_frame: int, instance: int) -> np.ndarray:
        """Get signal data for a specific window."""
        try:
            if signal_type == 'head':
                return self.signal_processor.calculate_head_distance_signal(
                    start_frame, end_frame, instance, self.head_config.interpolation_method
                )
            elif signal_type == 'left_ear' or signal_type == 'right_ear':
                left_distances, right_distances = self.signal_processor.calculate_ear_distances(
                    start_frame, end_frame, instance
                )
                return left_distances if signal_type == 'left_ear' else right_distances
            else:
                return np.array([])
        except Exception:
            return np.array([])
    
    def _get_oscillatory_features_for_signal(self, signal_data: np.ndarray, start_frame: int, end_frame: int) -> Dict[str, float]:
        """Calculate oscillatory features for a signal segment."""
        # Default features
        default_features = {
            'oscillation_count': 0,
            'frequency_hz': 0,
            'median_amplitude': 0,
            'max_amplitude': 0,
            'amplitude_std_dev': 0,
            'mean_inter_cycle_interval_ms': 0,
            'std_inter_cycle_interval_ms': 0
        }
        
        # Slice signal like legacy does - always assume full-length signal
        signal_slice = signal_data[start_frame:end_frame]
        
        if len(signal_slice) < 4:
            return default_features
        
        # Lenient peak finding parameters for general movement detection
        prominence = 0.5
        distance = 2
        
        if len(signal_slice) < distance * 2:
            return default_features
        
        # Find peaks and valleys
        try:
            peaks, _ = find_peaks(signal_slice, prominence=prominence, distance=distance)
            valleys, _ = find_peaks(-signal_slice, prominence=prominence, distance=distance)
        except Exception:
            return default_features
        
        # Create oscillation events
        oscillation_events = sorted(
            [(p, 'peak') for p in peaks] + [(v, 'valley') for v in valleys]
        )
        
        # Build cycles from alternating peaks and valleys
        cycles = []
        for i in range(len(oscillation_events) - 1):
            idx1, type1 = oscillation_events[i]
            idx2, type2 = oscillation_events[i + 1]
            
            if type1 == type2:  # Skip same-type consecutive events
                continue
            
            amplitude = abs(signal_slice[idx1] - signal_slice[idx2])
            cycles.append({
                'start': idx1,
                'end': idx2,
                'amplitude': amplitude
            })
        
        if not cycles:
            return default_features
        
        # Calculate features
        features = {}
        duration_s = (end_frame - start_frame) / self.fps
        
        features['oscillation_count'] = len(cycles)
        features['frequency_hz'] = features['oscillation_count'] / duration_s if duration_s > 0 else 0
        
        amplitudes = [c['amplitude'] for c in cycles]
        features['median_amplitude'] = np.median(amplitudes)
        features['max_amplitude'] = np.max(amplitudes)
        features['amplitude_std_dev'] = np.std(amplitudes)
        
        # Inter-cycle intervals
        if len(cycles) > 1:
            cycle_starts = [c['start'] for c in cycles]
            inter_cycle_intervals = np.diff(cycle_starts) / self.fps * 1000  # Convert to ms
            features['mean_inter_cycle_interval_ms'] = np.mean(inter_cycle_intervals)
            features['std_inter_cycle_interval_ms'] = np.std(inter_cycle_intervals)
        else:
            features['mean_inter_cycle_interval_ms'] = 0
            features['std_inter_cycle_interval_ms'] = 0
        
        return features
    
    def _add_kinematic_features(self, features: Dict, start_frame: int, end_frame: int, instance: int):
        """Add kinematic features based on node movement."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Head kinematics
            head_pos = self.data_loader.get_node_positions(
                self.node_mapping.head, start_frame, end_frame, instance
            )
            head_kinematics = self._calculate_kinematics(head_pos)
            features['total_displacement_head'] = head_kinematics['total_displacement']
            features['peak_velocity_head'] = head_kinematics['peak_velocity']
            features['mean_jerk_head'] = head_kinematics['mean_jerk']
            
            # Back kinematics  
            back_pos = self.data_loader.get_node_positions(
                self.node_mapping.back, start_frame, end_frame, instance
            )
            back_kinematics = self._calculate_kinematics(back_pos)
            features['total_displacement_back'] = back_kinematics['total_displacement']
            features['peak_velocity_back'] = back_kinematics['peak_velocity']
            
            # Ratio feature
            if features['peak_velocity_head'] > 0:
                features['back_head_velocity_ratio'] = features['peak_velocity_back'] / features['peak_velocity_head']
            else:
                features['back_head_velocity_ratio'] = 0
    
    def _calculate_kinematics(self, pos_data: np.ndarray) -> Dict[str, float]:
        """Calculate kinematic features from position time series."""
        if pos_data.shape[0] < 4:
            return {'total_displacement': 0, 'peak_velocity': 0, 'mean_jerk': 0}
        
        # Calculate displacements between consecutive frames
        displacements = np.linalg.norm(np.diff(pos_data, axis=0), axis=1)
        total_displacement = np.sum(displacements)
        
        # Calculate velocity (displacement * fps)
        velocity = displacements * self.fps
        peak_velocity = np.max(velocity) if velocity.size > 0 else 0
        
        # Calculate acceleration and jerk
        acceleration = np.diff(velocity, n=1) * self.fps if velocity.size > 1 else np.array([])
        jerk = np.diff(acceleration, n=1) * self.fps if acceleration.size > 1 else np.array([])
        mean_jerk = np.mean(np.abs(jerk)) if jerk.size > 0 else 0
        
        return {
            'total_displacement': total_displacement,
            'peak_velocity': peak_velocity,
            'mean_jerk': mean_jerk
        }
    
    def _add_confidence_features(self, features: Dict, start_frame: int, end_frame: int, instance: int):
        """Add confidence score features."""
        try:
            # Get confidence scores for the window
            score_slice = self.data_loader.point_scores[start_frame:end_frame, :, instance]
            
            if score_slice.size > 0:
                median_scores = np.median(score_slice, axis=0)
                
                # Individual node confidence scores (legacy compatibility)
                features['median_score_head'] = median_scores[self.node_mapping.head]
                features['median_score_leftear'] = median_scores[self.node_mapping.left_ear]
                features['median_score_rightear'] = median_scores[self.node_mapping.right_ear]
                
                # Overall confidence metric (legacy compatibility)
                features['min_score_any_node'] = np.min(score_slice)
            else:
                # Default values if no scores available (legacy compatibility)
                features['median_score_head'] = 0
                features['median_score_leftear'] = 0
                features['median_score_rightear'] = 0
                features['min_score_any_node'] = 0
                
        except (IndexError, KeyError):
            # Handle cases where score data is malformed (legacy compatibility)
            features['median_score_head'] = 0
            features['median_score_leftear'] = 0
            features['median_score_rightear'] = 0
            features['min_score_any_node'] = 0


class BatchFeatureExtractor:
    """Handles batch feature extraction for multiple H5 files."""
    
    def __init__(self, ear_config: EarDetectorConfig, head_config: HeadDetectorConfig, 
                 node_mapping: NodeMapping, fps: int):
        self.ear_config = ear_config
        self.head_config = head_config
        self.node_mapping = node_mapping
        self.fps = fps
    
    def process_file(self, h5_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Process a single H5 file and extract features for all candidate windows."""
        print(f"Processing: {h5_path}")
        
        # Load data
        data_loader = SleapDataLoader(h5_path)
        if not data_loader.load_data():
            print(f"Error: Could not load data from {h5_path}")
            return pd.DataFrame()
        
        print(f"Loaded {data_loader.total_frames} frames with {data_loader.locations.shape[1]} nodes")
        
        # Run combined detection to get candidate windows
        combined_detector = CombinedDetector(data_loader, self.ear_config, self.head_config, self.node_mapping)
        
        print("Running ear-based detection...")
        ear_events = combined_detector.ear_detector.detect_headshakes()
        print(f"Found {len(ear_events)} ear-based events")
        
        print("Running head-based detection...")
        head_events = combined_detector.head_detector.detect_headshakes()
        print(f"Found {len(head_events)} head-based events")
        
        # Combine and merge candidate windows
        print("Merging detections into candidate pool...")
        candidate_pool = self._combine_and_merge_candidates(ear_events, head_events)
        print(f"Created {len(candidate_pool)} unique candidate windows")
        
        if not candidate_pool:
            print("No candidates found")
            return pd.DataFrame()
        
        # Extract features
        print("Extracting features...")
        feature_extractor = FeatureExtractor(data_loader, self.ear_config, self.head_config, self.node_mapping, self.fps)
        features_df = feature_extractor.extract_candidate_features(
            candidate_pool, 
            combined_detector.ear_detector, 
            combined_detector.head_detector
        )
        
        # Save results if output path provided
        if output_path:
            features_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        print(f"Feature extraction complete. Generated {len(features_df)} feature vectors.")
        return features_df
    
    def _combine_and_merge_candidates(self, ear_events: List[Tuple[int, int]], 
                                    head_events: List[Tuple[int, int]], merge_gap: int = 5) -> List[Tuple[int, int]]:
        """Combine ear and head events into unique merged candidate windows."""
        all_events = sorted(ear_events + head_events, key=lambda x: x[0])
        
        if not all_events:
            return []
        
        # Merge overlapping/close events
        merged = [all_events[0]]
        
        for next_start, next_end in all_events[1:]:
            last_start, last_end = merged[-1]
            
            if next_start <= last_end + merge_gap:
                # Merge by extending the window
                merged[-1] = (last_start, max(last_end, next_end))
            else:
                # Add as new candidate
                merged.append((next_start, next_end))
        
        return merged
    
    def process_folder(self, input_folder: str, output_folder: str) -> Dict[str, Any]:
        """Process all H5 files in a folder and save features to output folder.
        
        Args:
            input_folder: Path to folder containing H5 files
            output_folder: Path to folder where feature CSV files will be saved
            
        Returns:
            Dict with processing results: {'success': bool, 'files_processed': int, 'error': str}
        """
        import os
        import glob
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all H5 files recursively
        h5_pattern = os.path.join(input_folder, "**", "*.h5")
        h5_files = glob.glob(h5_pattern, recursive=True)
        
        if not h5_files:
            return {
                'success': False,
                'files_processed': 0,
                'error': f'No H5 files found in {input_folder}'
            }
        
        print(f"Found {len(h5_files)} H5 files to process")
        
        files_processed = 0
        failed_files = []
        
        for h5_file in h5_files:
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(h5_file))[0]
                output_path = os.path.join(output_folder, f"{base_name}_htr_features.csv")
                
                # Skip if already exists
                if os.path.exists(output_path):
                    print(f"Skipping {base_name} (already exists)")
                    files_processed += 1
                    continue
                
                # Process the file
                features_df = self.process_file(h5_file, output_path)
                
                if not features_df.empty:
                    files_processed += 1
                    print(f"✓ Processed {base_name}: {len(features_df)} features")
                else:
                    print(f"⚠ No features found in {base_name}")
                    failed_files.append(h5_file)
                    
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(h5_file)}: {str(e)}")
                failed_files.append(h5_file)
        
        success = files_processed > 0
        error_msg = f"Failed to process {len(failed_files)} files" if failed_files else None
        
        return {
            'success': success,
            'files_processed': files_processed,
            'error': error_msg
        }