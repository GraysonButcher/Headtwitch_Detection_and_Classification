"""
Configuration management for HTR Analysis Tool.
Provides centralized parameter management with JSON persistence.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class EarDetectorConfig:
    """Configuration parameters for ear-based headshake detection."""
    peak_threshold: int = 30
    valley_threshold: int = 30
    max_gap: int = 2
    quick_gap: int = 5
    min_crisscrosses: int = 5
    between_unit_gap: int = 15
    merge_gap: int = 10
    apply_median_score_filter: bool = True
    median_score_threshold: float = 0.6


@dataclass
class HeadDetectorConfig:
    """Configuration parameters for head-based headshake detection."""
    interpolation_method: str = 'linear'
    min_oscillations: int = 4
    amplitude_threshold: int = 10
    amplitude_median: int = 15
    median_score_threshold: float = 0.6
    peak_prominence: int = 2
    peak_distance: int = 3
    use_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_polyorder: int = 2
    min_cycle_duration: int = 2
    max_cycle_duration: int = 5
    max_cycle_gap: int = 6


@dataclass
class NodeMapping:
    """SLEAP node index mapping for different export formats."""
    left_ear: int = 0
    right_ear: int = 1
    back: int = 2
    nose: int = 3
    head: int = 4


@dataclass
class AppConfig:
    """Main application configuration."""
    ear_detector: EarDetectorConfig
    head_detector: HeadDetectorConfig
    node_mapping: NodeMapping
    recent_files: list
    default_fps: int = 160
    iou_threshold: float = 0.1
    
    def __init__(self):
        self.ear_detector = EarDetectorConfig()
        self.head_detector = HeadDetectorConfig()
        self.node_mapping = NodeMapping()
        self.recent_files = []


class ConfigManager:
    """Manages application configuration with JSON persistence."""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = os.path.expanduser("~/.htr_analysis_tool")
        
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "config.json")
        self.config = AppConfig()
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Load existing config if available
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Update config object with loaded data
                if 'ear_detector' in data:
                    for key, value in data['ear_detector'].items():
                        if hasattr(self.config.ear_detector, key):
                            setattr(self.config.ear_detector, key, value)
                
                if 'head_detector' in data:
                    for key, value in data['head_detector'].items():
                        if hasattr(self.config.head_detector, key):
                            setattr(self.config.head_detector, key, value)
                
                if 'node_mapping' in data:
                    for key, value in data['node_mapping'].items():
                        if hasattr(self.config.node_mapping, key):
                            setattr(self.config.node_mapping, key, value)
                
                if 'recent_files' in data:
                    self.config.recent_files = data['recent_files'][:10]  # Keep max 10
                
                if 'default_fps' in data:
                    self.config.default_fps = data['default_fps']
                    
                if 'iou_threshold' in data:
                    self.config.iou_threshold = data['iou_threshold']
                
                return True
        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load config file: {e}")
            return False
        
        return False
    
    def save_config(self) -> bool:
        """Save configuration to JSON file."""
        try:
            config_data = {
                'ear_detector': asdict(self.config.ear_detector),
                'head_detector': asdict(self.config.head_detector),
                'node_mapping': asdict(self.config.node_mapping),
                'recent_files': self.config.recent_files,
                'default_fps': self.config.default_fps,
                'iou_threshold': self.config.iou_threshold
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error: Could not save config file: {e}")
            return False
    
    def export_parameters(self, file_path: str) -> bool:
        """Export detector parameters to a file."""
        try:
            params = {
                'ear_detector': asdict(self.config.ear_detector),
                'head_detector': asdict(self.config.head_detector),
                'node_mapping': asdict(self.config.node_mapping),
                'default_fps': self.config.default_fps,
                'iou_threshold': self.config.iou_threshold
            }
            
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error: Could not export parameters: {e}")
            return False
    
    def import_parameters(self, file_path: str) -> bool:
        """Import detector parameters from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Update detector configs
            if 'ear_detector' in data:
                for key, value in data['ear_detector'].items():
                    if hasattr(self.config.ear_detector, key):
                        setattr(self.config.ear_detector, key, value)
            
            if 'head_detector' in data:
                for key, value in data['head_detector'].items():
                    if hasattr(self.config.head_detector, key):
                        setattr(self.config.head_detector, key, value)
            
            if 'node_mapping' in data:
                for key, value in data['node_mapping'].items():
                    if hasattr(self.config.node_mapping, key):
                        setattr(self.config.node_mapping, key, value)
                        
            if 'default_fps' in data:
                self.config.default_fps = data['default_fps']
                
            if 'iou_threshold' in data:
                self.config.iou_threshold = data['iou_threshold']
            
            # Save the updated config
            self.save_config()
            return True
        
        except Exception as e:
            print(f"Error: Could not import parameters: {e}")
            return False
    
    def add_recent_file(self, file_path: str) -> None:
        """Add a file to the recent files list."""
        if file_path in self.config.recent_files:
            self.config.recent_files.remove(file_path)
        
        self.config.recent_files.insert(0, file_path)
        self.config.recent_files = self.config.recent_files[:10]  # Keep max 10
        self.save_config()
    
    def get_ear_config_dict(self) -> Dict[str, Any]:
        """Get ear detector config as dictionary (for legacy compatibility)."""
        config_dict = asdict(self.config.ear_detector)
        config_dict['fps'] = self.config.default_fps  # Override with app-level FPS
        return config_dict
    
    def get_head_config_dict(self) -> Dict[str, Any]:
        """Get head detector config as dictionary (for legacy compatibility)."""
        return asdict(self.config.head_detector)
    
    def get_node_indices(self) -> Dict[str, int]:
        """Get node indices as dictionary (for legacy compatibility)."""
        return asdict(self.config.node_mapping)


# Global config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager