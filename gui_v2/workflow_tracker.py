"""
Workflow Tracker - File Processing State Management

Tracks which files have been processed at each workflow stage and detects new files
that need processing. Supports incremental batch processing.
"""

import os
import glob
from typing import Dict, List, Tuple, Set
from datetime import datetime
import json


class WorkflowTracker:
    """Manages file processing state and detects workflow changes."""

    def __init__(self, project_path: str):
        """
        Initialize workflow tracker for a project.

        Args:
            project_path: Root path to the project directory
        """
        self.project_path = project_path
        self.input_folder = os.path.join(project_path, "input")
        self.features_folder = os.path.join(project_path, "features")
        self.predictions_folder = os.path.join(project_path, "predictions")
        self.reports_folder = os.path.join(project_path, "reports")

    def get_file_base_name(self, filepath: str) -> str:
        """
        Extract base name from file path (without extension or processing suffixes).

        This function removes ONLY the suffixes added during processing, not suffixes
        that are part of the original filename (like .predictions in SLEAP files).

        Examples:
            rat001.h5 -> rat001
            rat001.predictions.h5 -> rat001.predictions
            rat001_htr_features.csv -> rat001
            rat001.predictions_htr_features.csv -> rat001.predictions
            rat001_predicted.csv -> rat001
        """
        basename = os.path.basename(filepath)
        # Remove extension first
        name_no_ext = os.path.splitext(basename)[0]

        # Only remove suffixes that we ADD during processing
        # DO NOT remove patterns that might be part of the original filename!
        # Order matters - check compound suffixes first (most specific to least specific)
        processing_suffixes = [
            '_htr_features_predicted',    # Compound: prediction of feature file
            '_htr_features',              # Standard feature extraction output
            '_predicted',                 # Simple prediction output
        ]

        for suffix in processing_suffixes:
            if name_no_ext.endswith(suffix):
                name_no_ext = name_no_ext[:-len(suffix)]
                break  # Only remove one suffix

        return name_no_ext

    def scan_h5_files(self) -> List[Dict]:
        """
        Scan for all H5 files in input folder.

        Returns:
            List of dicts with file info: {path, basename, mtime}
        """
        if not os.path.exists(self.input_folder):
            return []

        h5_pattern = os.path.join(self.input_folder, "**", "*.h5")
        h5_files = glob.glob(h5_pattern, recursive=True)

        file_list = []
        for filepath in h5_files:
            file_list.append({
                'path': filepath,
                'basename': self.get_file_base_name(filepath),
                'filename': os.path.basename(filepath),
                'mtime': os.path.getmtime(filepath)
            })

        return sorted(file_list, key=lambda x: x['basename'])

    def scan_feature_files(self) -> List[Dict]:
        """
        Scan for all feature CSV files.

        Returns:
            List of dicts with file info: {path, basename, mtime}
        """
        if not os.path.exists(self.features_folder):
            return []

        feature_pattern = os.path.join(self.features_folder, "*.csv")
        feature_files = glob.glob(feature_pattern)

        file_list = []
        for filepath in feature_files:
            file_list.append({
                'path': filepath,
                'basename': self.get_file_base_name(filepath),
                'filename': os.path.basename(filepath),
                'mtime': os.path.getmtime(filepath)
            })

        return sorted(file_list, key=lambda x: x['basename'])

    def scan_prediction_files(self) -> List[Dict]:
        """
        Scan for all prediction CSV files.

        Returns:
            List of dicts with file info: {path, basename, mtime}
        """
        if not os.path.exists(self.predictions_folder):
            return []

        prediction_pattern = os.path.join(self.predictions_folder, "*.csv")
        prediction_files = glob.glob(prediction_pattern)

        file_list = []
        for filepath in prediction_files:
            file_list.append({
                'path': filepath,
                'basename': self.get_file_base_name(filepath),
                'filename': os.path.basename(filepath),
                'mtime': os.path.getmtime(filepath)
            })

        return sorted(file_list, key=lambda x: x['basename'])

    def detect_new_h5_files(self) -> Tuple[List[str], List[str]]:
        """
        Detect H5 files that don't have corresponding feature files.

        Returns:
            Tuple of (new_h5_files, already_processed_h5_files)
        """
        h5_files = self.scan_h5_files()
        feature_files = self.scan_feature_files()

        # Create set of feature basenames
        feature_basenames = {f['basename'] for f in feature_files}

        new_files = []
        processed_files = []

        for h5 in h5_files:
            if h5['basename'] in feature_basenames:
                processed_files.append(h5['path'])
            else:
                new_files.append(h5['path'])

        return new_files, processed_files

    def detect_unpredicted_features(self) -> Tuple[List[str], List[str]]:
        """
        Detect feature files that don't have corresponding prediction files.

        Returns:
            Tuple of (unpredicted_features, already_predicted_features)
        """
        feature_files = self.scan_feature_files()
        prediction_files = self.scan_prediction_files()

        # Create set of prediction basenames
        prediction_basenames = {p['basename'] for p in prediction_files}

        unpredicted = []
        predicted = []

        for feature in feature_files:
            if feature['basename'] in prediction_basenames:
                predicted.append(feature['path'])
            else:
                unpredicted.append(feature['path'])

        return unpredicted, predicted

    def get_workflow_status(self) -> Dict:
        """
        Get complete workflow status for the project.

        Returns:
            Dict with workflow status information
        """
        h5_files = self.scan_h5_files()
        feature_files = self.scan_feature_files()
        prediction_files = self.scan_prediction_files()

        new_h5, processed_h5 = self.detect_new_h5_files()
        unpredicted_features, predicted_features = self.detect_unpredicted_features()

        # Check for reports
        report_files = []
        if os.path.exists(self.reports_folder):
            report_pattern = os.path.join(self.reports_folder, "*.xlsx")
            report_files = glob.glob(report_pattern)

        status = {
            'h5_files': {
                'total': len(h5_files),
                'new': len(new_h5),
                'processed': len(processed_h5),
                'new_files': new_h5,
                'processed_files': processed_h5
            },
            'features': {
                'total': len(feature_files),
                'unpredicted': len(unpredicted_features),
                'predicted': len(predicted_features),
                'unpredicted_files': unpredicted_features,
                'predicted_files': predicted_features
            },
            'predictions': {
                'total': len(prediction_files)
            },
            'reports': {
                'total': len(report_files),
                'latest': max(report_files, key=os.path.getmtime) if report_files else None
            }
        }

        return status

    def get_processing_recommendation(self) -> str:
        """
        Analyze workflow state and recommend next action.

        Returns:
            Recommendation string
        """
        status = self.get_workflow_status()

        h5_total = status['h5_files']['total']
        h5_new = status['h5_files']['new']
        h5_processed = status['h5_files']['processed']

        features_total = status['features']['total']
        features_unpredicted = status['features']['unpredicted']

        predictions_total = status['predictions']['total']

        # No H5 files
        if h5_total == 0:
            return "no_h5_files"

        # All files are new (fresh project)
        if h5_new == h5_total and features_total == 0:
            return "fresh_batch"

        # Some new H5 files detected
        if h5_new > 0:
            return "incremental_extract"

        # All features extracted but some unpredicted
        if features_unpredicted > 0:
            return "incremental_predict"

        # Everything processed, just need report
        if predictions_total > 0 and status['reports']['total'] == 0:
            return "generate_report"

        # Everything complete, offer reprocess
        if predictions_total > 0 and status['reports']['total'] > 0:
            return "complete_offer_reprocess"

        # Default
        return "ready"

    def get_status_message(self) -> str:
        """
        Get human-readable status message based on workflow state.

        Returns:
            Status message string with workflow guidance
        """
        recommendation = self.get_processing_recommendation()
        status = self.get_workflow_status()

        if recommendation == "no_h5_files":
            return "📂 No H5 files found. Add SLEAP tracking files to the input/ folder to begin."

        elif recommendation == "fresh_batch":
            h5_count = status['h5_files']['total']
            return f"✨ Fresh project ready! Found {h5_count} H5 file(s). Click 'Run Full Pipeline' to process all files."

        elif recommendation == "incremental_extract":
            new_count = status['h5_files']['new']
            processed_count = status['h5_files']['processed']
            return f"🆕 {new_count} new H5 file(s) detected ({processed_count} already processed). Extract features for new files?"

        elif recommendation == "incremental_predict":
            unpredicted = status['features']['unpredicted']
            return f"⚙️ {unpredicted} feature file(s) need predictions. Run prediction step to continue."

        elif recommendation == "generate_report":
            predictions = status['predictions']['total']
            return f"📊 {predictions} prediction(s) complete. Generate report to compile results."

        elif recommendation == "complete_offer_reprocess":
            total = status['predictions']['total']
            return f"✅ Analysis complete for {total} file(s). Add new H5 files or reprocess with different settings."

        else:
            return "Ready for processing"
