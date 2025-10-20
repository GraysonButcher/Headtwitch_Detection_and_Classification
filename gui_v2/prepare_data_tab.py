"""
Prepare Data Tab - Feature Extraction and Ground Truth Labeling

Combines feature extraction from H5 files and ground truth labeling into one workflow.
Supports incremental processing and tracks labeling progress.
"""

import os
import sys
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QProgressBar, QTextEdit, QMessageBox,
    QFileDialog
)
from PySide6.QtCore import Qt, QDateTime, Signal
from PySide6.QtGui import QFont

# Import CSV editor widget
try:
    from .csv_editor_widget import CSVEditorWidget
    from .workflow_tracker import WorkflowTracker
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from csv_editor_widget import CSVEditorWidget
    from workflow_tracker import WorkflowTracker


class PrepareDataTab(QWidget):
    """Tab for preparing training data: extract features + label ground truth."""

    # Signals
    features_extracted = Signal()  # Emitted when features are extracted
    labels_updated = Signal()  # Emitted when ground truth labels are modified

    def __init__(self, parent=None, project_manager=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.workflow_tracker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Section 1: Extract Features
        self.create_extract_features_section(layout)

        # Separator
        separator = QLabel()
        separator.setStyleSheet("background-color: #dee2e6; min-height: 2px; max-height: 2px;")
        layout.addWidget(separator)

        # Section 2: Label Ground Truth
        self.create_label_ground_truth_section(layout)

        # Final stretch
        layout.addStretch()

    def create_extract_features_section(self, parent_layout):
        """Section 1: Extract Features from H5 files."""
        features_group = QGroupBox("Step 1: Extract Features from H5 Files")
        features_group.setFont(QFont("Arial", 10, QFont.Bold))
        features_layout = QVBoxLayout(features_group)
        features_layout.setContentsMargins(10, 15, 10, 10)
        features_layout.setSpacing(10)

        # Status display
        self.features_status_label = QLabel("No project loaded. Create or open a project to begin.")
        self.features_status_label.setFont(QFont("Arial", 9))
        self.features_status_label.setWordWrap(True)
        self.features_status_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 8px; border-radius: 4px; color: #6c757d;"
        )
        features_layout.addWidget(self.features_status_label)

        # Parameters selection
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setMinimumWidth(80)
        param_label.setFont(QFont("Arial", 9))
        param_layout.addWidget(param_label)

        self.features_param_edit = QLineEdit()
        self.features_param_edit.setPlaceholderText("Optional: Load parameter configuration")
        self.features_param_edit.setFont(QFont("Arial", 9))
        param_layout.addWidget(self.features_param_edit)

        param_browse_btn = QPushButton("Browse...")
        param_browse_btn.setMaximumWidth(80)
        param_browse_btn.clicked.connect(self.browse_parameters)
        param_layout.addWidget(param_browse_btn)

        features_layout.addLayout(param_layout)

        # Processing options
        options_layout = QHBoxLayout()

        self.extract_all_btn = QPushButton("📦 Extract All Features")
        self.extract_all_btn.setFont(QFont("Arial", 9))
        self.extract_all_btn.clicked.connect(lambda: self.extract_features(mode='all'))
        self.extract_all_btn.setEnabled(False)
        options_layout.addWidget(self.extract_all_btn)

        self.extract_new_btn = QPushButton("🆕 Extract New Files Only")
        self.extract_new_btn.setFont(QFont("Arial", 9))
        self.extract_new_btn.clicked.connect(lambda: self.extract_features(mode='new'))
        self.extract_new_btn.setEnabled(False)
        self.extract_new_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        options_layout.addWidget(self.extract_new_btn)

        features_layout.addLayout(options_layout)

        # Progress
        self.features_progress_text = QTextEdit()
        self.features_progress_text.setMaximumHeight(80)
        self.features_progress_text.setFont(QFont("Consolas", 8))
        self.features_progress_text.setReadOnly(True)
        self.features_progress_text.setPlaceholderText("Feature extraction progress will appear here...")
        self.features_progress_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #495057;
            }
        """)
        features_layout.addWidget(self.features_progress_text)

        parent_layout.addWidget(features_group)

    def create_label_ground_truth_section(self, parent_layout):
        """Section 2: Label Ground Truth data."""
        label_group = QGroupBox("Step 2: Label Ground Truth Data")
        label_group.setFont(QFont("Arial", 10, QFont.Bold))
        label_layout = QVBoxLayout(label_group)
        label_layout.setContentsMargins(10, 15, 10, 10)
        label_layout.setSpacing(10)

        # Instructions
        instructions = QLabel(
            "<b>Label feature files to create training data:</b> "
            "Load a feature CSV, mark HTR events (1) and non-HTR events (0), then save."
        )
        instructions.setFont(QFont("Arial", 9))
        instructions.setWordWrap(True)
        label_layout.addWidget(instructions)

        # CSV Editor Widget
        self.csv_editor = CSVEditorWidget(self)
        self.csv_editor.labels_changed.connect(self.on_labels_changed)
        self.csv_editor.progress_updated.connect(self.on_labeling_progress_updated)
        label_layout.addWidget(self.csv_editor)

        # Alternative: Open external editor
        external_layout = QHBoxLayout()
        external_label = QLabel("Or:")
        external_label.setFont(QFont("Arial", 9))
        external_layout.addWidget(external_label)

        open_folder_btn = QPushButton("📂 Open Features Folder (Edit Externally)")
        open_folder_btn.setFont(QFont("Arial", 9))
        open_folder_btn.clicked.connect(self.open_features_folder)
        external_layout.addWidget(open_folder_btn)

        external_layout.addStretch()

        label_layout.addLayout(external_layout)

        parent_layout.addWidget(label_group)

    def set_project_manager(self, project_manager):
        """Set the project manager and refresh display."""
        self.project_manager = project_manager
        self.refresh_status()

    def refresh_status(self):
        """Refresh feature extraction status display."""
        if not self.project_manager:
            self.features_status_label.setText("No project loaded. Create or open a project to begin.")
            self.extract_all_btn.setEnabled(False)
            self.extract_new_btn.setEnabled(False)
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            self.features_status_label.setText("No project loaded. Create or open a project to begin.")
            self.extract_all_btn.setEnabled(False)
            self.extract_new_btn.setEnabled(False)
            return

        # Initialize workflow tracker
        self.workflow_tracker = WorkflowTracker(project_path)

        # Get workflow status
        status = self.workflow_tracker.get_workflow_status()
        message = self.workflow_tracker.get_status_message()

        # Update status label
        self.features_status_label.setText(message)

        # Enable/disable buttons based on status
        h5_total = status['h5_files']['total']
        h5_new = status['h5_files']['new']

        if h5_total > 0:
            self.extract_all_btn.setEnabled(True)

        if h5_new > 0:
            self.extract_new_btn.setEnabled(True)
            self.extract_new_btn.setText(f"🆕 Extract New Files Only ({h5_new} new)")
        else:
            self.extract_new_btn.setEnabled(False)
            self.extract_new_btn.setText("🆕 Extract New Files Only")

    def browse_parameters(self):
        """Browse for parameter configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parameter File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.features_param_edit.setText(file_path)

    def extract_features(self, mode='all'):
        """
        Extract features from H5 files.

        Args:
            mode: 'all' to process all H5 files, 'new' to process only new files
        """
        if not self.project_manager:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Determine which files to process
        if mode == 'new':
            if not self.workflow_tracker:
                self.workflow_tracker = WorkflowTracker(project_path)

            new_h5_files, _ = self.workflow_tracker.detect_new_h5_files()
            if not new_h5_files:
                QMessageBox.information(self, "No New Files", "No new H5 files to process.")
                return

            confirm_msg = f"Extract features from {len(new_h5_files)} new H5 file(s)?"
        else:
            input_folder = os.path.join(project_path, "input")
            h5_pattern = os.path.join(input_folder, "**", "*.h5")
            all_h5_files = glob.glob(h5_pattern, recursive=True)

            if not all_h5_files:
                QMessageBox.warning(self, "No H5 Files", "No H5 files found in input/ folder.")
                return

            confirm_msg = f"Extract features from all {len(all_h5_files)} H5 file(s)?\n\n" \
                          "This will process all files, including those already processed."

        # Confirm
        reply = QMessageBox.question(
            self,
            "Extract Features",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Start extraction
        self.show_features_progress("Starting feature extraction...")

        try:
            # Import core modules
            sys.path.append(os.path.dirname(project_path))
            from core.feature_extraction import BatchFeatureExtractor
            from core.config import ConfigManager

            # Set up paths
            input_folder = os.path.join(project_path, "input")
            output_folder = os.path.join(project_path, "features")

            # Load parameters
            param_path = self.features_param_edit.text().strip()
            config_manager = ConfigManager()
            if param_path and os.path.exists(param_path):
                config_manager.import_parameters(param_path)
                self.show_features_progress(f"Using parameters from: {os.path.basename(param_path)}")
            else:
                self.show_features_progress("Using default parameters")

            # Create batch extractor
            extractor = BatchFeatureExtractor(
                config_manager.config.ear_detector,
                config_manager.config.head_detector,
                config_manager.config.node_mapping,
                config_manager.config.default_fps
            )

            # Process files
            if mode == 'new':
                # Process only new files
                self.show_features_progress(f"Processing {len(new_h5_files)} new files...")
                # TODO: Need to add method to process specific files
                # For now, process all and let it skip existing
                results = extractor.process_folder(input_folder, output_folder)
            else:
                # Process all files
                self.show_features_progress(f"Processing all files in {input_folder}...")
                results = extractor.process_folder(input_folder, output_folder)

            if results['success']:
                self.show_features_progress(f"✅ Successfully processed {results['files_processed']} files")
                self.show_features_progress(f"Features saved to: {output_folder}")

                # Refresh status
                self.refresh_status()

                # Emit signal
                self.features_extracted.emit()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Feature extraction complete!\n\n{results['files_processed']} files processed."
                )
            else:
                self.show_features_progress(f"❌ Feature extraction failed: {results.get('error', 'Unknown error')}")
                QMessageBox.warning(self, "Error", f"Feature extraction failed:\n{results.get('error', 'Unknown error')}")

        except ImportError as e:
            self.show_features_progress(f"❌ Core modules not available: {str(e)}")
            QMessageBox.critical(self, "Import Error", f"Required modules not found:\n{str(e)}")
        except Exception as e:
            self.show_features_progress(f"❌ Error during feature extraction: {str(e)}")
            QMessageBox.critical(self, "Error", f"Feature extraction failed:\n{str(e)}")

    def show_features_progress(self, message):
        """Show progress message in the features progress text area."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        self.features_progress_text.append(formatted_message)

        # Auto-scroll to bottom
        cursor = self.features_progress_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.features_progress_text.setTextCursor(cursor)

    def open_features_folder(self):
        """Open the features folder in file explorer."""
        if not self.project_manager:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        features_folder = os.path.join(project_path, "features")
        if not os.path.exists(features_folder):
            QMessageBox.information(
                self,
                "Folder Not Found",
                "Features folder does not exist yet. Extract features first."
            )
            return

        # Open in file explorer
        import subprocess
        if sys.platform == 'win32':
            subprocess.run(['explorer', features_folder], shell=True)
        elif sys.platform == 'darwin':
            subprocess.run(['open', features_folder])
        else:
            subprocess.run(['xdg-open', features_folder])

    def on_labels_changed(self):
        """Handle label changes in the CSV editor."""
        self.labels_updated.emit()

    def on_labeling_progress_updated(self, labeled_count, total_count):
        """Handle labeling progress updates."""
        # Could emit signal or update parent display
        pass
