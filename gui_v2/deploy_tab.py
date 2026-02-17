"""
Deploy Tab - Smart Batch Processing with Incremental Support

Handles both fresh batch processing and incremental updates to existing projects.
Automatically detects new files and offers appropriate processing options.
"""

import os
import sys
import glob
import shutil
import subprocess
import pandas as pd
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QProgressBar, QTextEdit, QMessageBox,
    QFileDialog, QRadioButton, QButtonGroup, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox
)
from PySide6.QtCore import Qt, QDateTime, Signal
from PySide6.QtGui import QFont

# Import workflow tracker and metadata dialog
try:
    from .workflow_tracker import WorkflowTracker
    from .metadata_config_dialog import MetadataConfigDialog
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from workflow_tracker import WorkflowTracker
    from metadata_config_dialog import MetadataConfigDialog


class DeployTab(QWidget):
    """Tab for deploying trained models to process data in production."""

    # Signals
    processing_complete = Signal()  # Emitted when processing completes

    def __init__(self, parent=None, project_manager=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.workflow_tracker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Title and instructions
        self.create_instructions_section(layout)

        # Project status
        self.create_status_section(layout)

        # Input data management
        self.create_input_data_section(layout)

        # Model and parameters configuration
        self.create_config_section(layout)

        # Processing mode selection
        self.create_mode_selection_section(layout)

        # Processing buttons
        self.create_processing_buttons_section(layout)

        # Results viewing
        self.create_results_section(layout)

        # Progress section
        self.create_progress_section(layout)

        # Final stretch
        layout.addStretch()

    def create_instructions_section(self, parent_layout):
        """Instructions header."""
        instructions_group = QGroupBox("Identify Headtwitches")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_layout.setContentsMargins(10, 10, 10, 10)

        instructions = QLabel(
            "<b>Use trained models to identify head-twitch responses in your data:</b> "
            "Select a model, choose processing mode (fresh or incremental), and extract features or predict HTRs."
        )
        instructions.setFont(QFont("Arial", 10))
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)

        parent_layout.addWidget(instructions_group)

    def create_status_section(self, parent_layout):
        """Project and workflow status display."""
        status_group = QGroupBox("Project Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(10, 10, 10, 10)

        self.status_label = QLabel("No project loaded. Create or open a project to begin.")
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 8px; border-radius: 4px; color: #6c757d;"
        )
        status_layout.addWidget(self.status_label)

        # File counts
        self.file_counts_label = QLabel("")
        self.file_counts_label.setFont(QFont("Arial", 9))
        status_layout.addWidget(self.file_counts_label)

        parent_layout.addWidget(status_group)

    def create_input_data_section(self, parent_layout):
        """Input data management section."""
        input_group = QGroupBox("Input Data")
        input_layout = QHBoxLayout(input_group)
        input_layout.setContentsMargins(10, 10, 10, 10)

        # Add H5 files button
        self.add_h5_btn = QPushButton("📁 Add H5 Files...")
        self.add_h5_btn.setFont(QFont("Arial", 9))
        self.add_h5_btn.setToolTip("Browse and add H5 tracking files to project/input/ folder")
        self.add_h5_btn.clicked.connect(self.add_h5_files)
        self.add_h5_btn.setEnabled(False)
        input_layout.addWidget(self.add_h5_btn)

        # Open input folder button
        self.open_input_btn = QPushButton("📂 Open Input Folder")
        self.open_input_btn.setFont(QFont("Arial", 9))
        self.open_input_btn.setToolTip("Open the project input folder in file explorer")
        self.open_input_btn.clicked.connect(self.open_input_folder)
        self.open_input_btn.setEnabled(False)
        input_layout.addWidget(self.open_input_btn)

        input_layout.addStretch()

        parent_layout.addWidget(input_group)

    def create_config_section(self, parent_layout):
        """Model and parameters configuration."""
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(8)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setMinimumWidth(80)
        model_label.setFont(QFont("Arial", 9))
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Arial", 9))
        self.model_combo.setEditable(False)
        model_layout.addWidget(self.model_combo)

        self.model_browse_btn = QPushButton("Browse...")
        self.model_browse_btn.setMaximumWidth(100)
        self.model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_browse_btn)

        config_layout.addLayout(model_layout)

        # Parameters selection
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setMinimumWidth(80)
        param_label.setFont(QFont("Arial", 9))
        param_layout.addWidget(param_label)

        self.param_path_edit = QLineEdit()
        self.param_path_edit.setPlaceholderText("Optional: Load parameter configuration")
        self.param_path_edit.setFont(QFont("Arial", 9))
        self.param_path_edit.setReadOnly(True)
        param_layout.addWidget(self.param_path_edit)

        self.param_browse_btn = QPushButton("Browse...")
        self.param_browse_btn.setMaximumWidth(100)
        self.param_browse_btn.clicked.connect(self.browse_parameters)
        param_layout.addWidget(self.param_browse_btn)

        config_layout.addLayout(param_layout)

        # Ear detection mode indicator
        self.ear_mode_label = QLabel("Ear Detection: Absolute Mode")
        self.ear_mode_label.setFont(QFont("Arial", 8))
        self.ear_mode_label.setStyleSheet("color: #666; padding: 2px 0;")
        config_layout.addWidget(self.ear_mode_label)

        parent_layout.addWidget(config_group)

    def create_mode_selection_section(self, parent_layout):
        """Processing mode selection (fresh vs incremental)."""
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(10, 5, 10, 5)  # Minimal top/bottom margins
        mode_layout.setSpacing(2)  # Minimal spacing

        self.mode_button_group = QButtonGroup(self)

        # Fresh batch mode
        self.fresh_mode_radio = QRadioButton("Fresh Batch (Process all files)")
        self.fresh_mode_radio.setFont(QFont("Arial", 9))
        self.mode_button_group.addButton(self.fresh_mode_radio, 0)
        mode_layout.addWidget(self.fresh_mode_radio)

        fresh_help = QLabel("   → Extract features, predict, and generate report for all H5 files")
        fresh_help.setFont(QFont("Arial", 8))
        fresh_help.setStyleSheet("color: #6c757d;")
        fresh_help.setWordWrap(True)
        mode_layout.addWidget(fresh_help)

        mode_layout.addSpacing(6)  # Small gap between options

        # Incremental mode
        self.incremental_mode_radio = QRadioButton("Incremental (Process only new files)")
        self.incremental_mode_radio.setFont(QFont("Arial", 9))
        self.mode_button_group.addButton(self.incremental_mode_radio, 1)
        mode_layout.addWidget(self.incremental_mode_radio)

        incremental_help = QLabel("   → Process only new H5 files added since last run, update existing report")
        incremental_help.setFont(QFont("Arial", 8))
        incremental_help.setStyleSheet("color: #6c757d;")
        incremental_help.setWordWrap(True)
        mode_layout.addWidget(incremental_help)

        # Default selection
        self.fresh_mode_radio.setChecked(True)

        parent_layout.addWidget(mode_group)

    def create_processing_buttons_section(self, parent_layout):
        """Processing control buttons."""
        buttons_layout = QHBoxLayout()

        # Processing step buttons
        self.step1_btn = QPushButton("1️⃣ Extract Features")
        self.step1_btn.setFont(QFont("Arial", 9))
        self.step1_btn.clicked.connect(self.run_extract_step)
        self.step1_btn.setEnabled(False)
        buttons_layout.addWidget(self.step1_btn)

        self.step2_btn = QPushButton("2️⃣ Predict HTRs")
        self.step2_btn.setFont(QFont("Arial", 9))
        self.step2_btn.clicked.connect(self.run_predict_step)
        self.step2_btn.setEnabled(False)
        buttons_layout.addWidget(self.step2_btn)

        buttons_layout.addStretch()

        parent_layout.addLayout(buttons_layout)

    def create_results_section(self, parent_layout):
        """Results viewing section."""
        results_group = QGroupBox("Results")
        results_main_layout = QVBoxLayout(results_group)
        results_main_layout.setContentsMargins(10, 10, 10, 10)
        results_main_layout.setSpacing(6)

        # Run selector row
        run_selector_layout = QHBoxLayout()

        run_label = QLabel("Prediction Run:")
        run_label.setFont(QFont("Arial", 9))
        run_selector_layout.addWidget(run_label)

        self.run_combo = QComboBox()
        self.run_combo.setFont(QFont("Arial", 9))
        self.run_combo.setMinimumWidth(300)
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        run_selector_layout.addWidget(self.run_combo)

        self.compare_runs_btn = QPushButton("Compare Runs")
        self.compare_runs_btn.setFont(QFont("Arial", 9))
        self.compare_runs_btn.setToolTip("Compare HTR counts between two prediction runs")
        self.compare_runs_btn.clicked.connect(self._show_comparison_dialog)
        self.compare_runs_btn.setEnabled(False)
        run_selector_layout.addWidget(self.compare_runs_btn)

        run_selector_layout.addStretch()
        results_main_layout.addLayout(run_selector_layout)

        # Buttons row
        results_layout = QHBoxLayout()

        # Open predictions folder button
        self.open_predictions_btn = QPushButton("📁 Open Predictions Folder")
        self.open_predictions_btn.setFont(QFont("Arial", 9))
        self.open_predictions_btn.setToolTip("Open the predictions folder in file explorer")
        self.open_predictions_btn.clicked.connect(self.open_predictions_folder)
        self.open_predictions_btn.setEnabled(False)
        results_layout.addWidget(self.open_predictions_btn)

        # View results summary button
        self.view_summary_btn = QPushButton("📊 View Results Summary")
        self.view_summary_btn.setFont(QFont("Arial", 9))
        self.view_summary_btn.setToolTip("View a summary of HTR counts for each file")
        self.view_summary_btn.clicked.connect(self.view_results_summary)
        self.view_summary_btn.setEnabled(False)
        results_layout.addWidget(self.view_summary_btn)

        results_layout.addStretch()
        results_main_layout.addLayout(results_layout)

        parent_layout.addWidget(results_group)

    def create_progress_section(self, parent_layout):
        """Progress tracking and logging."""
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(6)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        # Progress text
        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(120)
        self.progress_text.setFont(QFont("Consolas", 8))
        self.progress_text.setReadOnly(True)
        self.progress_text.setPlaceholderText("Processing progress will appear here...")
        self.progress_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #495057;
            }
        """)
        progress_layout.addWidget(self.progress_text)

        parent_layout.addWidget(progress_group)

    def set_project_manager(self, project_manager):
        """Set project manager and refresh status."""
        self.project_manager = project_manager
        self.refresh_status()

    def refresh_status(self):
        """Refresh deployment status display."""
        if not self.project_manager:
            self.status_label.setText("No project loaded. Create or open a project to begin.")
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            self.status_label.setText("No project loaded. Create or open a project to begin.")
            return

        # Initialize workflow tracker with saved metadata config
        metadata_config = self.project_manager.get_metadata_config()
        self.workflow_tracker = WorkflowTracker(project_path, metadata_config=metadata_config)

        # Get workflow status
        status = self.workflow_tracker.get_workflow_status()
        message = self.workflow_tracker.get_status_message()
        recommendation = self.workflow_tracker.get_processing_recommendation()

        # Update status label
        self.status_label.setText(message)

        # Auto-detect and load existing models and parameters
        self._auto_load_models_and_params(project_path)

        # Populate the run selector dropdown
        self._populate_run_selector()

        # Update file counts
        h5_total = status['h5_files']['total']
        h5_new = status['h5_files']['new']
        h5_processed = status['h5_files']['processed']
        features_total = status['features']['total']
        predictions_total = status['predictions']['total']

        # Show prediction count from the active run if available
        active_run_path = self._get_active_predictions_path()
        if active_run_path and os.path.exists(active_run_path):
            active_csvs = glob.glob(os.path.join(active_run_path, "*.csv"))
            counts_text = f"📊 Files: {h5_total} H5 | {features_total} Features | {len(active_csvs)} Predictions (active run)"
        else:
            counts_text = f"📊 Files: {h5_total} H5 | {features_total} Features | {predictions_total} Predictions"
        if h5_new > 0:
            counts_text += f" | 🆕 {h5_new} New"

        self.file_counts_label.setText(counts_text)

        # Auto-select processing mode
        if recommendation == "fresh_batch":
            self.fresh_mode_radio.setChecked(True)
        elif recommendation in ["incremental_extract", "incremental_predict"]:
            self.incremental_mode_radio.setChecked(True)

        # Enable/disable buttons
        model_path = self._get_selected_model_path()
        has_model = bool(model_path and os.path.exists(model_path))

        # Input data buttons
        self.add_h5_btn.setEnabled(bool(project_path))
        self.open_input_btn.setEnabled(bool(project_path))

        # Processing buttons
        if h5_total > 0 and has_model:
            self.step1_btn.setEnabled(True)
            self.step2_btn.setEnabled(features_total > 0)
        else:
            self.step1_btn.setEnabled(False)
            self.step2_btn.setEnabled(False)

        # Results viewing buttons
        self.open_predictions_btn.setEnabled(predictions_total > 0)
        self.view_summary_btn.setEnabled(predictions_total > 0)

    def _get_selected_model_path(self):
        """Get the currently selected model path from combo box."""
        current_data = self.model_combo.currentData()
        return current_data if current_data else ""

    def _auto_load_models_and_params(self, project_path):
        """Auto-detect and load existing models and parameters from project folders."""
        if not project_path:
            self.model_combo.clear()
            self.model_combo.addItem("No model selected", "")
            self.param_path_edit.clear()
            self.model_browse_btn.setText("Browse...")
            self.param_browse_btn.setText("Browse...")
            return

        # Auto-load models
        models_folder = os.path.join(project_path, "models")
        model_files = []
        if os.path.exists(models_folder):
            model_files = sorted(
                glob.glob(os.path.join(models_folder, "*.joblib")),
                key=os.path.getmtime,
                reverse=True  # Most recent first
            )

        self.model_combo.clear()
        if model_files:
            # Add all found models to dropdown
            for model_file in model_files:
                filename = os.path.basename(model_file)
                # Add checkmark prefix to show it's from project
                self.model_combo.addItem(f"✓ {filename}", model_file)

            # Update button text to "Replace..."
            self.model_browse_btn.setText("Replace...")
            self.show_progress(f"✓ Auto-loaded {len(model_files)} model(s) from project")
        else:
            self.model_combo.addItem("No model selected", "")
            self.model_browse_btn.setText("Browse...")

        # Auto-load most recent parameter file
        params_folder = os.path.join(project_path, "parameters")
        param_files = []
        if os.path.exists(params_folder):
            param_files = sorted(
                glob.glob(os.path.join(params_folder, "*.json")),
                key=os.path.getmtime,
                reverse=True  # Most recent first
            )

        if param_files:
            # Use most recent parameter file
            most_recent_param = param_files[0]
            filename = os.path.basename(most_recent_param)
            # Add checkmark prefix
            self.param_path_edit.setText(f"✓ {filename}")
            self.param_path_edit.setProperty("full_path", most_recent_param)
            self.param_path_edit.setStyleSheet("QLineEdit { background-color: #e8f5e9; }")
            self.param_browse_btn.setText("Replace...")
            self.show_progress(f"✓ Auto-loaded parameter file: {filename}")
            self._update_ear_mode_label()
        else:
            self.param_path_edit.clear()
            self.param_path_edit.setProperty("full_path", "")
            self.param_path_edit.setStyleSheet("")
            self.param_browse_btn.setText("Browse...")
            self._update_ear_mode_label()

    def browse_model(self):
        """Browse for model file and copy to project."""
        # Check if we're replacing an existing model
        current_model = self._get_selected_model_path()
        if current_model:
            reply = QMessageBox.question(
                self,
                "Replace Model",
                "Replace the current model with a different one?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.joblib);;All Files (*)"
        )
        if file_path:
            # Copy model to project/models/ folder
            if self.project_manager:
                project_path, _ = self.project_manager.get_current_project()
                if project_path:
                    models_folder = os.path.join(project_path, "models")
                    os.makedirs(models_folder, exist_ok=True)

                    # Copy file to models folder
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(models_folder, filename)

                    # Check if file already exists in project
                    if os.path.exists(dest_path) and os.path.samefile(file_path, dest_path):
                        QMessageBox.information(
                            self,
                            "Model Already in Project",
                            f"{filename} is already in this project.\n\nNo need to copy."
                        )
                        return

                    try:
                        shutil.copy2(file_path, dest_path)
                        self.show_progress(f"✓ Model copied to project: {filename}")
                        QMessageBox.information(
                            self,
                            "Model Added",
                            f"Model file copied to project:\n{filename}"
                        )
                        # Refresh to update dropdown
                        self.refresh_status()
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Copy Failed",
                            f"Could not copy model to project:\n{str(e)}"
                        )
                else:
                    QMessageBox.warning(self, "Error", "No project loaded.")
            else:
                QMessageBox.warning(self, "Error", "No project loaded.")

    def browse_parameters(self):
        """Browse for parameter file and copy to project."""
        # Check if we're replacing existing parameters
        current_param = self.param_path_edit.property("full_path")
        if current_param:
            reply = QMessageBox.question(
                self,
                "Replace Parameters",
                "Replace the current parameter file with a different one?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parameter File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            # Copy parameters to project/parameters/ folder
            if self.project_manager:
                project_path, _ = self.project_manager.get_current_project()
                if project_path:
                    params_folder = os.path.join(project_path, "parameters")
                    os.makedirs(params_folder, exist_ok=True)

                    # Copy file to parameters folder
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(params_folder, filename)

                    # Check if file already exists in project
                    if os.path.exists(dest_path) and os.path.samefile(file_path, dest_path):
                        QMessageBox.information(
                            self,
                            "Parameters Already in Project",
                            f"{filename} is already in this project.\n\nNo need to copy."
                        )
                        return

                    try:
                        shutil.copy2(file_path, dest_path)
                        self.show_progress(f"✓ Parameters copied to project: {filename}")
                        QMessageBox.information(
                            self,
                            "Parameters Added",
                            f"Parameter file copied to project:\n{filename}"
                        )
                        # Refresh to update display
                        self.refresh_status()
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Copy Failed",
                            f"Could not copy parameters to project:\n{str(e)}"
                        )
                else:
                    QMessageBox.warning(self, "Error", "No project loaded.")
            else:
                QMessageBox.warning(self, "Error", "No project loaded.")

    def run_full_pipeline(self):
        """Run the complete 3-step pipeline."""
        if not self.validate_prerequisites():
            return

        mode = "incremental" if self.incremental_mode_radio.isChecked() else "fresh"

        # Confirm action
        project_path, _ = self.project_manager.get_current_project()
        status = self.workflow_tracker.get_workflow_status()

        if mode == "incremental":
            h5_count = status['h5_files']['new']
            confirm_msg = f"Run incremental pipeline for {h5_count} new H5 file(s)?\n\n" \
                          "This will:\n" \
                          "1. Extract features for new files\n" \
                          "2. Predict HTRs for new files\n" \
                          "3. Update existing report"
        else:
            h5_count = status['h5_files']['total']
            confirm_msg = f"Run full pipeline for all {h5_count} H5 file(s)?\n\n" \
                          "This will:\n" \
                          "1. Extract features (all files)\n" \
                          "2. Predict HTRs (all files)\n" \
                          "3. Generate new report"

        reply = QMessageBox.question(
            self,
            "Run Pipeline",
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Load saved metadata config or configure
        metadata_config = self.project_manager.get_metadata_config()
        if metadata_config is not None:
            reply = QMessageBox.question(
                self,
                "Metadata Configuration",
                "Use existing metadata configuration?\n\nClick 'No' to reconfigure.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.No:
                metadata_config = None  # Fall through to dialog

        if metadata_config is None:
            input_folder = os.path.join(project_path, "input")
            metadata_dialog = MetadataConfigDialog(input_folder, self)
            if metadata_dialog.exec() != MetadataConfigDialog.Accepted:
                return  # User cancelled
            metadata_config = metadata_dialog.get_config()
            self.project_manager.save_metadata_config(metadata_config)

        # Run pipeline
        self.show_progress("🚀 Starting full pipeline...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        try:
            # Step 1: Extract features
            self.show_progress("📦 Step 1/3: Extracting features...")
            extract_success = self._extract_features_internal(mode, metadata_config)

            if not extract_success:
                self.show_progress("❌ Pipeline failed at feature extraction")
                self.progress_bar.setVisible(False)
                return

            # Step 2: Predict HTRs
            self.show_progress("🤖 Step 2/3: Predicting HTRs...")
            predict_success = self._predict_htrs_internal(mode)

            if not predict_success:
                self.show_progress("❌ Pipeline failed at HTR prediction")
                self.progress_bar.setVisible(False)
                return

            # Step 3: Generate report
            self.show_progress("📊 Step 3/3: Generating report...")
            report_success = self._generate_report_internal()

            if not report_success:
                self.show_progress("❌ Pipeline failed at report generation")
                self.progress_bar.setVisible(False)
                return

            # Success
            self.show_progress("✅ Pipeline completed successfully!")
            self.progress_bar.setVisible(False)

            # Refresh status
            self.refresh_status()

            # Emit signal
            self.processing_complete.emit()

            QMessageBox.information(
                self,
                "Pipeline Complete",
                "Full pipeline completed successfully!\n\nCheck the reports/ folder for results."
            )

        except Exception as e:
            self.show_progress(f"❌ Pipeline error: {str(e)}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Pipeline failed:\n{str(e)}")

    def run_extract_step(self):
        """Run only the feature extraction step."""
        if not self.validate_prerequisites(require_model=False):
            return

        # Get project path for input folder
        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Load saved metadata config or configure
        metadata_config = self.project_manager.get_metadata_config()
        if metadata_config is not None:
            reply = QMessageBox.question(
                self,
                "Metadata Configuration",
                "Use existing metadata configuration?\n\nClick 'No' to reconfigure.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.No:
                metadata_config = None  # Fall through to dialog

        if metadata_config is None:
            input_folder = os.path.join(project_path, "input")
            metadata_dialog = MetadataConfigDialog(input_folder, self)
            if metadata_dialog.exec() != MetadataConfigDialog.Accepted:
                return  # User cancelled
            metadata_config = metadata_dialog.get_config()
            self.project_manager.save_metadata_config(metadata_config)

        mode = "incremental" if self.incremental_mode_radio.isChecked() else "fresh"
        success = self._extract_features_internal(mode, metadata_config)

        if success:
            self.refresh_status()
            QMessageBox.information(self, "Success", "Feature extraction complete!")

    def run_predict_step(self):
        """Run only the prediction step."""
        if not self.validate_prerequisites():
            return

        mode = "incremental" if self.incremental_mode_radio.isChecked() else "fresh"
        success = self._predict_htrs_internal(mode)

        if success:
            self.refresh_status()
            QMessageBox.information(self, "Success", "HTR prediction complete!")

    def run_report_step(self):
        """Run only the report generation step."""
        if not self.validate_prerequisites(require_model=False):
            return

        success = self._generate_report_internal()

        if success:
            self.refresh_status()
            QMessageBox.information(self, "Success", "Report generated successfully!")

    def validate_prerequisites(self, require_model=True):
        """Validate prerequisites for processing."""
        if not self.project_manager:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return False

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return False

        if require_model:
            model_path = self._get_selected_model_path()
            if not model_path:
                QMessageBox.warning(self, "Error", "Please select a trained model file.")
                return False

            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Error", f"Model file not found: {model_path}")
                return False

        return True

    def _extract_features_internal(self, mode='fresh', metadata_config=None):
        """Internal method to extract features.

        Args:
            mode: 'fresh' to process all files, 'incremental' to process only new files
            metadata_config: Optional MetadataConfig for extracting metadata from file paths
        """
        try:
            project_path, _ = self.project_manager.get_current_project()

            # Import core modules
            sys.path.insert(0, os.path.dirname(project_path))
            from core.feature_extraction import BatchFeatureExtractor
            from core.config import get_config_manager

            # Get config
            config_manager = get_config_manager()

            # Load custom parameters if provided
            param_path = self.param_path_edit.property("full_path")
            if param_path and os.path.exists(param_path):
                config_manager.import_parameters(param_path)
                self.show_progress(f"Using custom parameters: {os.path.basename(param_path)}")
            else:
                self.show_progress("Using default parameters")

            # Set up paths
            input_folder = os.path.join(project_path, "input")
            features_folder = os.path.join(project_path, "features")
            os.makedirs(features_folder, exist_ok=True)

            # Log metadata configuration
            if metadata_config:
                self.show_progress(f"Metadata extraction configured: {list(metadata_config.folder_mappings.values())}")
            else:
                self.show_progress("No metadata extraction (using original filenames)")

            # Determine files to process based on mode
            if mode == 'incremental':
                new_h5_files, _ = self.workflow_tracker.detect_new_h5_files()
                if not new_h5_files:
                    self.show_progress("No new H5 files to process")
                    return True
                h5_files_to_process = new_h5_files  # Already full paths from detect_new_h5_files()
                self.show_progress(f"Processing {len(h5_files_to_process)} new H5 files...")
            else:
                # Fresh mode - process all
                h5_files_to_process = glob.glob(os.path.join(input_folder, "**", "*.h5"), recursive=True)
                self.show_progress(f"Processing all {len(h5_files_to_process)} H5 files...")

            # Create batch extractor with optional metadata config
            batch_extractor = BatchFeatureExtractor(
                ear_config=config_manager.config.ear_detector,
                head_config=config_manager.config.head_detector,
                node_mapping=config_manager.config.node_mapping,
                fps=config_manager.config.default_fps,
                metadata_config=metadata_config
            )

            # Process files
            files_processed = 0
            for h5_file in h5_files_to_process:
                try:
                    # Use extractor's method for output filename (handles metadata if configured)
                    output_filename = batch_extractor.get_output_filename(h5_file)
                    output_path = os.path.join(features_folder, output_filename)

                    if os.path.exists(output_path):
                        self.show_progress(f"⏭ Skipping {output_filename} (already exists)")
                        files_processed += 1
                        continue

                    self.show_progress(f"Processing: {os.path.basename(h5_file)}")
                    features_df = batch_extractor.process_file(h5_file, output_path)

                    if not features_df.empty:
                        files_processed += 1
                        self.show_progress(f"✓ {output_filename}: {len(features_df)} features extracted")
                    else:
                        self.show_progress(f"⚠ {output_filename}: No features found")

                except Exception as e:
                    self.show_progress(f"✗ Error with {os.path.basename(h5_file)}: {str(e)}")

            self.show_progress(f"Feature extraction complete: {files_processed}/{len(h5_files_to_process)} files processed")
            return files_processed > 0

        except ImportError as e:
            self.show_progress(f"❌ Import error: {str(e)}")
            QMessageBox.critical(self, "Import Error", f"Could not import core modules:\n{str(e)}")
            return False
        except Exception as e:
            self.show_progress(f"❌ Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Feature extraction failed:\n{str(e)}")
            return False

    def _predict_htrs_internal(self, mode='fresh'):
        """Internal method to predict HTRs."""
        try:
            project_path, _ = self.project_manager.get_current_project()

            # Import core modules
            sys.path.insert(0, os.path.dirname(project_path))
            from core.ml_models import HTRPredictor

            # Set up paths
            features_folder = os.path.join(project_path, "features")
            predictions_root = os.path.join(project_path, "predictions")
            os.makedirs(predictions_root, exist_ok=True)

            # Load model
            model_path = self._get_selected_model_path()
            predictor = HTRPredictor()

            self.show_progress(f"Loading model: {os.path.basename(model_path)}")
            if not predictor.load_model(model_path):
                self.show_progress("❌ Failed to load model")
                return False

            self.show_progress("Model loaded successfully")

            # Create versioned run subfolder
            run_folder_name = self._generate_run_folder_name(model_path)
            run_folder = os.path.join(predictions_root, run_folder_name)
            os.makedirs(run_folder, exist_ok=True)
            self.show_progress(f"Prediction run folder: {run_folder_name}")

            # Determine files to predict based on mode
            if mode == 'incremental':
                # Check unpredicted against the active run
                active_path = self._get_active_predictions_path()
                unpredicted_features, _ = self.workflow_tracker.detect_unpredicted_features(
                    run_path=active_path
                )
                if not unpredicted_features:
                    self.show_progress("No new feature files to predict")
                    return True
                self.show_progress(f"Predicting {len(unpredicted_features)} new feature files...")
            else:
                self.show_progress("Predicting all feature files...")

            # Run prediction into the new run subfolder
            result = predictor.predict_folder(features_folder, run_folder)

            if result['success']:
                self.show_progress(f"✓ Prediction complete: {result['files_processed']} files")
                if result.get('files_failed', 0) > 0:
                    self.show_progress(f"⚠ {result['files_failed']} files failed")

                # Store the new run path for report generation in the same pipeline call
                self._latest_run_path = run_folder

                # Refresh run selector and auto-select the new run
                self._populate_run_selector()
                # Select the new run (should be first non-legacy entry)
                for i in range(self.run_combo.count()):
                    if self.run_combo.itemData(i) == run_folder:
                        self.run_combo.setCurrentIndex(i)
                        break

                return True
            else:
                self.show_progress(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                return False

        except ImportError as e:
            self.show_progress(f"❌ Import error: {str(e)}")
            QMessageBox.critical(self, "Import Error", f"Could not import core modules:\n{str(e)}")
            return False
        except Exception as e:
            self.show_progress(f"❌ Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"HTR prediction failed:\n{str(e)}")
            return False

    def _generate_report_internal(self):
        """Internal method to generate report."""
        try:
            project_path, project_config = self.project_manager.get_current_project()

            # Import core modules
            sys.path.insert(0, os.path.dirname(project_path))
            from core.ml_models import HTRPredictor
            from datetime import datetime

            # Use the active run path (prefer _latest_run_path from same pipeline call)
            active_run_path = getattr(self, '_latest_run_path', None)
            if not active_run_path:
                active_run_path = self._get_active_predictions_path()
            if not active_run_path:
                active_run_path = os.path.join(project_path, "predictions")

            reports_folder = os.path.join(project_path, "reports")
            os.makedirs(reports_folder, exist_ok=True)

            # Generate report filename including run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = project_config.get("project_name", "HTR_Analysis")
            run_name = os.path.basename(active_run_path)
            if run_name == "predictions":
                run_name = "legacy"
            report_path = os.path.join(reports_folder, f"{project_name}_{run_name}_Report_{timestamp}.xlsx")

            self.show_progress(f"Compiling prediction results from run: {run_name}")

            # Use HTRPredictor to compile results
            predictor = HTRPredictor()
            result = predictor.compile_results(active_run_path, report_path)

            if result['success']:
                self.show_progress(f"✓ Report generated: {os.path.basename(report_path)}")
                self.show_progress(f"  Total events: {result.get('total_events', 0)}")
                self.show_progress(f"  HTR events: {result.get('htr_events', 0)}")
                if result.get('unique_rats'):
                    self.show_progress(f"  Unique subjects: {result['unique_rats']}")
                return True
            else:
                self.show_progress(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
                return False

        except ImportError as e:
            self.show_progress(f"❌ Import error: {str(e)}")
            QMessageBox.critical(self, "Import Error", f"Could not import core modules:\n{str(e)}")
            return False
        except Exception as e:
            self.show_progress(f"❌ Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Report generation failed:\n{str(e)}")
            return False

    def add_h5_files(self):
        """Browse and add H5 files to project input folder."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Browse for H5 files (allow multiple selection)
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select H5 Files to Add",
            "",
            "H5 Files (*.h5);;All Files (*)"
        )

        if not file_paths:
            return

        # Copy files to input folder
        input_folder = os.path.join(project_path, "input")
        os.makedirs(input_folder, exist_ok=True)

        files_copied = 0
        files_skipped = 0

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(input_folder, filename)

            # Check if file already exists
            if os.path.exists(dest_path):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"{filename} already exists in project.\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    files_skipped += 1
                    continue

            try:
                shutil.copy2(file_path, dest_path)
                files_copied += 1
                self.show_progress(f"✓ Added: {filename}")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Copy Failed",
                    f"Could not copy {filename}:\n{str(e)}"
                )

        # Show summary
        summary_msg = f"Added {files_copied} H5 file(s) to project"
        if files_skipped > 0:
            summary_msg += f"\n{files_skipped} file(s) skipped"

        QMessageBox.information(self, "Files Added", summary_msg)

        # Refresh status to show new files
        self.refresh_status()

    def open_input_folder(self):
        """Open the project input folder in file explorer."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        input_folder = os.path.join(project_path, "input")
        os.makedirs(input_folder, exist_ok=True)

        # Open folder in file explorer (cross-platform)
        try:
            if sys.platform == 'win32':
                os.startfile(input_folder)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', input_folder])
            else:  # Linux
                subprocess.Popen(['xdg-open', input_folder])
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open folder:\n{str(e)}"
            )

    def open_predictions_folder(self):
        """Open the active prediction run's folder in file explorer."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Use the active run's subfolder
        folder_to_open = self._get_active_predictions_path()
        if not folder_to_open:
            folder_to_open = os.path.join(project_path, "predictions")

        if not os.path.exists(folder_to_open):
            QMessageBox.warning(
                self,
                "Folder Not Found",
                "Predictions folder does not exist yet.\n\nRun predictions first."
            )
            return

        # Open folder in file explorer (cross-platform)
        try:
            if sys.platform == 'win32':
                os.startfile(folder_to_open)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', folder_to_open])
            else:  # Linux
                subprocess.Popen(['xdg-open', folder_to_open])
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open folder:\n{str(e)}"
            )

    def view_results_summary(self):
        """View a summary of HTR predictions for the active run."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        # Use the active run's folder
        predictions_folder = self._get_active_predictions_path()
        if not predictions_folder:
            predictions_folder = os.path.join(project_path, "predictions")

        if not os.path.exists(predictions_folder):
            QMessageBox.warning(
                self,
                "No Predictions",
                "No predictions found.\n\nRun HTR prediction first."
            )
            return

        # Load all prediction files and summarize
        # Try multiple patterns to find prediction files
        prediction_files = glob.glob(os.path.join(predictions_folder, "*_predictions.csv"))
        if not prediction_files:
            # Try looking for any CSV files in predictions folder
            prediction_files = glob.glob(os.path.join(predictions_folder, "*.csv"))

        if not prediction_files:
            # Check if folder is empty
            all_files = os.listdir(predictions_folder) if os.path.exists(predictions_folder) else []
            QMessageBox.warning(
                self,
                "No Predictions",
                f"No prediction CSV files found in:\n{predictions_folder}\n\nFiles in folder: {len(all_files)}"
            )
            return

        # Determine run name for the dialog title
        run_name = os.path.basename(predictions_folder)
        if run_name == "predictions":
            run_name = "legacy"

        # Create summary dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"HTR Results Summary — {run_name}")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        # Info label
        info_label = QLabel(f"Summary of HTR predictions for {len(prediction_files)} file(s):")
        info_label.setFont(QFont("Arial", 10))
        layout.addWidget(info_label)

        # Instruction label
        instruction_label = QLabel("Click on any file to view individual HTR events")
        instruction_label.setFont(QFont("Arial", 9))
        instruction_label.setStyleSheet("color: #6c757d; font-style: italic; margin-bottom: 5px;")
        layout.addWidget(instruction_label)

        # Create simplified table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["File", "HTR Count"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.setFont(QFont("Arial", 9))
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)

        # Load and process each file
        summary_data = []
        total_htrs_all = 0

        for pred_file in sorted(prediction_files):
            try:
                df = pd.read_csv(pred_file)
                filename = os.path.basename(pred_file).replace("_predictions.csv", "")
                htr_events = len(df[df['prediction'] == 1])

                summary_data.append({
                    'file': filename,
                    'htrs': htr_events,
                    'pred_file': pred_file  # Store full path for detail view
                })

                total_htrs_all += htr_events

            except Exception as e:
                print(f"Error reading {pred_file}: {e}")

        # Populate table
        table.setRowCount(len(summary_data) + 1)  # +1 for totals row

        for row, data in enumerate(summary_data):
            table.setItem(row, 0, QTableWidgetItem(data['file']))
            table.setItem(row, 1, QTableWidgetItem(str(data['htrs'])))

        # Add totals row
        totals_row = len(summary_data)

        total_item = QTableWidgetItem("TOTAL")
        total_item.setFont(QFont("Arial", 9, QFont.Bold))
        table.setItem(totals_row, 0, total_item)

        total_htrs_item = QTableWidgetItem(str(total_htrs_all))
        total_htrs_item.setFont(QFont("Arial", 9, QFont.Bold))
        table.setItem(totals_row, 1, total_htrs_item)

        # Connect row click to detail view
        def on_row_clicked(row, column):
            # Don't open detail for totals row
            if row < len(summary_data):
                self.show_htr_detail(summary_data[row]['pred_file'], summary_data[row]['file'])

        table.cellClicked.connect(on_row_clicked)

        layout.addWidget(table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_csv_btn = QPushButton("📄 Export to CSV")
        export_csv_btn.clicked.connect(lambda: self.export_results_to_csv(summary_data))
        button_layout.addWidget(export_csv_btn)

        open_folder_btn = QPushButton("Open Predictions Folder")
        open_folder_btn.clicked.connect(lambda: self.open_predictions_folder())
        button_layout.addWidget(open_folder_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def export_results_to_csv(self, summary_data):
        """Export HTR results to wide-format CSV file."""
        if not self.project_manager:
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        try:
            # Get FPS from config
            fps = 160  # Default
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.config import get_config_manager
                config_manager = get_config_manager()
                fps = config_manager.config.default_fps
            except Exception as e:
                print(f"Could not load config, using default FPS: {e}")

            # Load all prediction data
            all_htr_data = {}

            for data in summary_data:
                filename = data['file']
                pred_file = data['pred_file']

                try:
                    df = pd.read_csv(pred_file)
                    # Filter to only HTR events
                    htr_df = df[df['prediction'] == 1].copy()

                    if len(htr_df) > 0:
                        # Extract start frames
                        frames = htr_df['start_frame'].tolist()

                        # Convert frames to timestamps (hh:mm:ss)
                        timestamps = []
                        for frame in frames:
                            total_seconds = frame / fps
                            hours = int(total_seconds // 3600)
                            minutes = int((total_seconds % 3600) // 60)
                            seconds = int(total_seconds % 60)
                            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                            timestamps.append(time_str)

                        all_htr_data[filename] = {
                            'frames': frames,
                            'timestamps': timestamps
                        }
                except Exception as e:
                    print(f"Error reading {pred_file}: {e}")
                    continue

            if not all_htr_data:
                QMessageBox.warning(
                    self,
                    "No Data",
                    "No HTR events found to export."
                )
                return

            # Create wide-format DataFrame
            # Find maximum number of HTRs across all files
            max_htrs = max(len(data['frames']) for data in all_htr_data.values())

            # Build the DataFrame columns
            export_data = {}

            for filename in sorted(all_htr_data.keys()):
                data = all_htr_data[filename]
                frames = data['frames']
                timestamps = data['timestamps']

                # Pad with empty values to match max_htrs
                frames_padded = frames + [''] * (max_htrs - len(frames))
                timestamps_padded = timestamps + [''] * (max_htrs - len(timestamps))

                # Add columns for this file
                export_data[f'{filename}_Frame'] = frames_padded
                export_data[f'{filename}_Timestamp'] = timestamps_padded

            # Create DataFrame
            export_df = pd.DataFrame(export_data)

            # Generate filename (include run name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = project_config.get("project_name", "HTR_Analysis")

            active_run_path = self._get_active_predictions_path()
            run_name = ""
            if active_run_path:
                rn = os.path.basename(active_run_path)
                if rn != "predictions":
                    run_name = f"_{rn}"

            # Save to reports folder
            reports_folder = os.path.join(project_path, "reports")
            os.makedirs(reports_folder, exist_ok=True)

            csv_filename = f"{project_name}{run_name}_HTR_Summary_{timestamp}.csv"
            csv_path = os.path.join(reports_folder, csv_filename)

            # Export to CSV
            export_df.to_csv(csv_path, index=False)

            self.show_progress(f"✓ Exported HTR summary to: {csv_filename}")

            QMessageBox.information(
                self,
                "Export Successful",
                f"HTR summary exported successfully!\n\n"
                f"File: {csv_filename}\n"
                f"Location: {reports_folder}\n\n"
                f"Total videos: {len(all_htr_data)}\n"
                f"Total HTR events: {sum(len(data['frames']) for data in all_htr_data.values())}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Could not export HTR summary:\n{str(e)}"
            )

    def show_htr_detail(self, pred_file_path, filename):
        """Show detailed HTR events for a specific file."""
        try:
            # Load prediction file
            df = pd.read_csv(pred_file_path)

            # Filter to only HTR events
            htr_df = df[df['prediction'] == 1].copy()

            if len(htr_df) == 0:
                QMessageBox.information(
                    self,
                    "No HTR Events",
                    f"No HTR events found in {filename}"
                )
                return

            # Get FPS from config
            fps = 160  # Default
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.config import get_config_manager
                config_manager = get_config_manager()
                fps = config_manager.config.default_fps
            except Exception as e:
                print(f"Could not load config, using default FPS: {e}")

            # Create detail dialog
            detail_dialog = QDialog(self)
            detail_dialog.setWindowTitle(f"HTR Events: {filename}")
            detail_dialog.setMinimumSize(700, 500)

            layout = QVBoxLayout(detail_dialog)

            # Header info
            header_label = QLabel(f"<b>{filename}</b> - {len(htr_df)} HTR event(s) detected (FPS: {fps})")
            header_label.setFont(QFont("Arial", 10))
            layout.addWidget(header_label)

            # Create detail table
            detail_table = QTableWidget()
            detail_table.setColumnCount(3)
            detail_table.setHorizontalHeaderLabels(["Event #", "Start Frame", "Start Time (hh:mm:ss)"])
            detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            detail_table.setFont(QFont("Arial", 9))
            detail_table.setAlternatingRowColors(True)

            # Populate detail table
            detail_table.setRowCount(len(htr_df))

            for idx, (_, row) in enumerate(htr_df.iterrows()):
                # Event number
                detail_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))

                # Start frame
                start_frame = int(row['start_frame']) if 'start_frame' in row else 0
                detail_table.setItem(idx, 1, QTableWidgetItem(str(start_frame)))

                # Start time (convert frame to hh:mm:ss)
                total_seconds = start_frame / fps
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                detail_table.setItem(idx, 2, QTableWidgetItem(time_str))

            layout.addWidget(detail_table)

            # Buttons
            button_layout = QHBoxLayout()
            button_layout.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(detail_dialog.accept)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)

            detail_dialog.exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not load HTR details:\n{str(e)}"
            )

    # ── Run management helpers ──────────────────────────────────────────

    def _generate_run_folder_name(self, model_path):
        """Return a run folder name like ``{model_stem}_{YYYYMMDD}_{HHMMSS}``."""
        model_stem = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_stem}_{timestamp}"

    def _enumerate_prediction_runs(self, predictions_folder):
        """List all prediction run subfolders + detect legacy flat CSVs.

        Returns a list of dicts sorted most-recent-first::

            [{run_name, run_path, is_legacy, csv_count}, ...]
        """
        runs = []
        if not os.path.exists(predictions_folder):
            return runs

        # Check for legacy flat CSVs in root
        root_csvs = glob.glob(os.path.join(predictions_folder, "*.csv"))
        if root_csvs:
            runs.append({
                'run_name': 'legacy',
                'run_path': predictions_folder,
                'is_legacy': True,
                'csv_count': len(root_csvs),
            })

        # Check subfolders
        try:
            entries = sorted(os.listdir(predictions_folder), reverse=True)
        except OSError:
            entries = []

        for entry in entries:
            subfolder = os.path.join(predictions_folder, entry)
            if not os.path.isdir(subfolder):
                continue
            csvs = glob.glob(os.path.join(subfolder, "*.csv"))
            if csvs:
                runs.append({
                    'run_name': entry,
                    'run_path': subfolder,
                    'is_legacy': False,
                    'csv_count': len(csvs),
                })

        # Sort: non-legacy runs by name descending (newest timestamp first), legacy last
        non_legacy = [r for r in runs if not r['is_legacy']]
        legacy = [r for r in runs if r['is_legacy']]
        non_legacy.sort(key=lambda r: r['run_name'], reverse=True)
        return non_legacy + legacy

    def _get_active_predictions_path(self):
        """Return the folder path of the currently selected prediction run."""
        data = self.run_combo.currentData()
        return data if data else None

    def _populate_run_selector(self):
        """Fill the run combo from enumeration, auto-select most recent."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return

        predictions_folder = os.path.join(project_path, "predictions")
        runs = self._enumerate_prediction_runs(predictions_folder)

        # Block signals to avoid triggering _on_run_selected during population
        self.run_combo.blockSignals(True)
        self.run_combo.clear()

        if not runs:
            self.run_combo.addItem("No prediction runs", "")
        else:
            for run in runs:
                label = run['run_name']
                if run['is_legacy']:
                    label = f"legacy ({run['csv_count']} files)"
                else:
                    label = f"{run['run_name']} ({run['csv_count']} files)"
                self.run_combo.addItem(label, run['run_path'])

        self.run_combo.blockSignals(False)

        # Enable compare button when 2+ runs exist
        self.compare_runs_btn.setEnabled(len(runs) >= 2)

        # Trigger update for selected run
        self._on_run_selected(self.run_combo.currentIndex())

    def _on_run_selected(self, index):
        """Handle combo box selection change — update button states."""
        active_path = self._get_active_predictions_path()
        has_run = bool(active_path and os.path.exists(active_path))
        self.open_predictions_btn.setEnabled(has_run)
        self.view_summary_btn.setEnabled(has_run)

    def _load_htr_counts(self, run_path):
        """Load ``{filename: htr_count}`` dict from a prediction run folder."""
        counts = {}
        if not run_path or not os.path.exists(run_path):
            return counts

        csv_files = glob.glob(os.path.join(run_path, "*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                filename = os.path.basename(csv_file)
                htr_count = len(df[df['prediction'] == 1])
                counts[filename] = htr_count
            except Exception:
                continue
        return counts

    def _show_comparison_dialog(self):
        """Side-by-side HTR count comparison between two runs."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return

        predictions_folder = os.path.join(project_path, "predictions")
        runs = self._enumerate_prediction_runs(predictions_folder)

        if len(runs) < 2:
            QMessageBox.information(self, "Compare Runs", "Need at least 2 prediction runs to compare.")
            return

        # Build comparison dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare Prediction Runs")
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)

        # Run selectors
        selectors_layout = QHBoxLayout()

        selectors_layout.addWidget(QLabel("Run A:"))
        combo_a = QComboBox()
        combo_a.setFont(QFont("Arial", 9))
        for run in runs:
            label = run['run_name'] if not run['is_legacy'] else "legacy"
            combo_a.addItem(f"{label} ({run['csv_count']} files)", run['run_path'])
        selectors_layout.addWidget(combo_a)

        selectors_layout.addWidget(QLabel("Run B:"))
        combo_b = QComboBox()
        combo_b.setFont(QFont("Arial", 9))
        for run in runs:
            label = run['run_name'] if not run['is_legacy'] else "legacy"
            combo_b.addItem(f"{label} ({run['csv_count']} files)", run['run_path'])
        if len(runs) > 1:
            combo_b.setCurrentIndex(1)
        selectors_layout.addWidget(combo_b)

        compare_btn = QPushButton("Compare")
        compare_btn.setFont(QFont("Arial", 9))
        selectors_layout.addWidget(compare_btn)

        selectors_layout.addStretch()
        layout.addLayout(selectors_layout)

        # Results table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["File", "Run A HTRs", "Run B HTRs", "Delta"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.setFont(QFont("Arial", 9))
        table.setAlternatingRowColors(True)
        layout.addWidget(table)

        def do_compare():
            path_a = combo_a.currentData()
            path_b = combo_b.currentData()
            counts_a = self._load_htr_counts(path_a)
            counts_b = self._load_htr_counts(path_b)

            all_files = sorted(set(counts_a.keys()) | set(counts_b.keys()))
            table.setRowCount(len(all_files) + 1)  # +1 for totals

            total_a = 0
            total_b = 0

            for row, filename in enumerate(all_files):
                table.setItem(row, 0, QTableWidgetItem(filename))

                a_val = counts_a.get(filename)
                b_val = counts_b.get(filename)

                a_str = str(a_val) if a_val is not None else "-"
                b_str = str(b_val) if b_val is not None else "-"

                table.setItem(row, 1, QTableWidgetItem(a_str))
                table.setItem(row, 2, QTableWidgetItem(b_str))

                if a_val is not None and b_val is not None:
                    delta = b_val - a_val
                    delta_str = f"+{delta}" if delta > 0 else str(delta)
                    table.setItem(row, 3, QTableWidgetItem(delta_str))
                else:
                    table.setItem(row, 3, QTableWidgetItem("-"))

                total_a += a_val if a_val is not None else 0
                total_b += b_val if b_val is not None else 0

            # Totals row
            totals_row = len(all_files)
            total_label = QTableWidgetItem("TOTAL")
            total_label.setFont(QFont("Arial", 9, QFont.Bold))
            table.setItem(totals_row, 0, total_label)

            ta = QTableWidgetItem(str(total_a))
            ta.setFont(QFont("Arial", 9, QFont.Bold))
            table.setItem(totals_row, 1, ta)

            tb = QTableWidgetItem(str(total_b))
            tb.setFont(QFont("Arial", 9, QFont.Bold))
            table.setItem(totals_row, 2, tb)

            delta_total = total_b - total_a
            delta_str = f"+{delta_total}" if delta_total > 0 else str(delta_total)
            td = QTableWidgetItem(delta_str)
            td.setFont(QFont("Arial", 9, QFont.Bold))
            table.setItem(totals_row, 3, td)

        compare_btn.clicked.connect(do_compare)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Auto-compare on open
        do_compare()

        dialog.exec()

    def _update_ear_mode_label(self):
        """Update the ear detection mode label based on current config."""
        try:
            from core.config import get_config_manager
            config_manager = get_config_manager()
            if config_manager and config_manager.config:
                use_prominence = config_manager.config.ear_detector.use_prominence_mode
                if use_prominence:
                    self.ear_mode_label.setText("Ear Detection: Prominence Mode")
                    self.ear_mode_label.setStyleSheet("color: #1565c0; font-weight: bold; padding: 2px 0;")
                else:
                    self.ear_mode_label.setText("Ear Detection: Absolute Mode")
                    self.ear_mode_label.setStyleSheet("color: #666; padding: 2px 0;")
        except Exception:
            self.ear_mode_label.setText("Ear Detection: Absolute Mode")
            self.ear_mode_label.setStyleSheet("color: #666; padding: 2px 0;")

    def show_progress(self, message):
        """Show progress message."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        self.progress_text.append(formatted_message)

        # Auto-scroll
        cursor = self.progress_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.progress_text.setTextCursor(cursor)
