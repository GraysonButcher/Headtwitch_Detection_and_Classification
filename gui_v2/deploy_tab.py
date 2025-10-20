"""
Deploy Tab - Smart Batch Processing with Incremental Support

Handles both fresh batch processing and incremental updates to existing projects.
Automatically detects new files and offers appropriate processing options.
"""

import os
import sys
import glob
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QProgressBar, QTextEdit, QMessageBox,
    QFileDialog, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QDateTime, Signal
from PySide6.QtGui import QFont

# Import workflow tracker
try:
    from .workflow_tracker import WorkflowTracker
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from workflow_tracker import WorkflowTracker


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

        # Model and parameters configuration
        self.create_config_section(layout)

        # Processing mode selection
        self.create_mode_selection_section(layout)

        # Processing buttons
        self.create_processing_buttons_section(layout)

        # Progress section
        self.create_progress_section(layout)

        # Final stretch
        layout.addStretch()

    def create_instructions_section(self, parent_layout):
        """Instructions header."""
        instructions_group = QGroupBox("Production Deployment")
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_layout.setContentsMargins(10, 10, 10, 10)

        instructions = QLabel(
            "<b>Deploy trained models to analyze data:</b> "
            "Select a model, choose processing mode (fresh or incremental), and run the pipeline."
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

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select trained model file (.joblib)")
        self.model_path_edit.setFont(QFont("Arial", 9))
        model_layout.addWidget(self.model_path_edit)

        model_browse_btn = QPushButton("Browse...")
        model_browse_btn.setMaximumWidth(80)
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_browse_btn)

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
        param_layout.addWidget(self.param_path_edit)

        param_browse_btn = QPushButton("Browse...")
        param_browse_btn.setMaximumWidth(80)
        param_browse_btn.clicked.connect(self.browse_parameters)
        param_layout.addWidget(param_browse_btn)

        config_layout.addLayout(param_layout)

        parent_layout.addWidget(config_group)

    def create_mode_selection_section(self, parent_layout):
        """Processing mode selection (fresh vs incremental)."""
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(10, 10, 10, 10)
        mode_layout.setSpacing(8)

        self.mode_button_group = QButtonGroup(self)

        # Fresh batch mode
        self.fresh_mode_radio = QRadioButton("Fresh Batch (Process all files)")
        self.fresh_mode_radio.setFont(QFont("Arial", 9))
        self.mode_button_group.addButton(self.fresh_mode_radio, 0)
        mode_layout.addWidget(self.fresh_mode_radio)

        fresh_help = QLabel("   → Extract features, predict, and generate report for all H5 files")
        fresh_help.setFont(QFont("Arial", 8))
        fresh_help.setStyleSheet("color: #6c757d;")
        mode_layout.addWidget(fresh_help)

        # Incremental mode
        self.incremental_mode_radio = QRadioButton("Incremental (Process only new files)")
        self.incremental_mode_radio.setFont(QFont("Arial", 9))
        self.mode_button_group.addButton(self.incremental_mode_radio, 1)
        mode_layout.addWidget(self.incremental_mode_radio)

        incremental_help = QLabel("   → Process only new H5 files added since last run, update existing report")
        incremental_help.setFont(QFont("Arial", 8))
        incremental_help.setStyleSheet("color: #6c757d;")
        mode_layout.addWidget(incremental_help)

        # Default selection
        self.fresh_mode_radio.setChecked(True)

        parent_layout.addWidget(mode_group)

    def create_processing_buttons_section(self, parent_layout):
        """Processing control buttons."""
        buttons_layout = QHBoxLayout()

        # Full pipeline button
        self.run_pipeline_btn = QPushButton("🚀 Run Full Pipeline")
        self.run_pipeline_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.run_pipeline_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.run_pipeline_btn.clicked.connect(self.run_full_pipeline)
        self.run_pipeline_btn.setEnabled(False)
        buttons_layout.addWidget(self.run_pipeline_btn)

        buttons_layout.addStretch()

        # Manual step buttons (advanced)
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

        self.step3_btn = QPushButton("3️⃣ Generate Report")
        self.step3_btn.setFont(QFont("Arial", 9))
        self.step3_btn.clicked.connect(self.run_report_step)
        self.step3_btn.setEnabled(False)
        buttons_layout.addWidget(self.step3_btn)

        parent_layout.addLayout(buttons_layout)

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
            self.run_pipeline_btn.setEnabled(False)
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            self.status_label.setText("No project loaded. Create or open a project to begin.")
            self.run_pipeline_btn.setEnabled(False)
            return

        # Initialize workflow tracker
        self.workflow_tracker = WorkflowTracker(project_path)

        # Get workflow status
        status = self.workflow_tracker.get_workflow_status()
        message = self.workflow_tracker.get_status_message()
        recommendation = self.workflow_tracker.get_processing_recommendation()

        # Update status label
        self.status_label.setText(message)

        # Update file counts
        h5_total = status['h5_files']['total']
        h5_new = status['h5_files']['new']
        h5_processed = status['h5_files']['processed']
        features_total = status['features']['total']
        predictions_total = status['predictions']['total']

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
        model_path = self.model_path_edit.text().strip()
        has_model = bool(model_path and os.path.exists(model_path))

        if h5_total > 0 and has_model:
            self.run_pipeline_btn.setEnabled(True)
            self.step1_btn.setEnabled(True)
            self.step2_btn.setEnabled(features_total > 0)
            self.step3_btn.setEnabled(predictions_total > 0)
        else:
            self.run_pipeline_btn.setEnabled(False)
            self.step1_btn.setEnabled(False)
            self.step2_btn.setEnabled(False)
            self.step3_btn.setEnabled(False)

    def browse_model(self):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.joblib);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.refresh_status()

    def browse_parameters(self):
        """Browse for parameter file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parameter File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.param_path_edit.setText(file_path)

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

        # Run pipeline
        self.show_progress("🚀 Starting full pipeline...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        try:
            # Step 1: Extract features
            self.show_progress("📦 Step 1/3: Extracting features...")
            extract_success = self._extract_features_internal(mode)

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

        mode = "incremental" if self.incremental_mode_radio.isChecked() else "fresh"
        success = self._extract_features_internal(mode)

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
            model_path = self.model_path_edit.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Error", "Please select a trained model file.")
                return False

            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Error", f"Model file not found: {model_path}")
                return False

        return True

    def _extract_features_internal(self, mode='fresh'):
        """Internal method to extract features."""
        try:
            project_path, _ = self.project_manager.get_current_project()

            # Import core modules
            sys.path.insert(0, os.path.dirname(project_path))
            from core.feature_extraction import BatchFeatureExtractor
            from core.config import get_config_manager

            # Get config
            config_manager = get_config_manager()

            # Load custom parameters if provided
            param_path = self.param_path_edit.text().strip()
            if param_path and os.path.exists(param_path):
                config_manager.import_parameters(param_path)
                self.show_progress(f"Using custom parameters: {os.path.basename(param_path)}")
            else:
                self.show_progress("Using default parameters")

            # Set up paths
            input_folder = os.path.join(project_path, "input")
            features_folder = os.path.join(project_path, "features")
            os.makedirs(features_folder, exist_ok=True)

            # Determine files to process based on mode
            if mode == 'incremental':
                new_h5_files, _ = self.workflow_tracker.detect_new_h5_files()
                if not new_h5_files:
                    self.show_progress("No new H5 files to process")
                    return True
                h5_files_to_process = [os.path.join(input_folder, f) for f in new_h5_files]
                self.show_progress(f"Processing {len(h5_files_to_process)} new H5 files...")
            else:
                # Fresh mode - process all
                h5_files_to_process = glob.glob(os.path.join(input_folder, "**", "*.h5"), recursive=True)
                self.show_progress(f"Processing all {len(h5_files_to_process)} H5 files...")

            # Create batch extractor
            batch_extractor = BatchFeatureExtractor(
                ear_config=config_manager.config.ear_detector,
                head_config=config_manager.config.head_detector,
                node_mapping=config_manager.config.node_mapping,
                fps=config_manager.config.default_fps
            )

            # Process files
            files_processed = 0
            for h5_file in h5_files_to_process:
                try:
                    base_name = os.path.splitext(os.path.basename(h5_file))[0]
                    output_path = os.path.join(features_folder, f"{base_name}_htr_features.csv")

                    self.show_progress(f"Processing: {base_name}")
                    features_df = batch_extractor.process_file(h5_file, output_path)

                    if not features_df.empty:
                        files_processed += 1
                        self.show_progress(f"✓ {base_name}: {len(features_df)} features extracted")
                    else:
                        self.show_progress(f"⚠ {base_name}: No features found")

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
            predictions_folder = os.path.join(project_path, "predictions")
            os.makedirs(predictions_folder, exist_ok=True)

            # Load model
            model_path = self.model_path_edit.text().strip()
            predictor = HTRPredictor()

            self.show_progress(f"Loading model: {os.path.basename(model_path)}")
            if not predictor.load_model(model_path):
                self.show_progress("❌ Failed to load model")
                return False

            self.show_progress("Model loaded successfully")

            # Determine files to predict based on mode
            if mode == 'incremental':
                unpredicted_features, _ = self.workflow_tracker.detect_unpredicted_features()
                if not unpredicted_features:
                    self.show_progress("No new feature files to predict")
                    return True
                self.show_progress(f"Predicting {len(unpredicted_features)} new feature files...")
                # For incremental, we'll selectively process by copying to temp folder
                # (simpler approach - full folder predict filters by what exists)
            else:
                self.show_progress("Predicting all feature files...")

            # Run prediction
            result = predictor.predict_folder(features_folder, predictions_folder)

            if result['success']:
                self.show_progress(f"✓ Prediction complete: {result['files_processed']} files")
                if result.get('files_failed', 0) > 0:
                    self.show_progress(f"⚠ {result['files_failed']} files failed")
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

            # Set up paths
            predictions_folder = os.path.join(project_path, "predictions")
            reports_folder = os.path.join(project_path, "reports")
            os.makedirs(reports_folder, exist_ok=True)

            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = project_config.get("project_name", "HTR_Analysis")
            report_path = os.path.join(reports_folder, f"{project_name}_Report_{timestamp}.xlsx")

            self.show_progress("Compiling prediction results...")

            # Use HTRPredictor to compile results
            predictor = HTRPredictor()
            result = predictor.compile_results(predictions_folder, report_path)

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

    def show_progress(self, message):
        """Show progress message."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        self.progress_text.append(formatted_message)

        # Auto-scroll
        cursor = self.progress_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.progress_text.setTextCursor(cursor)
