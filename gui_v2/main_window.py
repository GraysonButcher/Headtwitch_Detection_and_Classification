"""
HTR Analysis Tool - Main Window v3 (Restructured)

New 5-tab workflow structure:
1. Welcome - Project overview and workflow navigation
2. Tune Parameters - Parameter optimization with video feedback
3. Prepare Data - Feature extraction + ground truth labeling
4. Train Model - ML training with evaluation and iteration
5. Deploy - Smart batch processing (fresh and incremental)

Version: 3.0 (2025-10-15)
"""

import sys
import os
import glob
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QHBoxLayout, QLabel, QPushButton, QGroupBox, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QMessageBox, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter
)
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QFont, QPixmap

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
try:
    from core.config import get_config_manager
except ImportError:
    def get_config_manager():
        return None

# Import GUI components
try:
    from .welcome_tab import WelcomeTab
    from .parameter_panel import ParameterPanel
    from .prepare_data_tab import PrepareDataTab
    from .deploy_tab import DeployTab
    from .project_dialog import ProjectDialog
    from .project_manager import ProjectManager
    from .video_inspector_widget import VideoInspectorWidget
    from .diagnostics_graph_widget import DiagnosticsGraphWidget
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from welcome_tab import WelcomeTab
    from parameter_panel import ParameterPanel
    from prepare_data_tab import PrepareDataTab
    from deploy_tab import DeployTab
    from project_dialog import ProjectDialog
    from project_manager import ProjectManager
    from video_inspector_widget import VideoInspectorWidget
    from diagnostics_graph_widget import DiagnosticsGraphWidget


class HTRAnalysisAppV3(QMainWindow):
    """Main application window with 5-tab workflow structure."""

    def __init__(self):
        super().__init__()
        self.config_manager = get_config_manager()
        self.project_manager = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("HTR Analysis Tool v3")
        self.setFixedSize(1400, 750)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setMaximumHeight(700)
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_tabs()

        # Create status bar
        self.status_bar = self.statusBar()
        self.project_status_label = QLabel("No project loaded")
        self.status_bar.addPermanentWidget(self.project_status_label)

        # Initialize project manager
        try:
            self.project_manager = ProjectManager()
            self.update_status_bar()
        except Exception as e:
            print(f"Warning: Could not initialize ProjectManager: {e}")
            self.project_status_label.setText("Project Manager Error")

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # New Project
        new_project_action = file_menu.addAction("New Project...")
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)

        # Open Project
        open_project_action = file_menu.addAction("Open Project...")
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self.open_project)

        file_menu.addSeparator()

        # Exit
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def update_status_bar(self):
        """Update the status bar with current project info."""
        if not self.project_manager:
            self.project_status_label.setText("No project loaded")
            return

        project_path, project_config = self.project_manager.get_current_project()
        if project_path and project_config:
            project_name = project_config.get("project_name", "Unknown")
            self.project_status_label.setText(f"Project: {project_name}")
        else:
            self.project_status_label.setText("No project loaded")

    def new_project(self):
        """Create a new project."""
        try:
            dialog = ProjectDialog(self, workflow_type="general", mode="create")
            if dialog.exec():
                # Refresh all tabs
                self.update_all_tabs()
                self.update_status_bar()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{str(e)}")

    def open_project(self):
        """Open an existing project."""
        try:
            dialog = ProjectDialog(self, workflow_type="general", mode="open")
            if dialog.exec():
                # Refresh all tabs
                self.update_all_tabs()
                self.update_status_bar()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{str(e)}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About HTR Analysis Tool",
            "<h3>HTR Analysis Tool v3.0</h3>"
            "<p>Head-Twitch Response detection and analysis using machine learning.</p>"
            "<p><b>New in v3:</b></p>"
            "<ul>"
            "<li>5-tab workflow structure</li>"
            "<li>Built-in CSV editor for labeling</li>"
            "<li>Smart incremental processing</li>"
            "<li>Model evaluation & iteration tools</li>"
            "</ul>"
            "<p>Developed with PySide6 and scikit-learn</p>"
        )

    def create_tabs(self):
        """Create all tabs in the workflow order."""
        # Tab 1: Welcome
        self.create_welcome_tab()

        # Tab 2: Tune Parameters
        self.create_tune_parameters_tab()

        # Tab 3: Prepare Data (NEW)
        self.create_prepare_data_tab()

        # Tab 4: Train Model
        self.create_train_model_tab()

        # Tab 5: Deploy (formerly Batch Process)
        self.create_deploy_tab()

    def create_welcome_tab(self):
        """Tab 1: Welcome and project overview."""
        try:
            self.welcome_tab = WelcomeTab()
            # Connect navigation signals
            self.welcome_tab.tune_parameters_requested.connect(lambda: self.switch_to_tab(1))
            self.welcome_tab.prepare_data_requested.connect(lambda: self.switch_to_tab(2))
            self.welcome_tab.train_model_requested.connect(lambda: self.switch_to_tab(3))
            self.welcome_tab.deploy_requested.connect(lambda: self.switch_to_tab(4))
            self.tab_widget.addTab(self.welcome_tab, "Welcome")
        except Exception as e:
            # Fallback if welcome tab fails
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Welcome Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "Welcome")

    def create_tune_parameters_tab(self):
        """Tab 2: Tune Parameters with video feedback."""
        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(10)

        # LEFT SIDE: Video Inspector (690px)
        try:
            self.video_inspector = VideoInspectorWidget(parent=self)
            self.video_inspector.setMaximumWidth(690)
            param_layout.addWidget(self.video_inspector)
        except Exception as e:
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Video Inspector Error: {str(e)}"))
            fallback_widget.setMaximumWidth(690)
            param_layout.addWidget(fallback_widget)
            self.video_inspector = None

        # RIGHT SIDE: Resizable Graph + Parameter Panel (690px)
        # Use QSplitter for user-adjustable sizing
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setMaximumWidth(690)

        # Top: Diagnostics Graph (resizable)
        try:
            self.diagnostics_graph = DiagnosticsGraphWidget(parent=self)
            # Remove fixed height - splitter will handle sizing
            right_splitter.addWidget(self.diagnostics_graph)
        except Exception as e:
            fallback_label = QLabel(f"Graph Widget Error: {str(e)}")
            fallback_label.setAlignment(Qt.AlignCenter)
            right_splitter.addWidget(fallback_label)
            self.diagnostics_graph = None

        # Bottom: Parameter Panel (scrollable, resizable)
        try:
            self.parameter_panel = ParameterPanel(parent=self, project_manager=self.project_manager)
            right_splitter.addWidget(self.parameter_panel)
        except Exception as e:
            fallback_label = QLabel(f"Parameter Panel Error: {str(e)}")
            fallback_label.setAlignment(Qt.AlignCenter)
            right_splitter.addWidget(fallback_label)
            self.parameter_panel = None

        # Set initial sizes: Give graph ~455px, parameters ~245px (total 700px)
        # This gives the graph significantly more space than before
        right_splitter.setSizes([455, 245])

        param_layout.addWidget(right_splitter)

        # WIRE UP SIGNALS
        if self.video_inspector and self.diagnostics_graph and self.parameter_panel:
            # H5 signals calculated -> plot on graph
            self.video_inspector.signals_calculated.connect(self.diagnostics_graph.set_signals)
            self.video_inspector.signals_calculated.connect(self.on_h5_loaded)

            # Frame changed -> update cursor
            self.video_inspector.frame_changed.connect(self.diagnostics_graph.update_frame_cursor)

            # Reanalyze requests -> run detection
            self.parameter_panel.reanalyze_view_requested.connect(self.reanalyze_current_view)
            self.parameter_panel.reanalyze_full_requested.connect(self.reanalyze_full_video)

        self.tab_widget.addTab(param_widget, "Tune Parameters")

    def create_prepare_data_tab(self):
        """Tab 3: Prepare Data - Feature extraction + Ground truth labeling."""
        try:
            self.prepare_data_tab = PrepareDataTab(parent=self, project_manager=self.project_manager)
            # Connect signals
            self.prepare_data_tab.features_extracted.connect(self.on_features_extracted)
            self.prepare_data_tab.labels_updated.connect(self.on_labels_updated)
            self.tab_widget.addTab(self.prepare_data_tab, "Prepare Data")
        except Exception as e:
            # Fallback
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Prepare Data Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "Prepare Data")

    def create_train_model_tab(self):
        """Tab 4: Train Model with evaluation and iteration."""
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        # Section A: Configure & Train
        self.create_training_config_section(layout)

        # Separator
        separator = QLabel()
        separator.setStyleSheet("background-color: #dee2e6; min-height: 2px; max-height: 2px;")
        layout.addWidget(separator)

        # Section B: Evaluate & Iterate (NEW)
        self.create_training_evaluate_section(layout)

        # Progress section
        self.create_training_progress_section(layout)

        layout.addStretch()

        self.tab_widget.addTab(training_widget, "Train Model")

    def create_training_config_section(self, parent_layout):
        """Training configuration section."""
        config_group = QGroupBox("Configure & Train Model")
        config_group.setFont(QFont("Arial", 10, QFont.Bold))
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 15, 10, 10)
        config_layout.setSpacing(10)

        # Training data status display
        self.training_status_label = QLabel("Loading training data...")
        self.training_status_label.setFont(QFont("Arial", 9))
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 10px; border-radius: 4px;"
        )
        config_layout.addWidget(self.training_status_label)

        # Refresh button
        refresh_training_btn = QPushButton("🔄 Refresh Training Data")
        refresh_training_btn.setFont(QFont("Arial", 9))
        refresh_training_btn.clicked.connect(self.refresh_training_status)
        config_layout.addWidget(refresh_training_btn)

        # Parameters file (optional)
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setMinimumWidth(120)
        param_label.setFont(QFont("Arial", 9))
        param_layout.addWidget(param_label)

        self.training_param_edit = QLineEdit()
        self.training_param_edit.setPlaceholderText("Optional: Parameter configuration")
        self.training_param_edit.setFont(QFont("Arial", 9))
        param_layout.addWidget(self.training_param_edit)

        browse_param_btn = QPushButton("Browse...")
        browse_param_btn.setMaximumWidth(80)
        browse_param_btn.clicked.connect(self.browse_training_params)
        param_layout.addWidget(browse_param_btn)

        config_layout.addLayout(param_layout)

        # Train button
        train_layout = QHBoxLayout()

        self.train_model_btn = QPushButton("🧠 Train Model")
        self.train_model_btn.setFont(QFont("Arial", 10, QFont.Bold))
        self.train_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.train_model_btn.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_model_btn)

        train_layout.addStretch()

        config_layout.addLayout(train_layout)

        parent_layout.addWidget(config_group)

    def create_training_evaluate_section(self, parent_layout):
        """Training evaluation and iteration section."""
        eval_group = QGroupBox("Evaluate & Iterate")
        eval_group.setFont(QFont("Arial", 10, QFont.Bold))
        eval_layout = QVBoxLayout(eval_group)
        eval_layout.setContentsMargins(10, 15, 10, 10)
        eval_layout.setSpacing(10)

        # Instructions
        instructions = QLabel(
            "<b>Review model performance:</b> "
            "Analyze misclassified events, fix labels, and retrain to improve accuracy."
        )
        instructions.setFont(QFont("Arial", 9))
        instructions.setWordWrap(True)
        eval_layout.addWidget(instructions)

        # Metrics display
        self.metrics_label = QLabel("Train a model to see performance metrics")
        self.metrics_label.setFont(QFont("Arial", 9))
        self.metrics_label.setStyleSheet(
            "background-color: #f8f9fa; padding: 8px; border-radius: 4px; color: #6c757d;"
        )
        eval_layout.addWidget(self.metrics_label)

        # Misclassified events section
        misclass_layout = QHBoxLayout()

        # Load misclassified button
        self.load_misclass_btn = QPushButton("📊 Load Misclassified Events")
        self.load_misclass_btn.setFont(QFont("Arial", 9))
        self.load_misclass_btn.clicked.connect(self.load_misclassified_events)
        self.load_misclass_btn.setEnabled(False)
        misclass_layout.addWidget(self.load_misclass_btn)

        # View confusion matrix button
        self.view_confusion_btn = QPushButton("📈 View Confusion Matrix")
        self.view_confusion_btn.setFont(QFont("Arial", 9))
        self.view_confusion_btn.clicked.connect(self.view_confusion_matrix)
        self.view_confusion_btn.setEnabled(False)
        misclass_layout.addWidget(self.view_confusion_btn)

        misclass_layout.addStretch()

        eval_layout.addLayout(misclass_layout)

        # Misclassified events table (compact)
        self.misclass_table = QTableWidget()
        self.misclass_table.setMaximumHeight(150)
        self.misclass_table.setColumnCount(5)
        self.misclass_table.setHorizontalHeaderLabels([
            "Error Type", "Start Frame", "End Frame", "File", "Notes"
        ])
        self.misclass_table.horizontalHeader().setStretchLastSection(True)
        self.misclass_table.setVisible(False)
        eval_layout.addWidget(self.misclass_table)

        parent_layout.addWidget(eval_group)

    def create_training_progress_section(self, parent_layout):
        """Training progress display."""
        progress_group = QGroupBox("Training Progress & Results")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(6)

        # Progress bar
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setVisible(False)
        progress_layout.addWidget(self.training_progress_bar)

        # Results text
        self.training_results_text = QTextEdit()
        self.training_results_text.setMaximumHeight(100)
        self.training_results_text.setFont(QFont("Consolas", 8))
        self.training_results_text.setReadOnly(True)
        self.training_results_text.setPlaceholderText("Training progress will appear here...")
        self.training_results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                color: #495057;
            }
        """)
        progress_layout.addWidget(self.training_results_text)

        parent_layout.addWidget(progress_group)

    def create_deploy_tab(self):
        """Tab 5: Deploy - Smart batch processing."""
        try:
            self.deploy_tab = DeployTab(parent=self, project_manager=self.project_manager)
            # Connect signals
            self.deploy_tab.processing_complete.connect(self.on_processing_complete)
            self.tab_widget.addTab(self.deploy_tab, "Deploy")
        except Exception as e:
            # Fallback
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Deploy Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "Deploy")

    # ==================== Tab Navigation ====================

    def switch_to_tab(self, index):
        """Switch to specified tab index."""
        self.tab_widget.setCurrentIndex(index)

    # ==================== Project Management ====================

    def update_all_tabs(self):
        """Update all tabs when project changes."""
        # Update Prepare Data tab
        if hasattr(self, 'prepare_data_tab'):
            self.prepare_data_tab.set_project_manager(self.project_manager)
            self.prepare_data_tab.refresh_status()

        # Update Deploy tab
        if hasattr(self, 'deploy_tab'):
            self.deploy_tab.set_project_manager(self.project_manager)
            self.deploy_tab.refresh_status()

        # Update Parameter Panel
        if hasattr(self, 'parameter_panel'):
            try:
                self.parameter_panel.refresh_project_status()
            except AttributeError:
                pass

        # Update Train Model tab status
        if hasattr(self, 'training_status_label'):
            self.refresh_training_status()

    # ==================== Signal Handlers ====================

    def on_features_extracted(self):
        """Handle features extracted signal."""
        # Refresh Deploy tab status
        if hasattr(self, 'deploy_tab'):
            self.deploy_tab.refresh_status()

    def on_labels_updated(self):
        """Handle labels updated signal."""
        # Could track labeling progress here
        pass

    def on_processing_complete(self):
        """Handle processing complete signal."""
        # Could update welcome tab statistics
        pass

    # ==================== Parameter Tuning Functions ====================

    def on_h5_loaded(self, signals_df):
        """Handle H5 file loaded event - enable analysis buttons."""
        if self.parameter_panel:
            self.parameter_panel.enable_analysis_buttons(True)

    def reanalyze_current_view(self):
        """Run detection on the visible graph range."""
        if not self.video_inspector or not self.video_inspector.signals_df is not None:
            QMessageBox.warning(self, "No Data", "Please load an H5 file first.")
            return

        if not self.diagnostics_graph:
            return

        # Get visible range from graph
        start, end = self.diagnostics_graph.get_view_range()

        # Run detection on this range
        try:
            events_df = self.run_detection_on_range(start, end)

            # Update graph overlays
            self.diagnostics_graph.update_events(events_df)

            # Update status
            if self.parameter_panel:
                self.parameter_panel.status_label.setText(
                    f"Analysis complete: {len(events_df)} events detected in frames {start}-{end}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Detection Error",
                f"Failed to run detection:\n\n{str(e)}"
            )
            if self.parameter_panel:
                self.parameter_panel.status_label.setText(f"Error: {str(e)}")

    def reanalyze_full_video(self):
        """Run detection on entire video."""
        if not self.video_inspector or not self.video_inspector.signals_df is not None:
            QMessageBox.warning(self, "No Data", "Please load an H5 file first.")
            return

        # Show progress dialog
        from PySide6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Analyzing entire video...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(10)
        QApplication.processEvents()

        try:
            # Run detection on full range
            total_frames = len(self.video_inspector.signals_df)
            events_df = self.run_detection_on_range(0, total_frames)

            progress.setValue(90)
            QApplication.processEvents()

            # Update graph overlays
            if self.diagnostics_graph:
                self.diagnostics_graph.update_events(events_df)

            progress.setValue(100)
            progress.close()

            # Update status
            if self.parameter_panel:
                self.parameter_panel.status_label.setText(
                    f"Full analysis complete: {len(events_df)} events detected"
                )

            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Found {len(events_df)} events in {total_frames} frames."
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Detection Error",
                f"Failed to run detection:\n\n{str(e)}"
            )
            if self.parameter_panel:
                self.parameter_panel.status_label.setText(f"Error: {str(e)}")

    def run_detection_on_range(self, start_frame: int, end_frame: int):
        """
        Run HTR detection on specified frame range using current parameters.

        Uses the full detector classes from core.detectors for accurate results.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            DataFrame with columns ['start_frame', 'end_frame', 'confidence', 'detection_method']
        """
        import pandas as pd

        # Import full detector classes
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from core.detectors import CombinedDetector
            from core.data_processing import SleapDataLoader
            from core.config import NodeMapping
        except ImportError as e:
            raise ImportError(f"Could not import detection modules: {e}")

        # Get current parameters from config manager
        if not self.config_manager:
            raise ValueError("Config manager not available")

        config = self.config_manager.config

        # Get H5 path from video inspector
        if not self.video_inspector or not self.video_inspector.h5_path:
            raise ValueError("No H5 file loaded")

        h5_path = self.video_inspector.h5_path

        # Create SleapDataLoader
        data_loader = SleapDataLoader(h5_path)
        if not data_loader.load_data():
            raise ValueError("Failed to load H5 data")

        # Create NodeMapping from video inspector's mapping
        node_mapping = NodeMapping(
            left_ear=self.video_inspector.node_mapping['left_ear'],
            right_ear=self.video_inspector.node_mapping['right_ear'],
            back=self.video_inspector.node_mapping['back'],
            nose=self.video_inspector.node_mapping['nose'],
            head=self.video_inspector.node_mapping['head']
        )

        # Create combined detector
        combined_detector = CombinedDetector(
            data_loader=data_loader,
            ear_config=config.ear_detector,
            head_config=config.head_detector,
            node_mapping=node_mapping
        )

        # Run detection on the specified range
        combined_events, ear_events, head_events = combined_detector.detect_headshakes(
            start_frame=start_frame,
            end_frame=end_frame,
            instance=0,  # Assuming single instance
            iou_threshold=config.iou_threshold
        )

        # Convert to DataFrame
        if combined_events:
            return pd.DataFrame(combined_events)
        else:
            return pd.DataFrame(columns=['start_frame', 'end_frame', 'confidence', 'detection_method'])

    # ==================== Training Functions ====================

    def refresh_training_status(self):
        """Scan training folder and display statistics with guidance."""
        if not self.project_manager:
            self.training_status_label.setText("⚠ No project loaded.")
            self.training_status_label.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 4px;")
            self.train_model_btn.setEnabled(False)
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            self.training_status_label.setText("⚠ No project loaded.")
            self.training_status_label.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 4px;")
            self.train_model_btn.setEnabled(False)
            return

        training_folder = os.path.join(project_path, "training")

        # Find CSV files in training folder
        training_csvs = glob.glob(os.path.join(training_folder, "*.csv"))

        if not training_csvs:
            self.training_status_label.setText(
                "⚠ <b>No training data found.</b><br>"
                "Go to the <b>Prepare Data</b> tab and label some ground truth events first."
            )
            self.training_status_label.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 4px;")
            self.train_model_btn.setEnabled(False)
            return

        # Count labels across all files
        total_htr = 0
        total_non_htr = 0
        total_events = 0

        import pandas as pd
        for csv_path in training_csvs:
            try:
                df = pd.read_csv(csv_path)
                if 'ground_truth' in df.columns:
                    df['ground_truth'] = df['ground_truth'].astype(str)
                    total_htr += len(df[df['ground_truth'].isin(['1', '1.0'])])
                    total_non_htr += len(df[df['ground_truth'].isin(['0', '0.0'])])
            except:
                pass  # Skip problematic files

        total_events = total_htr + total_non_htr

        # Determine if sufficient data
        recommended_min = 100
        recommended_ideal = 200

        if total_events < recommended_min:
            status_color = "#fff3cd"  # Yellow warning
            status_icon = "⚠"
            guidance = f"<br><i>Recommendation: Label at least {recommended_min-total_events} more events for reliable training.</i>"
        elif total_events < recommended_ideal:
            status_color = "#d1ecf1"  # Blue info
            status_icon = "ℹ"
            guidance = f"<br><i>Good start! {recommended_ideal-total_events} more events recommended for optimal performance.</i>"
        else:
            status_color = "#d4edda"  # Green success
            status_icon = "✅"
            guidance = "<br><i>Excellent! You have sufficient training data.</i>"

        # Calculate class balance
        class_balance = (total_htr/total_events*100) if total_events > 0 else 0

        # Display status
        self.training_status_label.setText(
            f"{status_icon} <b>Training Data Status:</b><br>"
            f"• <b>{len(training_csvs)} CSV files</b> in training folder<br>"
            f"• <b>{total_events} labeled events:</b> {total_htr} HTR, {total_non_htr} Non-HTR<br>"
            f"• <b>Class balance:</b> {class_balance:.1f}% positive"
            f"{guidance}"
        )
        self.training_status_label.setStyleSheet(f"background-color: {status_color}; padding: 10px; border-radius: 4px;")

        # Enable train button if sufficient data
        self.train_model_btn.setEnabled(total_events >= 50)  # Absolute minimum

    def browse_training_params(self):
        """Browse for training parameters file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parameters File",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.training_param_edit.setText(file_path)

    def train_model(self):
        """Train HTR detection model using all files in training folder."""
        if not self.project_manager:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            QMessageBox.warning(self, "Error", "No project loaded.")
            return

        training_folder = os.path.join(project_path, "training")
        training_csvs = glob.glob(os.path.join(training_folder, "*.csv"))

        if not training_csvs:
            QMessageBox.warning(
                self, "No Training Data",
                "No training data found. Label some ground truth events first."
            )
            return

        # Confirm training
        reply = QMessageBox.question(
            self,
            "Train Model",
            f"Train HTR detection model using {len(training_csvs)} labeled CSV file(s)?\n\n"
            f"This may take several minutes. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.start_training_progress("Model training")

        try:
            sys.path.append(os.path.dirname(project_path))
            from core.ml_models import ModelTrainer
            from core.config import ConfigManager
            import pandas as pd

            # Load and combine all training CSVs
            self.show_training_progress(f"Loading {len(training_csvs)} training files...")
            combined_df = pd.concat([pd.read_csv(f) for f in training_csvs], ignore_index=True)

            # Filter out unlabeled rows
            combined_df = combined_df[combined_df['ground_truth'] != '__']

            self.show_training_progress(f"Combined dataset: {len(combined_df)} labeled events")

            # Save combined CSV temporarily
            temp_combined_path = os.path.join(training_folder, "_combined_training_data.csv")
            combined_df.to_csv(temp_combined_path, index=False)

            # Load parameters
            param_path = self.training_param_edit.text().strip()
            config_manager = ConfigManager()
            if param_path and os.path.exists(param_path):
                config_manager.import_parameters(param_path)
                self.show_training_progress(f"Using parameters from: {os.path.basename(param_path)}")
            else:
                self.show_training_progress("Using default parameters")

            # Set up trainer
            trainer = ModelTrainer(config_manager)
            models_folder = os.path.join(project_path, "models")
            os.makedirs(models_folder, exist_ok=True)

            # Create model filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = project_config.get("project_name", "HTR_Model")
            model_file = os.path.join(models_folder, f"{project_name}_Model_{timestamp}.joblib")

            self.show_training_progress("Starting training...")

            # Train model using ModelTrainer with combined CSV
            features_folder = os.path.join(project_path, "features")
            results = trainer.train_model(features_folder, temp_combined_path, model_file)

            if not results.get('success', False):
                error_msg = results.get('error', 'Unknown error')
                self.show_training_progress(f"❌ Training failed: {error_msg}")
                self.finish_training_progress("Model training", False)
                QMessageBox.critical(self, "Training Failed", f"Model training failed:\n{error_msg}")
                return

            # Training successful
            training_details = results.get('training_details', {})
            val_results = training_details.get('validation_results', {})

            self.show_training_progress("✅ Model training completed!")
            self.show_training_progress(f"Model saved: {os.path.basename(model_file)}")

            # Display metrics
            accuracy = val_results.get('accuracy', 0)
            precision = val_results.get('precision', 0)
            recall = val_results.get('recall', 0)
            f1_score = val_results.get('f1_score', 0)

            self.show_training_progress(f"Validation Accuracy: {accuracy:.3f}")
            self.show_training_progress(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}")

            # Generate confusion matrix and misclassified events
            self._generate_training_analysis(temp_combined_path, model_file, project_path, training_details)

            # Enable evaluation buttons
            self.load_misclass_btn.setEnabled(True)
            self.view_confusion_btn.setEnabled(True)

            # Update metrics display
            self.metrics_label.setText(
                f"📊 Model Performance (Validation Set):\n"
                f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}\n"
                f"Review misclassified events to improve model."
            )
            self.metrics_label.setStyleSheet(
                "background-color: #d4edda; padding: 8px; border-radius: 4px; color: #155724;"
            )

            self.finish_training_progress("Model training", True)

            QMessageBox.information(
                self,
                "Training Complete",
                "Model trained successfully!\n\nReview the misclassified events to improve accuracy."
            )

        except ImportError as e:
            self.show_training_progress(f"❌ Core modules not available: {str(e)}")
            self.finish_training_progress("Model training", False)
            QMessageBox.critical(self, "Import Error", f"Required modules not found:\n{str(e)}")
        except Exception as e:
            self.show_training_progress(f"❌ Error during training: {str(e)}")
            self.finish_training_progress("Model training", False)
            QMessageBox.critical(self, "Error", f"Model training failed:\n{str(e)}")

    def _generate_training_analysis(self, csv_path, model_path, project_path, training_details):
        """Generate confusion matrix and misclassified events analysis."""
        try:
            import pandas as pd
            from core.ml_models import HTRClassifier, ModelEvaluator
            from sklearn.model_selection import train_test_split

            # Create analysis and plots folders
            analysis_folder = os.path.join(project_path, "analysis")
            plots_folder = os.path.join(project_path, "plots")
            os.makedirs(analysis_folder, exist_ok=True)
            os.makedirs(plots_folder, exist_ok=True)

            # Load the data
            df = pd.read_csv(csv_path)

            # Split features and labels (same split as training)
            feature_cols = [col for col in df.columns if col not in ['ground_truth', 'rat_id', 'start_frame', 'end_frame']]
            X = df[feature_cols]
            y = df['ground_truth']

            # Use same split as training (20% validation)
            _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            val_indices = X_val.index

            # Load model and predict
            classifier = HTRClassifier()
            classifier.load_model(model_path)
            predictions, _ = classifier.predict(X_val)

            # Generate confusion matrix plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_plot_path = os.path.join(plots_folder, f"confusion_matrix_{timestamp}.png")
            ModelEvaluator.plot_confusion_matrix(y_val.values, predictions, cm_plot_path)
            self.show_training_progress(f"Confusion matrix saved: {os.path.basename(cm_plot_path)}")

            # Find misclassified events
            misclassified_mask = y_val.values != predictions
            misclassified_indices = val_indices[misclassified_mask]

            if len(misclassified_indices) > 0:
                # Create misclassified events CSV
                misclass_df = df.loc[misclassified_indices].copy()
                misclass_df['predicted_label'] = predictions[misclassified_mask]
                misclass_df['error_type'] = misclass_df.apply(
                    lambda row: 'False Positive' if row['predicted_label'] == 1 else 'False Negative',
                    axis=1
                )

                # Save to CSV
                misclass_csv_path = os.path.join(analysis_folder, f"misclassified_events_{timestamp}.csv")
                misclass_df.to_csv(misclass_csv_path, index=False)
                self.show_training_progress(f"Misclassified events saved: {os.path.basename(misclass_csv_path)} ({len(misclassified_indices)} events)")
            else:
                self.show_training_progress("No misclassified events (perfect model!)")

        except Exception as e:
            self.show_training_progress(f"⚠ Warning: Could not generate analysis outputs: {str(e)}")

    def load_misclassified_events(self):
        """Load and display misclassified events CSV."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return

        # Look for misclassified events CSV in analysis folder
        analysis_folder = os.path.join(project_path, "analysis")
        if not os.path.exists(analysis_folder):
            QMessageBox.information(
                self,
                "No Analysis Files",
                "No analysis folder found. Train a model first to generate misclassified events."
            )
            return

        misclass_files = glob.glob(os.path.join(analysis_folder, "misclassified_events_*.csv"))
        if not misclass_files:
            QMessageBox.information(
                self,
                "No Misclassified Events",
                "No misclassified events file found. The model may be perfect, or training output is missing."
            )
            return

        # Use most recent file
        latest_file = max(misclass_files, key=os.path.getmtime)

        try:
            # Load and display CSV in table
            import pandas as pd
            df = pd.read_csv(latest_file)

            # Show table
            self.misclass_table.setVisible(True)
            self.misclass_table.setRowCount(len(df))

            # Display key columns
            display_cols = ['error_type', 'start_frame', 'end_frame']
            # Add rat_id or file column if available
            if 'rat_id' in df.columns:
                display_cols.append('rat_id')
            elif 'file' in df.columns:
                display_cols.append('file')
            else:
                display_cols.append('predicted_label')  # fallback

            for i, (idx, row) in enumerate(df.iterrows()):
                # Error type
                error_type = row.get('error_type', 'Unknown')
                self.misclass_table.setItem(i, 0, QTableWidgetItem(str(error_type)))

                # Start/End frames
                self.misclass_table.setItem(i, 1, QTableWidgetItem(str(int(row.get('start_frame', 0)))))
                self.misclass_table.setItem(i, 2, QTableWidgetItem(str(int(row.get('end_frame', 0)))))

                # File/rat info
                file_info = row.get('rat_id', row.get('file', row.get('predicted_label', '')))
                self.misclass_table.setItem(i, 3, QTableWidgetItem(str(file_info)))

                # Notes (ground truth vs predicted)
                gt = row.get('ground_truth', '')
                pred = row.get('predicted_label', '')
                notes = f"GT: {gt}, Pred: {pred}"
                self.misclass_table.setItem(i, 4, QTableWidgetItem(notes))

            # Resize columns
            self.misclass_table.resizeColumnsToContents()

            self.show_training_progress(f"Loaded {len(df)} misclassified events from {os.path.basename(latest_file)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load misclassified events:\n{str(e)}")
            self.show_training_progress(f"❌ Error loading misclassified events: {str(e)}")

    def view_confusion_matrix(self):
        """View confusion matrix plot."""
        if not self.project_manager:
            return

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return

        # Look for confusion matrix plot in plots folder
        plots_folder = os.path.join(project_path, "plots")
        if not os.path.exists(plots_folder):
            QMessageBox.information(
                self,
                "No Plots",
                "No plots folder found. Train a model first."
            )
            return

        confusion_files = glob.glob(os.path.join(plots_folder, "confusion_matrix_*.png"))
        if not confusion_files:
            QMessageBox.information(
                self,
                "No Confusion Matrix",
                "No confusion matrix plot found."
            )
            return

        # Use most recent file
        latest_plot = max(confusion_files, key=os.path.getmtime)

        try:
            # Create image viewer dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Confusion Matrix - {os.path.basename(latest_plot)}")
            dialog.setMinimumSize(800, 600)

            layout = QVBoxLayout(dialog)

            # Load and display image
            pixmap = QPixmap(latest_plot)
            if pixmap.isNull():
                QMessageBox.warning(self, "Error", f"Could not load image:\n{latest_plot}")
                return

            # Create label for image
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(780, 550, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)

            # Add close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setMaximumWidth(100)

            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            btn_layout.addWidget(close_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)

            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display confusion matrix:\n{str(e)}")

    def start_training_progress(self, operation_name):
        """Start training progress tracking."""
        self.training_progress_bar.setVisible(True)
        self.training_progress_bar.setRange(0, 0)  # Indeterminate
        self.show_training_progress(f"Starting {operation_name}...")

    def finish_training_progress(self, operation_name, success=True):
        """Finish training progress tracking."""
        self.training_progress_bar.setVisible(False)
        status = "completed successfully" if success else "failed"
        self.show_training_progress(f"{operation_name} {status}.")

    def show_training_progress(self, message):
        """Show training progress message."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        formatted_message = f"[{timestamp}] {message}"
        self.training_results_text.append(formatted_message)

        # Auto-scroll
        cursor = self.training_results_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.training_results_text.setTextCursor(cursor)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("HTR Analysis Tool v3")

    window = HTRAnalysisAppV3()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
