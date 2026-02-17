"""
HTR Analysis Tool - Main Window v3 (Restructured)

New 5-tab workflow structure:
1. Welcome - Project overview and workflow navigation
2. Tune Parameters - Parameter optimization with video feedback
3. Prepare Data - Feature extraction + ground truth labeling
4. Train Model - ML training with evaluation and iteration
5. Identify Headtwitches - Smart batch processing (fresh and incremental)

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
    QTableWidgetItem, QHeaderView, QSplitter, QScrollArea, QCheckBox
)
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QFont, QPixmap, QShortcut, QKeySequence

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

try:
    from .theme import Colors, Fonts, Spacing, get_icon, stylesheet_button_success, stylesheet_button_primary, stylesheet_status_info, stylesheet_status_warning, stylesheet_status_success, stylesheet_status_dynamic, stylesheet_log_area, stylesheet_separator
except ImportError:
    from theme import Colors, Fonts, Spacing, get_icon, stylesheet_button_success, stylesheet_button_primary, stylesheet_status_info, stylesheet_status_warning, stylesheet_status_success, stylesheet_status_dynamic, stylesheet_log_area, stylesheet_separator


class CachedDataLoader:
    """
    Lightweight data loader that wraps already-loaded tracking data.

    Provides the same interface as SleapDataLoader but uses cached data
    from VideoInspectorWidget instead of reloading from disk.
    """

    def __init__(self, tracking_data, point_scores, node_names=None):
        """
        Initialize with pre-loaded data.

        Args:
            tracking_data: Array of shape (n_frames, n_nodes, 2)
            point_scores: Array of shape (n_frames, n_nodes)
            node_names: Optional list of node names
        """
        import numpy as np

        # Add instance dimension to match SleapDataLoader format
        # SleapDataLoader.locations is (n_frames, n_nodes, 2, n_instances)
        self.locations = tracking_data[:, :, :, np.newaxis]

        # SleapDataLoader.point_scores is (n_frames, n_nodes, n_instances)
        self.point_scores = point_scores[:, :, np.newaxis]

        self.node_names = node_names
        self.total_frames = tracking_data.shape[0]

    def get_node_positions(self, node_idx, start_frame=0, end_frame=None, instance=0):
        """Get positions for a specific node across frames."""
        if end_frame is None:
            end_frame = self.total_frames
        return self.locations[start_frame:end_frame, node_idx, :, instance]

    def get_node_scores(self, node_idx, start_frame=0, end_frame=None, instance=0):
        """Get confidence scores for a specific node across frames."""
        if end_frame is None:
            end_frame = self.total_frames
        return self.point_scores[start_frame:end_frame, node_idx, instance]


class HTRAnalysisAppV3(QMainWindow):
    """Main application window with 5-tab workflow structure."""

    def __init__(self):
        super().__init__()
        self.config_manager = get_config_manager()
        self.project_manager = None
        self.cached_data_loader = None  # Cached data loader for fast reanalysis
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("HTR Analysis Tool v3")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 750)

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
        # Tab widget fills available space (no max height constraint)
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_tabs()

        # Create status bar
        self.status_bar = self.statusBar()
        self.status_message_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_message_label)
        self.project_status_label = QLabel("No project loaded")
        self.status_bar.addPermanentWidget(self.project_status_label)

        # Set up keyboard shortcuts for tab navigation
        self.setup_keyboard_shortcuts()

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

        # View menu — tab navigation
        view_menu = menubar.addMenu("&View")

        tab_names = [
            ("1. Welcome", "Ctrl+1"),
            ("2. Tune Parameters", "Ctrl+2"),
            ("3. Prepare Data", "Ctrl+3"),
            ("4. Train Model", "Ctrl+4"),
            ("5. Identify Headtwitches", "Ctrl+5"),
        ]
        for idx, (name, shortcut) in enumerate(tab_names):
            action = view_menu.addAction(f"{name}\t{shortcut}")
            action.triggered.connect(lambda checked, i=idx: self.switch_to_tab(i))

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

        # Tab 5: Identify Headtwitches (formerly Deploy/Batch Process)
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
            self.tab_widget.addTab(self.welcome_tab, "1. Welcome")
        except Exception as e:
            # Fallback if welcome tab fails
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Welcome Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "1. Welcome")

    def create_tune_parameters_tab(self):
        """Tab 2: Tune Parameters with video feedback."""
        # Use a horizontal splitter so user can resize video vs graph/params
        h_splitter = QSplitter(Qt.Horizontal)
        h_splitter.setContentsMargins(5, 5, 5, 5)

        # LEFT SIDE: Video Inspector
        try:
            self.video_inspector = VideoInspectorWidget(parent=self)
            h_splitter.addWidget(self.video_inspector)
        except Exception as e:
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Video Inspector Error: {str(e)}"))
            h_splitter.addWidget(fallback_widget)
            self.video_inspector = None

        # RIGHT SIDE: Vertical splitter for Graph + Parameter Panel
        right_splitter = QSplitter(Qt.Vertical)

        # Top: Diagnostics Graph (resizable)
        try:
            self.diagnostics_graph = DiagnosticsGraphWidget(parent=self)
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

        # Set initial sizes for vertical splitter: graph ~455px, parameters ~245px
        right_splitter.setSizes([455, 245])

        h_splitter.addWidget(right_splitter)

        # Set initial sizes for horizontal splitter: video ~690px, right ~690px
        h_splitter.setSizes([690, 690])

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

        self.tab_widget.addTab(h_splitter, "2. Tune Parameters")

    def create_prepare_data_tab(self):
        """Tab 3: Prepare Data - Feature extraction + Ground truth labeling."""
        try:
            self.prepare_data_tab = PrepareDataTab(parent=self, project_manager=self.project_manager)
            # Connect signals
            self.prepare_data_tab.features_extracted.connect(self.on_features_extracted)
            self.prepare_data_tab.labels_updated.connect(self.on_labels_updated)
            self.tab_widget.addTab(self.prepare_data_tab, "3. Prepare Data")
        except Exception as e:
            # Fallback
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Prepare Data Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "3. Prepare Data")

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
        separator.setStyleSheet(stylesheet_separator())
        layout.addWidget(separator)

        # Section B: Evaluate & Iterate (NEW)
        self.create_training_evaluate_section(layout)

        # Progress section
        self.create_training_progress_section(layout)

        layout.addStretch()

        self.tab_widget.addTab(training_widget, "4. Train Model")

    def create_training_config_section(self, parent_layout):
        """Training configuration section."""
        config_group = QGroupBox("Configure & Train Model")
        config_group.setFont(QFont(Fonts.FAMILY, 10, QFont.Bold))
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 15, 10, 10)
        config_layout.setSpacing(10)

        # Training data status display
        self.training_status_label = QLabel("Loading training data...")
        self.training_status_label.setFont(QFont(Fonts.FAMILY, 9))
        self.training_status_label.setWordWrap(True)
        self.training_status_label.setStyleSheet(stylesheet_status_info())
        config_layout.addWidget(self.training_status_label)

        # Refresh button
        refresh_training_btn = QPushButton("Refresh Training Data")
        refresh_training_btn.setIcon(get_icon('reload'))
        refresh_training_btn.setFont(QFont(Fonts.FAMILY, 9))
        refresh_training_btn.clicked.connect(self.refresh_training_status)
        config_layout.addWidget(refresh_training_btn)

        # Feature Selection group
        feature_sel_group = QGroupBox("Feature Selection")
        feature_sel_group.setFont(QFont(Fonts.FAMILY, 9))
        feature_sel_layout = QVBoxLayout(feature_sel_group)
        feature_sel_layout.setContentsMargins(8, 10, 8, 8)
        feature_sel_layout.setSpacing(4)

        # Buttons row + count label
        feat_btn_layout = QHBoxLayout()
        self.feat_select_all_btn = QPushButton("Select All")
        self.feat_select_all_btn.setFont(QFont(Fonts.FAMILY, 8))
        self.feat_select_all_btn.setMaximumWidth(80)
        self.feat_select_all_btn.clicked.connect(self._select_all_features)
        feat_btn_layout.addWidget(self.feat_select_all_btn)

        self.feat_deselect_all_btn = QPushButton("Deselect All")
        self.feat_deselect_all_btn.setFont(QFont(Fonts.FAMILY, 8))
        self.feat_deselect_all_btn.setMaximumWidth(80)
        self.feat_deselect_all_btn.clicked.connect(self._deselect_all_features)
        feat_btn_layout.addWidget(self.feat_deselect_all_btn)

        self.feature_count_label = QLabel("No features loaded")
        self.feature_count_label.setFont(QFont(Fonts.FAMILY, 8))
        feat_btn_layout.addWidget(self.feature_count_label)
        feat_btn_layout.addStretch()
        feature_sel_layout.addLayout(feat_btn_layout)

        # Scrollable checkbox area
        self.feature_scroll_area = QScrollArea()
        self.feature_scroll_area.setMaximumHeight(200)
        self.feature_scroll_area.setWidgetResizable(True)
        self.feature_checkbox_widget = QWidget()
        self.feature_checkbox_layout = QVBoxLayout(self.feature_checkbox_widget)
        self.feature_checkbox_layout.setContentsMargins(4, 4, 4, 4)
        self.feature_checkbox_layout.setSpacing(2)
        self.feature_scroll_area.setWidget(self.feature_checkbox_widget)
        feature_sel_layout.addWidget(self.feature_scroll_area)

        self.feature_checkboxes = {}

        config_layout.addWidget(feature_sel_group)

        # Parameters file (optional)
        param_layout = QHBoxLayout()
        param_label = QLabel("Parameters:")
        param_label.setMinimumWidth(120)
        param_label.setFont(QFont(Fonts.FAMILY, 9))
        param_layout.addWidget(param_label)

        self.training_param_edit = QLineEdit()
        self.training_param_edit.setPlaceholderText("Optional: Parameter configuration")
        self.training_param_edit.setFont(QFont(Fonts.FAMILY, 9))
        param_layout.addWidget(self.training_param_edit)

        browse_param_btn = QPushButton("Browse...")
        browse_param_btn.setMaximumWidth(80)
        browse_param_btn.clicked.connect(self.browse_training_params)
        param_layout.addWidget(browse_param_btn)

        config_layout.addLayout(param_layout)

        # Train button
        train_layout = QHBoxLayout()

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.setIcon(get_icon('computer'))
        self.train_model_btn.setFont(QFont(Fonts.FAMILY, 10, QFont.Bold))
        self.train_model_btn.setStyleSheet(stylesheet_button_success())
        self.train_model_btn.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_model_btn)

        train_layout.addStretch()

        config_layout.addLayout(train_layout)

        parent_layout.addWidget(config_group)

    def create_training_evaluate_section(self, parent_layout):
        """Training evaluation and iteration section."""
        eval_group = QGroupBox("Evaluate & Iterate")
        eval_group.setFont(QFont(Fonts.FAMILY, 10, QFont.Bold))
        eval_layout = QVBoxLayout(eval_group)
        eval_layout.setContentsMargins(10, 15, 10, 10)
        eval_layout.setSpacing(10)

        # Instructions
        instructions = QLabel(
            "<b>Review model performance:</b> "
            "Analyze misclassified events, fix labels, and retrain to improve accuracy."
        )
        instructions.setFont(QFont(Fonts.FAMILY, 9))
        instructions.setWordWrap(True)
        eval_layout.addWidget(instructions)

        # Metrics display
        self.metrics_label = QLabel("Train a model to see performance metrics")
        self.metrics_label.setFont(QFont(Fonts.FAMILY, 9))
        self.metrics_label.setStyleSheet(stylesheet_status_info())
        eval_layout.addWidget(self.metrics_label)

        # Misclassified events section
        misclass_layout = QHBoxLayout()

        # Load misclassified button
        self.load_misclass_btn = QPushButton("Load Misclassified Events")
        self.load_misclass_btn.setIcon(get_icon('chart'))
        self.load_misclass_btn.setFont(QFont(Fonts.FAMILY, 9))
        self.load_misclass_btn.clicked.connect(self.load_misclassified_events)
        self.load_misclass_btn.setEnabled(False)
        misclass_layout.addWidget(self.load_misclass_btn)

        # View confusion matrix button
        self.view_confusion_btn = QPushButton("View Confusion Matrix")
        self.view_confusion_btn.setIcon(get_icon('chart'))
        self.view_confusion_btn.setFont(QFont(Fonts.FAMILY, 9))
        self.view_confusion_btn.clicked.connect(self.view_confusion_matrix)
        self.view_confusion_btn.setEnabled(False)
        misclass_layout.addWidget(self.view_confusion_btn)

        # View feature importance button
        self.view_importance_btn = QPushButton("View Feature Importance")
        self.view_importance_btn.setIcon(get_icon('chart'))
        self.view_importance_btn.setFont(QFont(Fonts.FAMILY, 9))
        self.view_importance_btn.clicked.connect(self.view_feature_importance)
        self.view_importance_btn.setEnabled(False)
        misclass_layout.addWidget(self.view_importance_btn)

        # View threshold curve button
        self.view_threshold_btn = QPushButton("View Threshold Curve")
        self.view_threshold_btn.setIcon(get_icon('chart'))
        self.view_threshold_btn.setFont(QFont(Fonts.FAMILY, 9))
        self.view_threshold_btn.clicked.connect(self.view_threshold_curve)
        self.view_threshold_btn.setEnabled(False)
        misclass_layout.addWidget(self.view_threshold_btn)

        # View SHAP analysis button
        self.view_shap_btn = QPushButton("View SHAP Analysis")
        self.view_shap_btn.setIcon(get_icon('search'))
        self.view_shap_btn.setFont(QFont(Fonts.FAMILY, 9))
        self.view_shap_btn.clicked.connect(self.view_shap_analysis)
        self.view_shap_btn.setEnabled(False)
        misclass_layout.addWidget(self.view_shap_btn)

        misclass_layout.addStretch()

        eval_layout.addLayout(misclass_layout)

        # Misclassified events table (compact)
        self.misclass_table = QTableWidget()
        self.misclass_table.setMaximumHeight(150)
        self.misclass_table.setColumnCount(6)
        self.misclass_table.setHorizontalHeaderLabels([
            "Error Type", "Start Frame", "End Frame", "Confidence", "File", "Notes"
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
        self.training_results_text.setFont(QFont(Fonts.MONO_FAMILY, 8))
        self.training_results_text.setReadOnly(True)
        self.training_results_text.setPlaceholderText("Training progress will appear here...")
        self.training_results_text.setStyleSheet(stylesheet_log_area())
        progress_layout.addWidget(self.training_results_text)

        parent_layout.addWidget(progress_group)

    def create_deploy_tab(self):
        """Tab 5: Identify Headtwitches - Smart batch processing."""
        try:
            self.deploy_tab = DeployTab(parent=self, project_manager=self.project_manager)
            # Connect signals
            self.deploy_tab.processing_complete.connect(self.on_processing_complete)
            self.tab_widget.addTab(self.deploy_tab, "5. Identify Headtwitches")
        except Exception as e:
            # Fallback
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel(f"Identify Headtwitches Tab Error: {str(e)}"))
            self.tab_widget.addTab(fallback_widget, "5. Identify Headtwitches")

    # ==================== Tab Navigation ====================

    def setup_keyboard_shortcuts(self):
        """Set up Ctrl+1 through Ctrl+5 for tab navigation."""
        for i in range(5):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i+1}"), self)
            shortcut.activated.connect(lambda idx=i: self.switch_to_tab(idx))

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

        # Update Identify Headtwitches tab
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
        # Refresh Identify Headtwitches tab status
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
        """Handle H5 file loaded event - enable analysis buttons and cache data loader."""
        if self.parameter_panel:
            self.parameter_panel.enable_analysis_buttons(True)

        # Create cached data loader from video inspector's already-loaded data
        if self.video_inspector and self.video_inspector.tracking_data is not None:
            self.cached_data_loader = CachedDataLoader(
                tracking_data=self.video_inspector.tracking_data,
                point_scores=self.video_inspector.point_scores,
                node_names=None  # Node names not needed for detection
            )

    def reanalyze_current_view(self):
        """Run detection on the visible graph range."""
        import time

        if not self.video_inspector or not self.video_inspector.signals_df is not None:
            QMessageBox.warning(self, "No Data", "Please load an H5 file first.")
            return

        if not self.diagnostics_graph:
            return

        # Get visible range from graph
        start, end = self.diagnostics_graph.get_view_range()
        print(f"[DEBUG] Reanalyzing frames {start}-{end} ({end-start} frames)")
        print(f"[DEBUG] Cached data loader: {'YES' if self.cached_data_loader else 'NO'}")

        # Run detection on this range
        try:
            t0 = time.time()
            events_df = self.run_detection_on_range(start, end)
            elapsed = time.time() - t0
            print(f"[DEBUG] Detection completed in {elapsed:.2f}s, found {len(events_df)} events")

            # Update graph overlays
            self.diagnostics_graph.update_events(events_df)

            # Update peak markers with current parameters
            self._update_peak_markers()
            print(f"[DEBUG] Graph updated")

            # Update status
            if self.parameter_panel:
                self.parameter_panel.status_label.setText(
                    f"Analysis complete: {len(events_df)} events detected in frames {start}-{end}"
                )
                print(f"[DEBUG] Status label updated")

        except Exception as e:
            print(f"[DEBUG] ERROR: {e}")
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
                # Update peak markers with current parameters
                self._update_peak_markers()

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

        Uses cached data loader for fast reanalysis (no H5 reload).

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
            from core.config import NodeMapping
        except ImportError as e:
            raise ImportError(f"Could not import detection modules: {e}")

        # Get current parameters from config manager
        if not self.config_manager:
            raise ValueError("Config manager not available")

        config = self.config_manager.config

        # Use cached data loader (fast) instead of reloading from disk
        if not self.cached_data_loader:
            raise ValueError("No H5 data loaded. Please load an H5 file first.")

        # Create NodeMapping from video inspector's mapping
        node_mapping = NodeMapping(
            left_ear=self.video_inspector.node_mapping['left_ear'],
            right_ear=self.video_inspector.node_mapping['right_ear'],
            back=self.video_inspector.node_mapping['back'],
            nose=self.video_inspector.node_mapping['nose'],
            head=self.video_inspector.node_mapping['head']
        )

        # Create combined detector using cached data
        combined_detector = CombinedDetector(
            data_loader=self.cached_data_loader,
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

    def _update_peak_markers(self):
        """
        Update peak/valley markers on the diagnostics graph with current parameters.

        Parameters are mapped to match the actual detector logic in core/detectors.py:
        - EarsDetector uses height (peak_threshold, valley_threshold) and distance (max_gap)
        - HeadDetector uses prominence (peak_prominence), distance (peak_distance),
          and amplitude_threshold for cycle filtering
        - Oscillation grouping uses max_cycle_gap, min_oscillations, between_unit_gap, etc.
        """
        if not self.diagnostics_graph or not self.config_manager:
            return

        config = self.config_manager.config

        # Head detector params - uses prominence, distance, amplitude threshold, grouping, and smoothing params
        head_prominence = config.head_detector.peak_prominence
        head_distance = config.head_detector.peak_distance
        head_amplitude_threshold = config.head_detector.amplitude_threshold
        head_max_cycle_gap = config.head_detector.max_cycle_gap
        head_min_oscillations = config.head_detector.min_oscillations
        head_use_smoothing = config.head_detector.use_smoothing
        head_smoothing_window = config.head_detector.smoothing_window
        head_smoothing_polyorder = config.head_detector.smoothing_polyorder

        # Ear detector params - mode selection and thresholds
        ear_use_prominence_mode = config.ear_detector.use_prominence_mode
        ear_prominence = config.ear_detector.ear_prominence
        ear_peak_height = config.ear_detector.peak_threshold
        ear_valley_height = config.ear_detector.valley_threshold
        ear_distance = config.ear_detector.max_gap
        ear_quick_gap = config.ear_detector.quick_gap
        ear_between_unit_gap = config.ear_detector.between_unit_gap
        ear_min_crisscrosses = config.ear_detector.min_crisscrosses

        # Update peaks/valleys on the graph
        self.diagnostics_graph.update_peaks(
            head_prominence=head_prominence,
            head_distance=head_distance,
            head_amplitude_threshold=head_amplitude_threshold,
            head_max_cycle_gap=head_max_cycle_gap,
            head_min_oscillations=head_min_oscillations,
            head_use_smoothing=head_use_smoothing,
            head_smoothing_window=head_smoothing_window,
            head_smoothing_polyorder=head_smoothing_polyorder,
            ear_use_prominence_mode=ear_use_prominence_mode,
            ear_prominence=ear_prominence,
            ear_peak_height=ear_peak_height,
            ear_valley_height=ear_valley_height,
            ear_distance=ear_distance,
            ear_quick_gap=ear_quick_gap,
            ear_between_unit_gap=ear_between_unit_gap,
            ear_min_crisscrosses=ear_min_crisscrosses
        )

    # ==================== Training Functions ====================

    def _select_all_features(self):
        """Check all feature checkboxes."""
        for cb in self.feature_checkboxes.values():
            cb.setChecked(True)
        self._update_feature_count()

    def _deselect_all_features(self):
        """Uncheck all feature checkboxes."""
        for cb in self.feature_checkboxes.values():
            cb.setChecked(False)
        self._update_feature_count()

    def _update_feature_count(self):
        """Update the feature count label."""
        total = len(self.feature_checkboxes)
        selected = sum(1 for cb in self.feature_checkboxes.values() if cb.isChecked())
        self.feature_count_label.setText(f"Using {selected}/{total} features")

    def _get_selected_feature_names(self):
        """Return list of currently checked feature names."""
        return [name for name, cb in self.feature_checkboxes.items() if cb.isChecked()]

    def _populate_feature_checkboxes(self, training_csvs):
        """Read feature column names from training CSVs and populate checkboxes."""
        import pandas as pd

        # Clear existing checkboxes
        for cb in self.feature_checkboxes.values():
            cb.setParent(None)
        self.feature_checkboxes.clear()

        metadata_cols = {
            'ground_truth', 'rat_id', 'dose', 'drug', 'cohort', 'source_file',
            'start_frame', 'end_frame', 'prediction', 'prediction_confidence'
        }

        # Collect feature columns from headers only (no full read)
        all_feature_cols = []
        for csv_path in training_csvs:
            try:
                header_df = pd.read_csv(csv_path, nrows=0)
                cols = [c for c in header_df.columns if c not in metadata_cols]
                for c in cols:
                    if c not in all_feature_cols:
                        all_feature_cols.append(c)
            except Exception:
                pass

        if not all_feature_cols:
            self.feature_count_label.setText("No features loaded")
            return

        # Get saved selection (None = use all)
        saved_selection = None
        if self.project_manager:
            saved_selection = self.project_manager.get_selected_features()

        # Create checkboxes
        for col_name in all_feature_cols:
            cb = QCheckBox(col_name)
            cb.setFont(QFont(Fonts.FAMILY, 8))
            if saved_selection is not None:
                cb.setChecked(col_name in saved_selection)
            else:
                cb.setChecked(True)
            cb.stateChanged.connect(lambda _: self._update_feature_count())
            self.feature_checkbox_layout.addWidget(cb)
            self.feature_checkboxes[col_name] = cb

        self._update_feature_count()

    def refresh_training_status(self):
        """Scan training folder and display statistics with guidance."""
        if not self.project_manager:
            self.training_status_label.setText("⚠ No project loaded.")
            self.training_status_label.setStyleSheet(stylesheet_status_warning())
            self.train_model_btn.setEnabled(False)
            return

        project_path, project_config = self.project_manager.get_current_project()
        if not project_path:
            self.training_status_label.setText("⚠ No project loaded.")
            self.training_status_label.setStyleSheet(stylesheet_status_warning())
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
            self.training_status_label.setStyleSheet(stylesheet_status_warning())
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
        self.training_status_label.setStyleSheet(stylesheet_status_dynamic(status_color))

        # Enable train button if sufficient data
        self.train_model_btn.setEnabled(total_events >= 50)  # Absolute minimum

        # Populate feature checkboxes from training CSV headers
        self._populate_feature_checkboxes(training_csvs)

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

            # Apply feature selection — keep only selected features + metadata
            selected_features = self._get_selected_feature_names()
            if selected_features and len(selected_features) < len(self.feature_checkboxes):
                metadata_cols_set = {
                    'ground_truth', 'rat_id', 'dose', 'drug', 'cohort', 'source_file',
                    'start_frame', 'end_frame', 'prediction', 'prediction_confidence'
                }
                keep_cols = [c for c in combined_df.columns if c in metadata_cols_set or c in selected_features]
                combined_df = combined_df[keep_cols]
                self.show_training_progress(
                    f"Using {len(selected_features)} of {len(self.feature_checkboxes)} features"
                )
            else:
                self.show_training_progress(f"Using all {len(self.feature_checkboxes)} features")

            # Create per-run subfolder: training/run_YYYY-MM-DD_HH-MM-SS/
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_folder = os.path.join(training_folder, f"run_{timestamp}")
            os.makedirs(run_folder, exist_ok=True)
            self.show_training_progress(f"Run folder: {os.path.basename(run_folder)}")

            # Save combined CSV into the run folder
            temp_combined_path = os.path.join(run_folder, "_combined_training_data.csv")
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

            # Create model filename (use same timestamp as run folder)
            ts_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = project_config.get("project_name", "HTR_Model")
            model_file = os.path.join(models_folder, f"{project_name}_Model_{ts_compact}.joblib")

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

            # Generate confusion matrix and misclassified events into run folder
            self._generate_training_analysis(temp_combined_path, model_file, run_folder, training_details)

            # Save feature selection to project config
            if self.project_manager:
                self.project_manager.save_selected_features(self._get_selected_feature_names())

            # Enable evaluation buttons
            self.load_misclass_btn.setEnabled(True)
            self.view_confusion_btn.setEnabled(True)
            self.view_importance_btn.setEnabled(True)
            self.view_threshold_btn.setEnabled(True)
            self.view_shap_btn.setEnabled(True)

            # Update metrics display
            self.metrics_label.setText(
                f"Model Performance (Validation Set):\n"
                f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}\n"
                f"Review misclassified events to improve model."
            )
            self.metrics_label.setStyleSheet(stylesheet_status_success())

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

    def _generate_training_analysis(self, csv_path, model_path, run_folder, training_details):
        """Generate confusion matrix and misclassified events analysis into a run folder."""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from core.ml_models import HTRClassifier, ModelEvaluator
            from sklearn.model_selection import train_test_split

            # Create analysis and plots folders inside the run folder
            analysis_folder = os.path.join(run_folder, "analysis")
            plots_folder = os.path.join(run_folder, "plots")
            os.makedirs(analysis_folder, exist_ok=True)
            os.makedirs(plots_folder, exist_ok=True)

            # Load the data
            df = pd.read_csv(csv_path)

            # Split features and labels (same split as training, exclude metadata columns)
            metadata_cols = [
                'ground_truth', 'rat_id', 'dose', 'drug', 'cohort', 'source_file',
                'start_frame', 'end_frame', 'prediction', 'prediction_confidence'
            ]
            feature_cols = [col for col in df.columns if col not in metadata_cols]

            # Apply feature selection to match training
            selected_features = self._get_selected_feature_names()
            if selected_features and len(selected_features) < len(self.feature_checkboxes):
                feature_cols = [c for c in feature_cols if c in selected_features]

            X = df[feature_cols]
            y = df['ground_truth']

            # Use same split as training (20% validation)
            _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            val_indices = X_val.index

            # Load model and predict
            classifier = HTRClassifier()
            classifier.load_model(model_path)
            predictions, probabilities = classifier.predict(X_val)

            # Generate confusion matrix plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_plot_path = os.path.join(plots_folder, f"confusion_matrix_{timestamp}.png")
            ModelEvaluator.plot_confusion_matrix(y_val.values, predictions, cm_plot_path)
            self.show_training_progress(f"Confusion matrix saved: {os.path.basename(cm_plot_path)}")

            # Generate feature importance plot
            try:
                importance_df = classifier.get_feature_importance()
                fi_plot_path = os.path.join(plots_folder, f"feature_importance_{timestamp}.png")
                fig_fi = ModelEvaluator.plot_feature_importance(importance_df, save_path=fi_plot_path)
                plt.close(fig_fi)
                self.show_training_progress(f"Feature importance saved: {os.path.basename(fi_plot_path)}")
            except Exception as e:
                self.show_training_progress(f"⚠ Could not generate feature importance plot: {e}")

            # Generate SHAP summary plot
            shap_values_for_log = None
            try:
                shap_plot_path = os.path.join(plots_folder, f"shap_summary_{timestamp}.png")
                fig_shap, shap_vals = ModelEvaluator.plot_shap_summary(
                    classifier.model, X_val, save_path=shap_plot_path
                )
                # Compute mean |SHAP| per feature for the training log
                mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
                shap_values_for_log = sorted(
                    zip(X_val.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
                )
                plt.close(fig_shap)
                self.show_training_progress(f"SHAP summary saved: {os.path.basename(shap_plot_path)}")
            except ImportError:
                self.show_training_progress("SHAP analysis skipped (shap package not installed)")
            except Exception as e:
                self.show_training_progress(f"⚠ Could not generate SHAP analysis: {e}")

            # Generate threshold curve
            optimal_threshold = None
            try:
                fig_tc = ModelEvaluator.plot_threshold_curve(
                    y_val.values, probabilities[:, 1],
                    save_path=os.path.join(plots_folder, f"threshold_curve_{timestamp}.png")
                )
                # Extract optimal threshold from the plot for the training log
                from sklearn.metrics import precision_recall_curve as prc
                prec, rec, thresholds = prc(y_val.values, probabilities[:, 1])
                f1_arr = np.where((prec[:-1] + rec[:-1]) > 0,
                                  2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1]), 0.0)
                optimal_threshold = float(thresholds[np.argmax(f1_arr)])
                plt.close(fig_tc)
                self.show_training_progress(f"Threshold curve saved (optimal threshold: {optimal_threshold:.3f})")
            except Exception as e:
                self.show_training_progress(f"⚠ Could not generate threshold curve: {e}")

            # Generate training log text file
            try:
                log_path = os.path.join(analysis_folder, f"training_log_{timestamp}.txt")
                with open(log_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write(f"  HTR Model Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 70 + "\n\n")

                    f.write(f"Model Path: {model_path}\n\n")

                    # Best hyperparameters
                    best_params = training_details.get('best_params', {})
                    if best_params:
                        f.write("--- Best Hyperparameters ---\n")
                        for k, v in best_params.items():
                            f.write(f"  {k}: {v}\n")
                        f.write("\n")

                    # Classification report
                    val_results = training_details.get('validation_results', {})
                    report = val_results.get('classification_report', {})
                    if report:
                        f.write("--- Classification Report ---\n")
                        # Header
                        f.write(f"{'':>18} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n")
                        for label, metrics in report.items():
                            if isinstance(metrics, dict):
                                f.write(f"{label:>18} {metrics.get('precision', 0):>10.4f} "
                                        f"{metrics.get('recall', 0):>10.4f} "
                                        f"{metrics.get('f1-score', 0):>10.4f} "
                                        f"{metrics.get('support', 0):>10.0f}\n")
                            else:
                                f.write(f"{'accuracy':>18} {'':>10} {'':>10} {metrics:>10.4f}\n")
                        f.write("\n")

                    # Confusion matrix
                    cm = val_results.get('confusion_matrix', None)
                    if cm:
                        f.write("--- Confusion Matrix ---\n")
                        f.write("  (rows = actual, cols = predicted)\n")
                        for row in cm:
                            f.write("  " + "  ".join(f"{v:>6}" for v in row) + "\n")
                        f.write("\n")

                    # All feature importances
                    try:
                        imp_df = classifier.get_feature_importance()
                        f.write("--- All Feature Importances ---\n")
                        for _, row in imp_df.iterrows():
                            f.write(f"  {row['feature']:<40} {row['importance']:.6f}\n")
                        f.write("\n")
                    except Exception:
                        pass

                    # SHAP feature importances
                    if shap_values_for_log:
                        f.write("--- SHAP Feature Importances (mean |SHAP|) ---\n")
                        for feat_name, shap_val in shap_values_for_log:
                            f.write(f"  {feat_name:<40} {shap_val:.6f}\n")
                        f.write("\n")

                    # Optimal threshold
                    if optimal_threshold is not None:
                        f.write(f"--- Optimal Classification Threshold ---\n")
                        f.write(f"  {optimal_threshold:.4f}  (maximizes F1 score)\n\n")

                    f.write("=" * 70 + "\n")
                self.show_training_progress(f"Training log saved: {os.path.basename(log_path)}")
            except Exception as e:
                self.show_training_progress(f"⚠ Could not generate training log: {e}")

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

    def _get_latest_run_folder(self):
        """Find the most recent run_* subfolder inside training/."""
        if not self.project_manager:
            return None
        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return None
        training_folder = os.path.join(project_path, "training")
        run_dirs = glob.glob(os.path.join(training_folder, "run_*"))
        if not run_dirs:
            return None
        return max(run_dirs, key=os.path.getmtime)

    def load_misclassified_events(self):
        """Load and display misclassified events CSV."""
        run_folder = self._get_latest_run_folder()
        if not run_folder:
            QMessageBox.information(
                self,
                "No Analysis Files",
                "No training run found. Train a model first to generate misclassified events."
            )
            return

        analysis_folder = os.path.join(run_folder, "analysis")
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

                # Confidence (prediction probability)
                confidence = row.get('prediction_confidence', '')
                if confidence != '' and not pd.isna(confidence):
                    self.misclass_table.setItem(i, 3, QTableWidgetItem(f"{float(confidence):.3f}"))
                else:
                    self.misclass_table.setItem(i, 3, QTableWidgetItem("N/A"))

                # File/rat info
                file_info = row.get('rat_id', row.get('file', row.get('predicted_label', '')))
                self.misclass_table.setItem(i, 4, QTableWidgetItem(str(file_info)))

                # Notes (ground truth vs predicted)
                gt = row.get('ground_truth', '')
                pred = row.get('predicted_label', '')
                notes = f"GT: {gt}, Pred: {pred}"
                self.misclass_table.setItem(i, 5, QTableWidgetItem(notes))

            # Resize columns
            self.misclass_table.resizeColumnsToContents()

            self.show_training_progress(f"Loaded {len(df)} misclassified events from {os.path.basename(latest_file)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load misclassified events:\n{str(e)}")
            self.show_training_progress(f"❌ Error loading misclassified events: {str(e)}")

    def view_confusion_matrix(self):
        """View confusion matrix plot."""
        self._view_plot("confusion_matrix_*.png", "Confusion Matrix")

    def view_feature_importance(self):
        """View feature importance plot."""
        self._view_plot("feature_importance_*.png", "Feature Importance")

    def view_threshold_curve(self):
        """View threshold curve plot."""
        self._view_plot("threshold_curve_*.png", "Threshold Curve")

    def view_shap_analysis(self):
        """View SHAP summary plot."""
        self._view_plot("shap_summary_*.png", "SHAP Analysis")

    def _view_plot(self, pattern, title):
        """Generic plot viewer — finds the most recent file matching pattern and shows it in a scrollable dialog."""
        run_folder = self._get_latest_run_folder()
        if not run_folder:
            QMessageBox.information(self, "No Plots", "No training run found. Train a model first.")
            return

        plots_folder = os.path.join(run_folder, "plots")
        if not os.path.exists(plots_folder):
            QMessageBox.information(self, "No Plots", "No plots folder found. Train a model first.")
            return

        matching = glob.glob(os.path.join(plots_folder, pattern))
        if not matching:
            QMessageBox.information(self, f"No {title}", f"No {title.lower()} plot found.")
            return

        latest_plot = max(matching, key=os.path.getmtime)

        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{title} - {os.path.basename(latest_plot)}")
            dialog.setMinimumSize(800, 600)

            layout = QVBoxLayout(dialog)

            pixmap = QPixmap(latest_plot)
            if pixmap.isNull():
                QMessageBox.warning(self, "Error", f"Could not load image:\n{latest_plot}")
                return

            # Scale to fixed width, keep aspect ratio (may be very tall)
            scaled_pixmap = pixmap.scaledToWidth(780, Qt.SmoothTransformation)

            image_label = QLabel()
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)

            scroll_area = QScrollArea()
            scroll_area.setWidget(image_label)
            scroll_area.setWidgetResizable(False)
            layout.addWidget(scroll_area)

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
            QMessageBox.critical(self, "Error", f"Failed to display {title.lower()}:\n{str(e)}")

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
    from .theme import get_app_stylesheet
    app = QApplication(sys.argv)
    app.setApplicationName("HTR Analysis Tool v3")
    app.setStyleSheet(get_app_stylesheet())

    window = HTRAnalysisAppV3()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
