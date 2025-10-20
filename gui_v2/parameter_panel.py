"""
Parameter Panel - Real-time parameter editing for HTR detection.
Component-based implementation for the Tune Parameters tab.
"""
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QFileDialog, QMessageBox,
    QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

try:
    from core.config import ConfigManager, get_config_manager
except ImportError:
    ConfigManager = None
    get_config_manager = None


class ParameterPanel(QWidget):
    """Parameter loading and editing panel for real-time HTR detection tuning."""

    # Signals emitted when parameters change or reanalysis requested
    parameters_changed = Signal()
    reanalyze_view_requested = Signal()     # Reanalyze visible graph range
    reanalyze_full_requested = Signal()     # Reanalyze entire video
    
    def __init__(self, parent=None, project_manager=None):
        super().__init__(parent)
        self.config_manager = get_config_manager() if get_config_manager else None
        self.project_manager = project_manager
        self.parameter_widgets = {}  # Store references to all parameter widgets
        self.init_ui()
        self.load_current_parameters()
    
    def init_ui(self):
        """Initialize the parameter panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Scrollable parameter area (NOW INCLUDES BUTTONS!)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Parameter content widget
        param_content = QWidget()
        param_layout = QVBoxLayout(param_content)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(8)

        # ADD BUTTONS AT TOP OF SCROLLABLE AREA
        self.create_header_buttons(param_layout)  # Load/Save/Reset + Reanalyze buttons

        # Create parameter groups in 2-column layout
        # Note: Node Mapping removed - configured during H5 load via dialog
        self.create_general_settings_compact(param_layout)  # General Settings only
        self.create_detector_row_groups(param_layout)  # Ear + Head Detectors

        param_layout.addStretch()  # Push everything to top

        scroll.setWidget(param_content)
        layout.addWidget(scroll)

        # Status label (keep at bottom, outside scroll)
        self.status_label = QLabel("Ready to load parameters")
        self.status_label.setFont(QFont("Arial", 8))
        self.status_label.setStyleSheet("color: #666; padding: 4px;")
        layout.addWidget(self.status_label)
    
    def create_header_buttons(self, parent_layout):
        """Create compact header buttons (now inside scrollable area)."""
        # Compact Parameter Management Row
        param_button_layout = QHBoxLayout()

        # Load parameters button
        load_btn = QPushButton("📁 Load")
        load_btn.setFont(QFont("Arial", 8))
        load_btn.setToolTip("Load parameters from file")
        load_btn.setMaximumWidth(70)
        load_btn.clicked.connect(self.load_parameters_from_file)
        param_button_layout.addWidget(load_btn)

        # Save parameters button
        save_btn = QPushButton("💾 Save")
        save_btn.setFont(QFont("Arial", 8))
        save_btn.setToolTip("Save parameters to file")
        save_btn.setMaximumWidth(70)
        save_btn.clicked.connect(self.save_parameters_to_file)
        param_button_layout.addWidget(save_btn)

        # Reset to defaults button
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setFont(QFont("Arial", 8))
        reset_btn.setToolTip("Reset to default parameters")
        reset_btn.setMaximumWidth(70)
        reset_btn.clicked.connect(self.reset_to_defaults)
        param_button_layout.addWidget(reset_btn)

        param_button_layout.addStretch()
        parent_layout.addLayout(param_button_layout)

        # Analysis Control Section (compact)
        self.create_analysis_controls(parent_layout)

    def create_analysis_controls(self, parent_layout):
        """Create analysis control section with reanalysis buttons."""
        analysis_group = QGroupBox("Detection Analysis")
        analysis_group.setMaximumHeight(75)
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setContentsMargins(8, 8, 8, 8)
        analysis_layout.setSpacing(4)

        # Info label (more compact)
        info_label = QLabel("Adjust parameters, then reanalyze:")
        info_label.setFont(QFont("Arial", 8))
        info_label.setStyleSheet("color: #666;")
        analysis_layout.addWidget(info_label)

        # Buttons row
        button_row = QHBoxLayout()

        self.reanalyze_view_btn = QPushButton("🔍 Reanalyze Current View")
        self.reanalyze_view_btn.setFont(QFont("Arial", 8, QFont.Bold))
        self.reanalyze_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.reanalyze_view_btn.clicked.connect(self.reanalyze_current_view)
        self.reanalyze_view_btn.setEnabled(False)  # Disabled until H5 loaded
        button_row.addWidget(self.reanalyze_view_btn)

        self.reanalyze_full_btn = QPushButton("📊 Reanalyze Full Video")
        self.reanalyze_full_btn.setFont(QFont("Arial", 8))
        self.reanalyze_full_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.reanalyze_full_btn.clicked.connect(self.reanalyze_full_video)
        self.reanalyze_full_btn.setEnabled(False)  # Disabled until H5 loaded
        button_row.addWidget(self.reanalyze_full_btn)

        button_row.addStretch()

        analysis_layout.addLayout(button_row)
        parent_layout.addWidget(analysis_group)

    def reanalyze_current_view(self):
        """Emit signal to reanalyze the current visible graph range."""
        self.apply_parameters_to_config()  # Ensure config is up-to-date
        self.reanalyze_view_requested.emit()
        self.status_label.setText("Reanalyzing current view...")

    def reanalyze_full_video(self):
        """Emit signal to reanalyze the entire video."""
        self.apply_parameters_to_config()  # Ensure config is up-to-date
        self.reanalyze_full_requested.emit()
        self.status_label.setText("Reanalyzing full video...")

    def enable_analysis_buttons(self, enabled: bool):
        """Enable or disable analysis buttons (called when H5 is loaded)."""
        self.reanalyze_view_btn.setEnabled(enabled)
        self.reanalyze_full_btn.setEnabled(enabled)

    def create_general_settings_compact(self, parent_layout):
        """Create compact General Settings section (node mapping now in dialog)."""
        general_group = QGroupBox("General Settings")
        general_layout = QVBoxLayout(general_group)
        general_layout.setSpacing(6)

        self.add_compact_spinbox_parameter(general_layout, "default_fps", "Default FPS:", 1, 300, "Default frame rate")
        self.add_compact_double_parameter(general_layout, "iou_threshold", "IoU Threshold:", 0.0, 1.0, 0.01, "Intersection over Union threshold")

        parent_layout.addWidget(general_group)
    
    def create_detector_row_groups(self, parent_layout):
        """Create detector row with Ear Detector (left) and Head Detector (right)."""
        row_layout = QHBoxLayout()
        row_layout.setSpacing(10)
        
        # Left column - Ear Detector
        ear_group = QGroupBox("Ear Detector Parameters")
        ear_layout = QVBoxLayout(ear_group)
        ear_layout.setSpacing(6)
        
        # Thresholds
        self.add_compact_spinbox_parameter(ear_layout, "ear_peak_threshold", "Peak Thresh:", 1, 100, "Minimum peak height")
        self.add_compact_spinbox_parameter(ear_layout, "ear_valley_threshold", "Valley Thresh:", 1, 100, "Minimum valley depth")
        
        # Gaps and timing
        self.add_compact_spinbox_parameter(ear_layout, "ear_max_gap", "Max Gap:", 1, 20, "Maximum gap between peaks")
        self.add_compact_spinbox_parameter(ear_layout, "ear_quick_gap", "Quick Gap:", 1, 20, "Quick gap threshold")
        self.add_compact_spinbox_parameter(ear_layout, "ear_min_crisscrosses", "Min Crisscross:", 1, 20, "Minimum crisscross events")
        self.add_compact_spinbox_parameter(ear_layout, "ear_between_unit_gap", "Unit Gap:", 1, 50, "Gap between units")
        self.add_compact_spinbox_parameter(ear_layout, "ear_merge_gap", "Merge Gap:", 1, 30, "Gap for merging events")
        
        # Filters
        self.add_compact_checkbox_parameter(ear_layout, "ear_apply_median_score_filter", "Median Score Filter")
        self.add_compact_double_parameter(ear_layout, "ear_median_score_threshold", "Median Thresh:", 0.0, 1.0, 0.01)
        
        row_layout.addWidget(ear_group)
        
        # Right column - Head Detector
        head_group = QGroupBox("Head Detector Parameters")
        head_layout = QVBoxLayout(head_group)
        head_layout.setSpacing(6)
        
        # Interpolation
        self.add_compact_combo_parameter(head_layout, "head_interpolation_method", "Interpolation:", 
                                        ["linear", "cubic", "nearest"], "Method for data interpolation")
        
        # Detection parameters
        self.add_compact_spinbox_parameter(head_layout, "head_min_oscillations", "Min Oscillations:", 1, 20, "Minimum oscillations required")
        self.add_compact_spinbox_parameter(head_layout, "head_amplitude_threshold", "Amp Thresh:", 1, 100, "Minimum amplitude")
        self.add_compact_spinbox_parameter(head_layout, "head_amplitude_median", "Amp Median:", 1, 100, "Median amplitude threshold")
        self.add_compact_double_parameter(head_layout, "head_median_score_threshold", "Median Thresh:", 0.0, 1.0, 0.01)
        
        # Peak detection
        self.add_compact_spinbox_parameter(head_layout, "head_peak_prominence", "Peak Prominence:", 1, 20, "Required peak prominence")
        self.add_compact_spinbox_parameter(head_layout, "head_peak_distance", "Peak Distance:", 1, 20, "Minimum distance between peaks")
        
        # Smoothing
        self.add_compact_checkbox_parameter(head_layout, "head_use_smoothing", "Use Smoothing")
        self.add_compact_spinbox_parameter(head_layout, "head_smoothing_window", "Smooth Window:", 3, 21, "Smoothing window size (odd)")
        self.add_compact_spinbox_parameter(head_layout, "head_smoothing_polyorder", "Poly Order:", 1, 5, "Polynomial order for smoothing")
        
        # Cycle parameters
        self.add_compact_spinbox_parameter(head_layout, "head_min_cycle_duration", "Min Cycle:", 1, 20, "Minimum cycle duration")
        self.add_compact_spinbox_parameter(head_layout, "head_max_cycle_duration", "Max Cycle:", 1, 30, "Maximum cycle duration")
        self.add_compact_spinbox_parameter(head_layout, "head_max_cycle_gap", "Cycle Gap:", 1, 20, "Maximum gap between cycles")
        
        row_layout.addWidget(head_group)
        parent_layout.addLayout(row_layout)
    
    # Note: Old full-width group methods removed - now using compact 2-column layout
    
    def add_spinbox_parameter(self, layout, key, label, min_val, max_val, tooltip=""):
        """Add a spinbox parameter control."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(140)
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setFont(QFont("Arial", 8))
        spinbox.valueChanged.connect(self.on_parameter_changed)
        if tooltip:
            spinbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = spinbox
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(spinbox)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def add_double_parameter(self, layout, key, label, min_val, max_val, step, tooltip=""):
        """Add a double spinbox parameter control."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(140)
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setFont(QFont("Arial", 8))
        spinbox.valueChanged.connect(self.on_parameter_changed)
        if tooltip:
            spinbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = spinbox
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(spinbox)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def add_checkbox_parameter(self, layout, key, label, tooltip=""):
        """Add a checkbox parameter control."""
        checkbox = QCheckBox(label)
        checkbox.setFont(QFont("Arial", 8))
        checkbox.stateChanged.connect(self.on_parameter_changed)
        if tooltip:
            checkbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = checkbox
        layout.addWidget(checkbox)
    
    def add_combo_parameter(self, layout, key, label, options, tooltip=""):
        """Add a combobox parameter control."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(140)
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        combo = QComboBox()
        combo.addItems(options)
        combo.setFont(QFont("Arial", 8))
        combo.currentTextChanged.connect(self.on_parameter_changed)
        if tooltip:
            combo.setToolTip(tooltip)
        
        self.parameter_widgets[key] = combo
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(combo)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def add_compact_spinbox_parameter(self, layout, key, label, min_val, max_val, tooltip=""):
        """Add a compact spinbox parameter control for 2-column layout."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(80)  # Shorter for compact layout
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setFont(QFont("Arial", 8))
        spinbox.setMaximumWidth(80)  # Increased width for 3-digit numbers
        spinbox.valueChanged.connect(self.on_parameter_changed)
        if tooltip:
            spinbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = spinbox
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(spinbox)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def add_compact_double_parameter(self, layout, key, label, min_val, max_val, step, tooltip=""):
        """Add a compact double spinbox parameter control for 2-column layout."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(80)  # Shorter for compact layout
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setFont(QFont("Arial", 8))
        spinbox.setMaximumWidth(90)  # Increased width for decimal values
        spinbox.valueChanged.connect(self.on_parameter_changed)
        if tooltip:
            spinbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = spinbox
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(spinbox)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def add_compact_checkbox_parameter(self, layout, key, label, tooltip=""):
        """Add a compact checkbox parameter control for 2-column layout."""
        checkbox = QCheckBox(label)
        checkbox.setFont(QFont("Arial", 8))
        checkbox.stateChanged.connect(self.on_parameter_changed)
        if tooltip:
            checkbox.setToolTip(tooltip)
        
        self.parameter_widgets[key] = checkbox
        layout.addWidget(checkbox)
    
    def add_compact_combo_parameter(self, layout, key, label, options, tooltip=""):
        """Add a compact combobox parameter control for 2-column layout."""
        row_layout = QHBoxLayout()
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(80)  # Shorter for compact layout
        label_widget.setFont(QFont("Arial", 8))
        if tooltip:
            label_widget.setToolTip(tooltip)
        
        combo = QComboBox()
        combo.addItems(options)
        combo.setFont(QFont("Arial", 8))
        combo.setMaximumWidth(80)  # Compact width
        combo.currentTextChanged.connect(self.on_parameter_changed)
        if tooltip:
            combo.setToolTip(tooltip)
        
        self.parameter_widgets[key] = combo
        
        row_layout.addWidget(label_widget)
        row_layout.addWidget(combo)
        row_layout.addStretch()
        
        layout.addLayout(row_layout)
    
    def load_current_parameters(self):
        """Load current parameters from config manager."""
        if not self.config_manager:
            self.status_label.setText("Config manager not available")
            return
        
        # Temporarily disconnect signals to prevent individual parameter updates
        self.disconnect_parameter_signals()
        
        try:
            config = self.config_manager.config
            
            # Load ear detector parameters
            ear = config.ear_detector
            self.set_parameter_value("ear_peak_threshold", ear.peak_threshold)
            self.set_parameter_value("ear_valley_threshold", ear.valley_threshold)
            self.set_parameter_value("ear_max_gap", ear.max_gap)
            self.set_parameter_value("ear_quick_gap", ear.quick_gap)
            self.set_parameter_value("ear_min_crisscrosses", ear.min_crisscrosses)
            self.set_parameter_value("ear_between_unit_gap", ear.between_unit_gap)
            self.set_parameter_value("ear_merge_gap", ear.merge_gap)
            self.set_parameter_value("ear_apply_median_score_filter", ear.apply_median_score_filter)
            self.set_parameter_value("ear_median_score_threshold", ear.median_score_threshold)
            
            # Load head detector parameters
            head = config.head_detector
            self.set_parameter_value("head_interpolation_method", head.interpolation_method)
            self.set_parameter_value("head_min_oscillations", head.min_oscillations)
            self.set_parameter_value("head_amplitude_threshold", head.amplitude_threshold)
            self.set_parameter_value("head_amplitude_median", head.amplitude_median)
            self.set_parameter_value("head_median_score_threshold", head.median_score_threshold)
            self.set_parameter_value("head_peak_prominence", head.peak_prominence)
            self.set_parameter_value("head_peak_distance", head.peak_distance)
            self.set_parameter_value("head_use_smoothing", head.use_smoothing)
            self.set_parameter_value("head_smoothing_window", head.smoothing_window)
            self.set_parameter_value("head_smoothing_polyorder", head.smoothing_polyorder)
            self.set_parameter_value("head_min_cycle_duration", head.min_cycle_duration)
            self.set_parameter_value("head_max_cycle_duration", head.max_cycle_duration)
            self.set_parameter_value("head_max_cycle_gap", head.max_cycle_gap)
            
            # Note: Node mapping removed - configured during H5 load via dialog

            # Load general settings
            self.set_parameter_value("default_fps", config.default_fps)
            self.set_parameter_value("iou_threshold", config.iou_threshold)
            
            self.status_label.setText("Current parameters loaded")
            
        except Exception as e:
            self.status_label.setText(f"Error loading parameters: {str(e)}")
        finally:
            # Reconnect signals after all parameters are set
            self.connect_parameter_signals()
    
    def set_parameter_value(self, key, value):
        """Set parameter widget value safely."""
        if key not in self.parameter_widgets:
            return
        
        widget = self.parameter_widgets[key]
        
        try:
            if isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
        except (ValueError, TypeError):
            pass  # Ignore invalid values
    
    def get_parameter_value(self, key):
        """Get parameter widget value safely."""
        if key not in self.parameter_widgets:
            return None
        
        widget = self.parameter_widgets[key]
        
        if isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        
        return None
    
    def disconnect_parameter_signals(self):
        """Temporarily disconnect all parameter change signals."""
        for widget in self.parameter_widgets.values():
            if isinstance(widget, QSpinBox):
                widget.valueChanged.disconnect()
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.disconnect()
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.disconnect()
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.disconnect()
    
    def connect_parameter_signals(self):
        """Reconnect all parameter change signals."""
        for widget in self.parameter_widgets.values():
            if isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.on_parameter_changed)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self.on_parameter_changed)
    
    def on_parameter_changed(self):
        """Handle parameter value changes."""
        if self.config_manager:
            self.apply_parameters_to_config()
        self.parameters_changed.emit()
        self.status_label.setText("Parameters modified")
    
    def apply_parameters_to_config(self):
        """Apply current parameter values to config manager."""
        if not self.config_manager:
            return
        
        try:
            config = self.config_manager.config
            
            # Apply ear detector parameters
            config.ear_detector.peak_threshold = self.get_parameter_value("ear_peak_threshold")
            config.ear_detector.valley_threshold = self.get_parameter_value("ear_valley_threshold")
            config.ear_detector.max_gap = self.get_parameter_value("ear_max_gap")
            config.ear_detector.quick_gap = self.get_parameter_value("ear_quick_gap")
            config.ear_detector.min_crisscrosses = self.get_parameter_value("ear_min_crisscrosses")
            config.ear_detector.between_unit_gap = self.get_parameter_value("ear_between_unit_gap")
            config.ear_detector.merge_gap = self.get_parameter_value("ear_merge_gap")
            config.ear_detector.apply_median_score_filter = self.get_parameter_value("ear_apply_median_score_filter")
            config.ear_detector.median_score_threshold = self.get_parameter_value("ear_median_score_threshold")
            
            # Apply head detector parameters
            config.head_detector.interpolation_method = self.get_parameter_value("head_interpolation_method")
            config.head_detector.min_oscillations = self.get_parameter_value("head_min_oscillations")
            config.head_detector.amplitude_threshold = self.get_parameter_value("head_amplitude_threshold")
            config.head_detector.amplitude_median = self.get_parameter_value("head_amplitude_median")
            config.head_detector.median_score_threshold = self.get_parameter_value("head_median_score_threshold")
            config.head_detector.peak_prominence = self.get_parameter_value("head_peak_prominence")
            config.head_detector.peak_distance = self.get_parameter_value("head_peak_distance")
            config.head_detector.use_smoothing = self.get_parameter_value("head_use_smoothing")
            config.head_detector.smoothing_window = self.get_parameter_value("head_smoothing_window")
            config.head_detector.smoothing_polyorder = self.get_parameter_value("head_smoothing_polyorder")
            config.head_detector.min_cycle_duration = self.get_parameter_value("head_min_cycle_duration")
            config.head_detector.max_cycle_duration = self.get_parameter_value("head_max_cycle_duration")
            config.head_detector.max_cycle_gap = self.get_parameter_value("head_max_cycle_gap")
            
            # Note: Node mapping removed - configured during H5 load via dialog

            # Apply general settings
            config.default_fps = self.get_parameter_value("default_fps")
            config.iou_threshold = self.get_parameter_value("iou_threshold")
            
        except Exception as e:
            self.status_label.setText(f"Error applying parameters: {str(e)}")
    
    def update_project_status(self):
        """Update the project status display (no-op after UI simplification)."""
        pass  # Project status removed from UI for cleaner layout
    
    def get_project_parameters_folder(self):
        """Get the parameters folder for the current project."""
        if not self.project_manager:
            return None
        
        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return None
        
        parameters_folder = os.path.join(project_path, "parameters")
        os.makedirs(parameters_folder, exist_ok=True)  # Ensure it exists
        return parameters_folder
    
    def load_parameters_from_file(self):
        """Load parameters from a JSON file (project-aware)."""
        # Determine starting directory
        start_dir = self.get_project_parameters_folder()
        if not start_dir:
            start_dir = ""
            title = "Load Parameters (No Project Loaded)"
        else:
            title = "Load Parameters from Project"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, start_dir, "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path and self.config_manager:
            if self.config_manager.import_parameters(file_path):
                self.load_current_parameters()
                rel_path = os.path.relpath(file_path, start_dir) if start_dir else os.path.basename(file_path)
                self.status_label.setText(f"Loaded parameters from: {rel_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load parameter file.")
    
    def save_parameters_to_file(self):
        """Save current parameters to a JSON file (project-aware)."""
        # Determine starting directory and default filename
        start_dir = self.get_project_parameters_folder()
        if not start_dir:
            start_dir = ""
            default_filename = "htr_parameters.json"
            title = "Save Parameters (No Project Loaded)"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"htr_parameters_{timestamp}.json"
            title = "Save Parameters to Project"
        
        default_path = os.path.join(start_dir, default_filename) if start_dir else default_filename
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, title, default_path, "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path and self.config_manager:
            self.apply_parameters_to_config()  # Ensure config is up to date
            if self.config_manager.export_parameters(file_path):
                rel_path = os.path.relpath(file_path, start_dir) if start_dir else os.path.basename(file_path)
                self.status_label.setText(f"Saved parameters to: {rel_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save parameter file.")
    
    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        if self.config_manager:
            # Create new config with defaults
            from core.config import ConfigManager
            default_config = ConfigManager()
            self.config_manager.config = default_config.config
            self.load_current_parameters()
            self.status_label.setText("Reset to default parameters")
    
    def refresh_project_status(self):
        """Refresh the project status display (call when project changes)."""
        self.update_project_status()
    
    def manage_project(self):
        """Open project management dialog (removed from UI, kept for compatibility)."""
        pass  # Method kept for backward compatibility but no longer used in UI