"""
Batch Analysis Tab v2 - Compact layout for batch processing workflow.
Built component by component with size constraints for 1400×700 layout.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QLineEdit, QFileDialog, QTextEdit, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# For backend integration (when ready)
try:
    from core.config import ConfigManager
except ImportError:
    ConfigManager = None


class BatchAnalysisTabV2(QWidget):
    """Batch analysis tab with compact layout for 1400×700 window."""
    
    def __init__(self, config_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI component by component."""
        # Main layout with tight spacing
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)
        
        # Component 1: Instructions header
        self.create_instructions_component(layout)
        
        # Component 2: Configuration section  
        self.create_configuration_component(layout)
        
        # Temporary spacer to see how components look
        layout.addStretch()
    
    def create_instructions_component(self, parent_layout):
        """Component 1: Instructions header with workflow overview."""
        # Instructions section - compact and clear
        instructions_group = QGroupBox("Batch Analysis Workflow")
        instructions_group.setMaximumHeight(100)  # Keep it compact
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_layout.setContentsMargins(10, 10, 10, 10)
        instructions_layout.setSpacing(5)
        
        # Main instruction text - more concise than original
        instructions = QLabel(
            "<b>Process multiple videos automatically:</b> "
            "1) Select parameters and model files → "
            "2) Choose input/output folders → "
            "3) Run 3-step pipeline or full automation"
        )
        instructions.setFont(QFont("Arial", 10))
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #495057;")
        instructions_layout.addWidget(instructions)
        
        parent_layout.addWidget(instructions_group)
    
    def create_configuration_component(self, parent_layout):
        """Component 2: Configuration section for parameters and model selection."""
        config_group = QGroupBox("Configuration")
        config_group.setMaximumHeight(120)  # Keep compact - 2 rows
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(8)
        
        # Parameter file row
        param_layout = QHBoxLayout()
        param_layout.setSpacing(10)
        
        param_label = QLabel("Parameters:")
        param_label.setMinimumWidth(80)
        param_label.setFont(QFont("Arial", 9))
        param_layout.addWidget(param_label)
        
        self.param_path_edit = QLineEdit()
        self.param_path_edit.setPlaceholderText("Optional: Load saved parameter configuration")
        self.param_path_edit.setFont(QFont("Arial", 9))
        param_layout.addWidget(self.param_path_edit)
        
        param_browse_btn = QPushButton("Browse...")
        param_browse_btn.setMaximumWidth(80)
        param_browse_btn.setFont(QFont("Arial", 9))
        param_browse_btn.clicked.connect(self.browse_param_file)
        param_layout.addWidget(param_browse_btn)
        
        config_layout.addLayout(param_layout)
        
        # Model file row
        model_layout = QHBoxLayout()
        model_layout.setSpacing(10)
        
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
        model_browse_btn.setFont(QFont("Arial", 9))
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(model_browse_btn)
        
        config_layout.addLayout(model_layout)
        
        parent_layout.addWidget(config_group)
    
    def browse_param_file(self):
        """Browse for parameter configuration file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Parameter File", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.param_path_edit.setText(file_path)
    
    def browse_model_file(self):
        """Browse for trained model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.joblib);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)


def main():
    """Test the batch analysis component."""
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test widget
    widget = BatchAnalysisTabV2()
    widget.resize(1400, 700)
    widget.setWindowTitle("Batch Analysis Tab v2 - Component Test")
    widget.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()