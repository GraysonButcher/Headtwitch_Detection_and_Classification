"""
HTR Analysis Tool - Main Application
A comprehensive tool for Head-Twitch Response analysis from SLEAP tracking data.
"""
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMenuBar, QMenu, QMessageBox
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import Qt

# Import our core modules
from core.config import get_config_manager
from gui.parameter_inspector import ParameterInspectorTab
from gui.batch_analysis import BatchAnalysisTab
from gui.model_training import ModelTrainingTab


class HTRAnalysisApp(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        self.config_manager = get_config_manager()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("HTR Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and tab widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.parameter_tab = ParameterInspectorTab(self.config_manager)
        self.batch_tab = BatchAnalysisTab(self.config_manager)
        self.training_tab = ModelTrainingTab(self.config_manager)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.parameter_tab, "Parameter Inspector")
        self.tab_widget.addTab(self.batch_tab, "Batch Analysis")
        self.tab_widget.addTab(self.training_tab, "Model Training")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Set window icon if available
        # self.setWindowIcon(QIcon("icons/htr_icon.png"))
        
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Import/Export Parameters
        import_params_action = QAction("Import Parameters...", self)
        import_params_action.triggered.connect(self.import_parameters)
        file_menu.addAction(import_params_action)
        
        export_params_action = QAction("Export Parameters...", self)
        export_params_action.triggered.connect(self.export_parameters)
        file_menu.addAction(export_params_action)
        
        file_menu.addSeparator()
        
        # Recent files
        recent_menu = file_menu.addMenu("Recent Files")
        self.update_recent_files_menu(recent_menu)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        node_mapping_action = QAction("Configure Node Mapping...", self)
        node_mapping_action.triggered.connect(self.configure_node_mapping)
        settings_menu.addAction(node_mapping_action)
        
        preferences_action = QAction("Preferences...", self)
        preferences_action.triggered.connect(self.show_preferences)
        settings_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_action = QAction("User Guide", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
    
    def update_recent_files_menu(self, recent_menu):
        """Update the recent files menu."""
        recent_menu.clear()
        
        for file_path in self.config_manager.config.recent_files:
            if os.path.exists(file_path):
                action = QAction(os.path.basename(file_path), self)
                action.setToolTip(file_path)
                action.triggered.connect(lambda checked, path=file_path: self.open_recent_file(path))
                recent_menu.addAction(action)
    
    def import_parameters(self):
        """Import detection parameters from file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Parameters",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.config_manager.import_parameters(file_path):
                QMessageBox.information(self, "Success", "Parameters imported successfully!")
                # Notify tabs to update their UI
                self.parameter_tab.refresh_parameters()
            else:
                QMessageBox.warning(self, "Error", "Failed to import parameters.")
    
    def export_parameters(self):
        """Export detection parameters to file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Parameters",
            "htr_parameters.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.config_manager.export_parameters(file_path):
                QMessageBox.information(self, "Success", "Parameters exported successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to export parameters.")
    
    def configure_node_mapping(self):
        """Configure SLEAP node mapping."""
        from gui.dialogs.node_mapping_dialog import NodeMappingDialog
        
        dialog = NodeMappingDialog(self.config_manager, self)
        if dialog.exec_() == dialog.Accepted:
            # Node mapping was updated, save config
            self.config_manager.save_config()
            QMessageBox.information(self, "Success", "Node mapping updated successfully!")
    
    def show_preferences(self):
        """Show application preferences dialog."""
        from gui.dialogs.preferences_dialog import PreferencesDialog
        
        dialog = PreferencesDialog(self.config_manager, self)
        if dialog.exec_() == dialog.Accepted:
            self.config_manager.save_config()
    
    def open_recent_file(self, file_path):
        """Open a recent file in the parameter inspector."""
        self.tab_widget.setCurrentIndex(0)  # Switch to parameter inspector tab
        self.parameter_tab.load_h5_file(file_path)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h3>HTR Analysis Tool</h3>
        <p>Version 1.0</p>
        <p>A comprehensive tool for analyzing Head-Twitch Responses (HTRs) from SLEAP tracking data.</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Interactive parameter tuning</li>
        <li>Batch processing and analysis</li>
        <li>Machine learning model training</li>
        <li>Comprehensive reporting</li>
        </ul>
        <p><b>Developed for behavioral neuroscience research.</b></p>
        """
        
        QMessageBox.about(self, "About HTR Analysis Tool", about_text)
    
    def show_help(self):
        """Show help documentation."""
        QMessageBox.information(
            self, 
            "User Guide", 
            "User documentation can be found in the docs/ folder or online at:\n"
            "https://github.com/your-repo/htr-analysis-tool"
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Save configuration before closing
        self.config_manager.save_config()
        
        # Check if any tabs have unsaved work
        if self.parameter_tab.has_unsaved_changes():
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes",
                "You have unsaved parameter changes. Do you want to save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                if not self.parameter_tab.save_current_parameters():
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        event.accept()


def main():
    """Main entry point for the application."""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("HTR Analysis Tool")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("HTR Analysis")
    
    # Set application style (optional)
    # app.setStyle('Fusion')
    
    # Create and show main window
    window = HTRAnalysisApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()