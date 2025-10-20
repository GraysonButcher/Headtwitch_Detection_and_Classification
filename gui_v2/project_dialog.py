"""
Project Selection Dialog for HTR Analysis Tool v2
Allows users to create new projects or open existing ones.
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QGroupBox, QMessageBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import sys
import os
sys.path.append(os.path.dirname(__file__))
from project_manager import ProjectManager


class ProjectDialog(QDialog):
    """Dialog for creating or opening HTR analysis projects."""

    def __init__(self, parent=None, workflow_type="batch", mode="both"):
        super().__init__(parent)
        self.workflow_type = workflow_type
        self.mode = mode  # "create", "open", or "both"
        # Use the parent's project manager instead of creating a new one
        if parent and hasattr(parent, 'project_manager'):
            self.project_manager = parent.project_manager
        else:
            self.project_manager = ProjectManager()  # Fallback
        self.selected_project_path = None

        # Set window title based on mode
        if mode == "create":
            self.setWindowTitle("Create New HTR Project")
        elif mode == "open":
            self.setWindowTitle("Open HTR Project")
        else:
            self.setWindowTitle(f"HTR Project - {workflow_type.title()} Workflow")

        self.setFixedSize(600, 400)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title (only show if mode is "both")
        if self.mode == "both":
            title_label = QLabel(f"HTR Analysis Project - {self.workflow_type.title()} Workflow")
            title_label.setFont(QFont("Arial", 14, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)

        # Description
        if self.mode == "create":
            desc_text = "Create a new HTR analysis project with organized folder structure."
        elif self.mode == "open":
            desc_text = "Open an existing HTR analysis project to continue your work."
        else:
            desc_text = {
                "batch": "Create or open a project to organize batch processing of multiple videos.",
                "train": "Create or open a project to manage model training data and results.",
                "tune": "Create or open a project to save optimized parameters for future use."
            }
            desc_text = desc_text.get(self.workflow_type, "Create or open an HTR analysis project.")

        desc_label = QLabel(desc_text)
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #6c757d; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Create new project section (only if mode is "create" or "both")
        if self.mode in ["create", "both"]:
            create_group = QGroupBox("Create New Project")
            create_layout = QVBoxLayout(create_group)
            create_layout.setSpacing(10)

            # Project name
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Project Name:"))
            self.project_name_edit = QLineEdit()
            self.project_name_edit.setPlaceholderText("Enter project name...")
            name_layout.addWidget(self.project_name_edit)
            create_layout.addLayout(name_layout)

            # Project location
            location_layout = QHBoxLayout()
            location_layout.addWidget(QLabel("Location:"))
            self.project_location_edit = QLineEdit()
            self.project_location_edit.setPlaceholderText("Select folder for project...")
            location_layout.addWidget(self.project_location_edit)

            browse_location_btn = QPushButton("Browse...")
            browse_location_btn.setMaximumWidth(80)
            browse_location_btn.clicked.connect(self.browse_project_location)
            location_layout.addWidget(browse_location_btn)
            create_layout.addLayout(location_layout)

            # Create button
            create_btn = QPushButton("Create Project")
            create_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            create_btn.clicked.connect(self.create_project)
            create_layout.addWidget(create_btn)

            layout.addWidget(create_group)

        # Open existing project section (only if mode is "open" or "both")
        if self.mode in ["open", "both"]:
            open_group = QGroupBox("Open Existing Project")
            open_layout = QVBoxLayout(open_group)
            open_layout.setSpacing(10)

            # Project folder
            open_layout.addWidget(QLabel("Select existing HTR project folder:"))
            folder_layout = QHBoxLayout()
            self.existing_project_edit = QLineEdit()
            self.existing_project_edit.setPlaceholderText("Browse for project folder...")
            folder_layout.addWidget(self.existing_project_edit)

            browse_existing_btn = QPushButton("Browse...")
            browse_existing_btn.setMaximumWidth(80)
            browse_existing_btn.clicked.connect(self.browse_existing_project)
            folder_layout.addWidget(browse_existing_btn)
            open_layout.addLayout(folder_layout)

            # Open button
            open_btn = QPushButton("Open Project")
            open_btn.setStyleSheet("""
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    padding: 10px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
            """)
            open_btn.clicked.connect(self.open_project)
            open_layout.addWidget(open_btn)

            layout.addWidget(open_group)

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
    def browse_project_location(self):
        """Browse for project location."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Project Location", ""
        )
        if folder_path:
            self.project_location_edit.setText(folder_path)
            
    def browse_existing_project(self):
        """Browse for existing project."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select HTR Project Folder", ""
        )
        if folder_path:
            self.existing_project_edit.setText(folder_path)
            
    def create_project(self):
        """Create a new project."""
        project_name = self.project_name_edit.text().strip()
        project_location = self.project_location_edit.text().strip()
        
        if not project_name:
            QMessageBox.warning(self, "Error", "Please enter a project name.")
            return
            
        if not project_location:
            QMessageBox.warning(self, "Error", "Please select a project location.")
            return
            
        # Check if location exists
        if not os.path.exists(project_location):
            QMessageBox.warning(self, "Error", "Project location does not exist.")
            return
        
        # Create project
        success, result = self.project_manager.create_project(
            project_location, project_name, self.workflow_type
        )
        
        if success:
            self.selected_project_path = result
            QMessageBox.information(self, "Success", f"Project '{project_name}' created successfully!")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", result)
            
    def open_project(self):
        """Open an existing project."""
        project_folder = self.existing_project_edit.text().strip()
        
        if not project_folder:
            QMessageBox.warning(self, "Error", "Please select a project folder.")
            return
            
        # Load project
        success, result = self.project_manager.load_project(project_folder)
        
        if success:
            self.selected_project_path = result
            QMessageBox.information(self, "Success", "Project loaded successfully!")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", result)
            
    def get_selected_project(self):
        """Get the selected project path and manager."""
        return self.selected_project_path, self.project_manager