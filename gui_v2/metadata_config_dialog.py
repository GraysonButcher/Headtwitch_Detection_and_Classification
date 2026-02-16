"""
Metadata Configuration Dialog - Configure how metadata is extracted from file paths.

Allows users to map folder levels to metadata fields (cohort, drug, dose) and
define templates for parsing combined fields.
"""

import os
import re
import glob
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QLineEdit, QMessageBox,
    QFrame, QScrollArea, QWidget, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extraction import MetadataConfig, MetadataExtractor


class MetadataConfigDialog(QDialog):
    """
    Dialog for configuring metadata extraction from file paths.

    Shows sample file paths from the input folder and allows users to:
    - Assign folder levels to metadata fields (cohort, drug_dose, etc.)
    - Define templates for parsing combined fields (e.g., "{drug}_{dose}")
    - Preview extracted metadata and output filenames
    """

    def __init__(self, input_folder: str, parent=None):
        """
        Initialize metadata configuration dialog.

        Args:
            input_folder: Path to the folder containing H5 files
            parent: Parent widget
        """
        super().__init__(parent)
        self.input_folder = input_folder
        self.sample_files = self._find_sample_files()
        self.current_sample_index = 0
        self.result_config: Optional[MetadataConfig] = None

        # UI state
        self.folder_combos: Dict[int, QComboBox] = {}
        self.template_edits: Dict[str, QLineEdit] = {}

        self.setWindowTitle("Configure Metadata Extraction")
        self.setModal(True)
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self.init_ui()
        self.update_preview()

    def _find_sample_files(self) -> List[str]:
        """Find sample H5 files in the input folder."""
        pattern = os.path.join(self.input_folder, "**", "*.h5")
        files = glob.glob(pattern, recursive=True)
        # Sort by path depth (more nested files first) to get representative examples
        files.sort(key=lambda x: x.count(os.sep), reverse=True)
        return files[:10]  # Limit to 10 samples

    def _get_folder_levels(self, file_path: str) -> List[Tuple[int, str]]:
        """Get folder levels relative to the file.

        Returns list of (level, folder_name) tuples.
        Level -1 is the immediate parent folder, -2 is grandparent, etc.
        """
        file_path = os.path.normpath(file_path)
        parts = file_path.split(os.sep)

        # Remove filename
        folders = parts[:-1]

        # Get relative levels (negative, from file)
        levels = []
        for i, folder in enumerate(reversed(folders)):
            level = -(i + 1)
            levels.append((level, folder))

        return levels[:5]  # Limit to 5 levels up

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Configure Metadata Extraction")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Configure how metadata (cohort, drug, dose, rat_id) is extracted from your file paths.\n"
            "This metadata will be added as columns in the output CSV files."
        )
        instructions.setFont(QFont("Arial", 9))
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 5px 0 10px 0;")
        layout.addWidget(instructions)

        # Sample file display
        sample_group = QGroupBox("Sample File Path")
        sample_layout = QVBoxLayout(sample_group)

        # Sample navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self._prev_sample)
        nav_layout.addWidget(self.prev_btn)

        self.sample_label = QLabel()
        self.sample_label.setFont(QFont("Consolas", 9))
        self.sample_label.setWordWrap(True)
        self.sample_label.setStyleSheet("background: #f5f5f5; padding: 8px; border-radius: 4px;")
        nav_layout.addWidget(self.sample_label, 1)

        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self._next_sample)
        nav_layout.addWidget(self.next_btn)

        sample_layout.addLayout(nav_layout)

        self.sample_count_label = QLabel()
        self.sample_count_label.setFont(QFont("Arial", 8))
        self.sample_count_label.setStyleSheet("color: #888;")
        self.sample_count_label.setAlignment(Qt.AlignCenter)
        sample_layout.addWidget(self.sample_count_label)

        layout.addWidget(sample_group)

        # Folder level mapping
        folder_group = QGroupBox("Step 1: Assign Folder Levels to Metadata Fields")
        folder_layout = QVBoxLayout(folder_group)

        folder_instructions = QLabel(
            "Select which metadata field each folder level represents. "
            "Use 'Skip' for folders that don't contain metadata."
        )
        folder_instructions.setFont(QFont("Arial", 9))
        folder_instructions.setStyleSheet("color: #666;")
        folder_instructions.setWordWrap(True)
        folder_layout.addWidget(folder_instructions)

        # Folder mapping form
        self.folder_form = QFormLayout()
        self.folder_form.setSpacing(8)
        folder_layout.addLayout(self.folder_form)

        layout.addWidget(folder_group)

        # Template parsing
        template_group = QGroupBox("Step 2: Parse Combined Fields (Optional)")
        template_layout = QVBoxLayout(template_group)

        template_instructions = QLabel(
            "If a folder combines multiple fields (e.g., 'Psi_0.1mgkg' = drug + dose), "
            "enter a template to parse it. Use {field_name} placeholders."
        )
        template_instructions.setFont(QFont("Arial", 9))
        template_instructions.setStyleSheet("color: #666;")
        template_instructions.setWordWrap(True)
        template_layout.addWidget(template_instructions)

        # Template form
        self.template_form = QFormLayout()
        self.template_form.setSpacing(8)
        template_layout.addLayout(self.template_form)

        # Match mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("When template is ambiguous:")
        mode_label.setFont(QFont("Arial", 9))
        mode_layout.addWidget(mode_label)

        self.match_mode_group = QButtonGroup(self)
        self.first_radio = QRadioButton("Match first delimiter")
        self.first_radio.setChecked(True)
        self.first_radio.toggled.connect(self.update_preview)
        self.match_mode_group.addButton(self.first_radio)
        mode_layout.addWidget(self.first_radio)

        self.last_radio = QRadioButton("Match last delimiter")
        self.last_radio.toggled.connect(self.update_preview)
        self.match_mode_group.addButton(self.last_radio)
        mode_layout.addWidget(self.last_radio)

        mode_layout.addStretch()
        template_layout.addLayout(mode_layout)

        layout.addWidget(template_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel()
        self.preview_label.setFont(QFont("Consolas", 9))
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet(
            "background: #e8f5e9; padding: 10px; border-radius: 4px; border: 1px solid #c8e6c9;"
        )
        preview_layout.addWidget(self.preview_label)

        layout.addWidget(preview_group)

        # Buttons
        button_layout = QHBoxLayout()

        skip_btn = QPushButton("Skip (No Metadata)")
        skip_btn.setFont(QFont("Arial", 9))
        skip_btn.setToolTip("Extract features without metadata columns")
        skip_btn.clicked.connect(self._skip_metadata)
        button_layout.addWidget(skip_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Arial", 9))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        confirm_btn = QPushButton("Apply Configuration")
        confirm_btn.setFont(QFont("Arial", 9, QFont.Bold))
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        confirm_btn.clicked.connect(self._confirm)
        button_layout.addWidget(confirm_btn)

        layout.addLayout(button_layout)

        # Populate UI with sample data
        self._update_sample_display()
        self._populate_folder_combos()

    def _update_sample_display(self):
        """Update the sample file display."""
        if not self.sample_files:
            self.sample_label.setText("No H5 files found in input folder")
            self.sample_count_label.setText("")
            return

        file_path = self.sample_files[self.current_sample_index]
        # Show path relative to input folder for clarity
        rel_path = os.path.relpath(file_path, self.input_folder)
        self.sample_label.setText(rel_path)
        self.sample_count_label.setText(
            f"Sample {self.current_sample_index + 1} of {len(self.sample_files)}"
        )

        self.prev_btn.setEnabled(self.current_sample_index > 0)
        self.next_btn.setEnabled(self.current_sample_index < len(self.sample_files) - 1)

    def _prev_sample(self):
        """Show previous sample file."""
        if self.current_sample_index > 0:
            self.current_sample_index -= 1
            self._update_sample_display()
            self.update_preview()

    def _next_sample(self):
        """Show next sample file."""
        if self.current_sample_index < len(self.sample_files) - 1:
            self.current_sample_index += 1
            self._update_sample_display()
            self.update_preview()

    def _populate_folder_combos(self):
        """Populate folder level combo boxes based on sample file structure."""
        # Clear existing
        while self.folder_form.count():
            item = self.folder_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.folder_combos.clear()

        # Clear template form too
        while self.template_form.count():
            item = self.template_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.template_edits.clear()

        if not self.sample_files:
            return

        file_path = self.sample_files[self.current_sample_index]
        levels = self._get_folder_levels(file_path)

        # Metadata field options
        field_options = [
            ("skip", "-- Skip --"),
            ("cohort", "Cohort"),
            ("drug", "Drug"),
            ("dose", "Dose"),
            ("drug_dose", "Drug + Dose (combined)"),
            ("condition", "Condition (custom)"),
        ]

        for level, folder_name in levels:
            # Create row with folder name and combo
            row_layout = QHBoxLayout()

            folder_label = QLabel(f'"{folder_name}"')
            folder_label.setFont(QFont("Consolas", 9))
            folder_label.setFixedWidth(200)
            folder_label.setStyleSheet("background: #f0f0f0; padding: 4px; border-radius: 3px;")
            row_layout.addWidget(folder_label)

            arrow_label = QLabel("→")
            arrow_label.setFixedWidth(20)
            row_layout.addWidget(arrow_label)

            combo = QComboBox()
            combo.setFont(QFont("Arial", 9))
            for value, display in field_options:
                combo.addItem(display, value)

            # Auto-detect based on folder name
            folder_lower = folder_name.lower()
            if 'cohort' in folder_lower:
                combo.setCurrentIndex(1)  # cohort
            elif any(x in folder_lower for x in ['drug', 'psi', 'saline', 'vehicle']):
                if any(x in folder_lower for x in ['mg', 'dose', '0.']):
                    combo.setCurrentIndex(4)  # drug_dose combined
                else:
                    combo.setCurrentIndex(2)  # drug
            elif any(x in folder_lower for x in ['mg', 'dose', '0.']):
                combo.setCurrentIndex(3)  # dose

            combo.currentIndexChanged.connect(self._on_folder_combo_changed)
            self.folder_combos[level] = combo
            row_layout.addWidget(combo, 1)

            # Add to form
            level_label = QLabel(f"Level {level}:")
            level_label.setFont(QFont("Arial", 9))
            self.folder_form.addRow(level_label, row_layout)

        # Update template fields based on selections
        self._update_template_fields()

    def _on_folder_combo_changed(self):
        """Handle folder combo box selection change."""
        self._update_template_fields()
        self.update_preview()

    def _update_template_fields(self):
        """Update template input fields based on folder combo selections."""
        # Clear existing template fields
        while self.template_form.count():
            item = self.template_form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.template_edits.clear()

        # Find combined fields that need templates
        combined_fields = []
        for level, combo in self.folder_combos.items():
            field = combo.currentData()
            if field == 'drug_dose':
                combined_fields.append(('drug_dose', level))

        # Create template inputs for combined fields
        for field_name, level in combined_fields:
            if not self.sample_files:
                continue

            file_path = self.sample_files[self.current_sample_index]
            levels = self._get_folder_levels(file_path)

            # Find the folder value for this level
            folder_value = ""
            for lv, name in levels:
                if lv == level:
                    folder_value = name
                    break

            # Create template input
            row_layout = QHBoxLayout()

            value_label = QLabel(f'"{folder_value}"')
            value_label.setFont(QFont("Consolas", 9))
            value_label.setFixedWidth(150)
            value_label.setStyleSheet("background: #f0f0f0; padding: 4px; border-radius: 3px;")
            row_layout.addWidget(value_label)

            arrow_label = QLabel("→")
            arrow_label.setFixedWidth(20)
            row_layout.addWidget(arrow_label)

            template_edit = QLineEdit()
            template_edit.setFont(QFont("Consolas", 9))
            template_edit.setPlaceholderText("e.g., {drug}_{dose}")

            # Try to auto-detect a reasonable template
            if '_' in folder_value:
                template_edit.setText("{drug}_{dose}")
            elif ' ' in folder_value:
                template_edit.setText("{drug} {dose}")

            template_edit.textChanged.connect(self.update_preview)
            self.template_edits[field_name] = template_edit
            row_layout.addWidget(template_edit, 1)

            label = QLabel(f"Parse {field_name}:")
            label.setFont(QFont("Arial", 9))
            self.template_form.addRow(label, row_layout)

    def update_preview(self):
        """Update the preview based on current configuration."""
        if not self.sample_files:
            self.preview_label.setText("No files to preview")
            return

        try:
            config = self._build_config()
            if config is None:
                self.preview_label.setText("Configure folder mappings above to see preview")
                return

            extractor = MetadataExtractor(config)
            file_path = self.sample_files[self.current_sample_index]

            # Extract metadata
            metadata = extractor.extract_metadata(file_path)

            # Generate filename
            output_filename = extractor.generate_output_filename(file_path)

            # Build preview text
            preview_lines = ["Extracted Metadata:"]
            for key, value in sorted(metadata.items()):
                preview_lines.append(f"  {key}: {value}")

            preview_lines.append("")
            preview_lines.append(f"Output Filename: {output_filename}")

            self.preview_label.setText("\n".join(preview_lines))
            self.preview_label.setStyleSheet(
                "background: #e8f5e9; padding: 10px; border-radius: 4px; border: 1px solid #c8e6c9;"
            )

        except Exception as e:
            self.preview_label.setText(f"Error: {str(e)}")
            self.preview_label.setStyleSheet(
                "background: #ffebee; padding: 10px; border-radius: 4px; border: 1px solid #ffcdd2;"
            )

    def _build_config(self) -> Optional[MetadataConfig]:
        """Build MetadataConfig from current UI state."""
        folder_mappings = {}
        field_templates = {}

        # Get folder mappings
        for level, combo in self.folder_combos.items():
            field = combo.currentData()
            if field and field != 'skip':
                folder_mappings[level] = field

        if not folder_mappings:
            return None

        # Get templates
        for field_name, edit in self.template_edits.items():
            template = edit.text().strip()
            if template:
                field_templates[field_name] = template

        # Get match mode
        match_mode = 'first' if self.first_radio.isChecked() else 'last'

        return MetadataConfig(
            folder_mappings=folder_mappings,
            field_templates=field_templates,
            template_match_mode=match_mode
        )

    def _skip_metadata(self):
        """Skip metadata configuration - extract features without metadata."""
        reply = QMessageBox.question(
            self,
            "Skip Metadata",
            "Extract features without metadata columns?\n\n"
            "The output files will use the original H5 filenames and won't include "
            "cohort, drug, or dose information.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.result_config = None
            self.accept()

    def _confirm(self):
        """Confirm configuration and close dialog."""
        config = self._build_config()

        if config is None:
            QMessageBox.warning(
                self,
                "No Mappings",
                "Please assign at least one folder level to a metadata field,\n"
                "or click 'Skip' to extract without metadata."
            )
            return

        # Validate that rat_id will be extractable
        if not self.sample_files:
            QMessageBox.warning(self, "No Files", "No H5 files found in input folder.")
            return

        # Test extraction on sample
        try:
            extractor = MetadataExtractor(config)
            file_path = self.sample_files[self.current_sample_index]
            metadata = extractor.extract_metadata(file_path)

            if not metadata.get('rat_id'):
                reply = QMessageBox.question(
                    self,
                    "Missing Rat ID",
                    "Could not extract rat_id from the filename.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        except Exception as e:
            QMessageBox.warning(
                self,
                "Configuration Error",
                f"Error testing configuration:\n{str(e)}"
            )
            return

        self.result_config = config
        self.accept()

    def get_config(self) -> Optional[MetadataConfig]:
        """Get the configured MetadataConfig, or None if skipped."""
        return self.result_config


def show_metadata_config_dialog(input_folder: str, parent=None) -> Optional[MetadataConfig]:
    """
    Convenience function to show the metadata config dialog and return the result.

    Args:
        input_folder: Path to the folder containing H5 files
        parent: Parent widget

    Returns:
        MetadataConfig if configured, None if skipped or cancelled
    """
    dialog = MetadataConfigDialog(input_folder, parent)
    if dialog.exec() == QDialog.Accepted:
        return dialog.get_config()
    return None
