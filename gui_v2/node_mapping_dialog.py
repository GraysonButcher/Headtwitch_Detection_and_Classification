"""
Node Mapping Dialog - Configure SLEAP node indices for HTR detection.

Allows users to map detected SLEAP nodes to the required anatomical landmarks
before signal calculation.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import Dict, List


class NodeMappingDialog(QDialog):
    """
    Dialog for configuring SLEAP node indices before signal calculation.

    Shows detected node names from H5 file and allows user to map them
    to the required anatomical landmarks (left_ear, right_ear, back, nose, head).
    """

    def __init__(self, node_names: List[str], default_mapping: Dict[str, int], parent=None):
        """
        Initialize node mapping dialog.

        Args:
            node_names: List of node names detected in H5 file
            default_mapping: Default node mapping from config
            parent: Parent widget
        """
        super().__init__(parent)
        self.node_names = node_names
        self.default_mapping = default_mapping
        self.result_mapping = default_mapping.copy()
        self.combo_boxes = {}

        self.setWindowTitle("Configure Node Mapping")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.init_ui()
        self.auto_detect_mapping()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Configure SLEAP Node Mapping")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)

        # Instructions
        instructions = QLabel(
            "Map the detected SLEAP nodes to the required anatomical landmarks.\n"
            "The system will use these mappings to calculate head-twitch signals."
        )
        instructions.setFont(QFont("Arial", 9))
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(instructions)

        # Detected nodes display
        nodes_group = QGroupBox("Detected Nodes in H5 File")
        nodes_layout = QVBoxLayout(nodes_group)

        # Create scrollable text area for node list
        nodes_text = QTextEdit()
        nodes_text.setReadOnly(True)
        nodes_text.setMaximumHeight(100)
        nodes_text.setFont(QFont("Consolas", 9))

        # Populate with node list
        node_list_text = ""
        for i, name in enumerate(self.node_names):
            node_list_text += f"  {i}: {name}\n"

        nodes_text.setPlainText(node_list_text)
        nodes_layout.addWidget(nodes_text)
        layout.addWidget(nodes_group)

        # Mapping configuration
        mapping_group = QGroupBox("Anatomical Landmark Mapping")
        mapping_layout = QFormLayout(mapping_group)
        mapping_layout.setSpacing(10)

        # Create combo boxes for each required landmark
        required_landmarks = [
            ("left_ear", "Left Ear:"),
            ("right_ear", "Right Ear:"),
            ("back", "Back:"),
            ("nose", "Nose:"),
            ("head", "Head:")
        ]

        for key, label in required_landmarks:
            combo = QComboBox()
            combo.setFont(QFont("Arial", 9))

            # Add all node indices as options
            for i, name in enumerate(self.node_names):
                combo.addItem(f"{i}: {name}", i)

            # Set default value
            default_idx = self.default_mapping.get(key, 0)
            combo.setCurrentIndex(default_idx)

            # Connect change signal
            combo.currentIndexChanged.connect(self.on_mapping_changed)

            self.combo_boxes[key] = combo
            mapping_layout.addRow(label, combo)

        layout.addWidget(mapping_group)

        # Validation warning label
        self.validation_label = QLabel("")
        self.validation_label.setFont(QFont("Arial", 8))
        self.validation_label.setStyleSheet("color: #d32f2f; padding: 5px;")
        self.validation_label.setVisible(False)
        layout.addWidget(self.validation_label)

        # Buttons
        button_layout = QHBoxLayout()

        use_defaults_btn = QPushButton("Use Defaults")
        use_defaults_btn.setFont(QFont("Arial", 9))
        use_defaults_btn.clicked.connect(self.use_defaults)
        button_layout.addWidget(use_defaults_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Arial", 9))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        confirm_btn = QPushButton("Confirm")
        confirm_btn.setFont(QFont("Arial", 9, QFont.Bold))
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        confirm_btn.clicked.connect(self.confirm)
        button_layout.addWidget(confirm_btn)

        layout.addLayout(button_layout)

    def auto_detect_mapping(self):
        """
        Attempt to auto-detect node mapping based on node names.

        Looks for common naming patterns like "left-ear", "right_ear", etc.
        """
        # Common name patterns for each landmark
        patterns = {
            "left_ear": ["left-ear", "left_ear", "leftear", "l-ear", "l_ear", "lear"],
            "right_ear": ["right-ear", "right_ear", "rightear", "r-ear", "r_ear", "rear"],
            "back": ["back", "base", "body"],
            "nose": ["nose", "snout"],
            "head": ["head", "top"]
        }

        for landmark, pattern_list in patterns.items():
            for i, node_name in enumerate(self.node_names):
                node_name_lower = node_name.lower()
                if any(pattern in node_name_lower for pattern in pattern_list):
                    # Found a match - update combo box
                    self.combo_boxes[landmark].setCurrentIndex(i)
                    break

    def on_mapping_changed(self):
        """Handle mapping change - validate for duplicates."""
        self.validate_mapping()

    def validate_mapping(self) -> bool:
        """
        Validate current mapping configuration.

        Checks for duplicate indices (same node used for multiple landmarks).

        Returns:
            True if mapping is valid, False otherwise
        """
        # Get current selections
        selections = {key: combo.currentData() for key, combo in self.combo_boxes.items()}

        # Check for duplicates
        indices = list(selections.values())
        if len(indices) != len(set(indices)):
            # Found duplicates
            self.validation_label.setText(
                "⚠ Warning: The same node is assigned to multiple landmarks. "
                "This may cause incorrect signal calculation."
            )
            self.validation_label.setVisible(True)
            return False
        else:
            self.validation_label.setVisible(False)
            return True

    def use_defaults(self):
        """Reset all mappings to default values from config."""
        for key, combo in self.combo_boxes.items():
            default_idx = self.default_mapping.get(key, 0)
            combo.setCurrentIndex(default_idx)

    def confirm(self):
        """Confirm mapping and close dialog."""
        # Validate before accepting
        if not self.validate_mapping():
            reply = QMessageBox.question(
                self,
                "Duplicate Mapping",
                "The same node is assigned to multiple landmarks.\n\n"
                "This may cause incorrect results. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Store result mapping
        self.result_mapping = {
            key: combo.currentData()
            for key, combo in self.combo_boxes.items()
        }

        self.accept()

    def get_mapping(self) -> Dict[str, int]:
        """
        Get the configured node mapping.

        Returns:
            Dictionary mapping landmark names to node indices
        """
        return self.result_mapping
