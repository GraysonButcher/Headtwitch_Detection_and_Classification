"""
CSV Editor Widget - Ground Truth Labeling Interface

Provides an interactive table for editing ground truth labels in feature CSV files.
Supports keyboard shortcuts and progress tracking.
"""

import os
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QProgressBar, QCheckBox, QMessageBox, QFileDialog,
    QHeaderView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont


class CSVEditorWidget(QWidget):
    """Interactive CSV editor for ground truth labeling."""

    # Signals
    labels_changed = Signal()  # Emitted when labels are modified
    progress_updated = Signal(int, int)  # Emitted with (labeled_count, total_count)

    def __init__(self, parent=None, project_manager=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.csv_path = None
        self.df = None
        self.unlabeled_value = '__'
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header section
        header_layout = QHBoxLayout()

        self.file_label = QLabel("No file loaded")
        self.file_label.setFont(QFont("Arial", 9))
        header_layout.addWidget(self.file_label)

        header_layout.addStretch()

        load_btn = QPushButton("📁 Load CSV")
        load_btn.clicked.connect(self.load_csv_dialog)
        header_layout.addWidget(load_btn)

        save_btn = QPushButton("💾 Save Changes")
        save_btn.clicked.connect(self.save_csv)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.save_btn = save_btn
        self.save_btn.setEnabled(False)
        header_layout.addWidget(save_btn)

        layout.addLayout(header_layout)

        # Filter section
        filter_layout = QHBoxLayout()

        self.show_unlabeled_only = QCheckBox("Show only unlabeled rows (__)")
        self.show_unlabeled_only.setChecked(True)
        self.show_unlabeled_only.stateChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.show_unlabeled_only)

        filter_layout.addStretch()

        # Progress bar
        self.progress_label = QLabel("Labeled: 0 / 0 (0%)")
        self.progress_label.setFont(QFont("Arial", 9))
        filter_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(20)
        filter_layout.addWidget(self.progress_bar)

        layout.addLayout(filter_layout)

        # Instructions
        instructions = QLabel(
            "💡 <b>Keyboard Shortcuts:</b> "
            "Press <b>1</b> for HTR, <b>0</b> for not HTR, <b>Space</b> to toggle, <b>↑/↓</b> to navigate"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #e7f3ff; padding: 6px; border-radius: 3px; color: #004085;")
        layout.addWidget(instructions)

        # Table
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)  # We'll handle editing via shortcuts
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.keyPressEvent = self.table_key_press_event
        self.table.cellDoubleClicked.connect(self.cell_double_clicked)
        layout.addWidget(self.table)

    def load_csv_dialog(self):
        """Open file dialog to load a CSV file."""
        # Determine default path
        default_path = ""
        if self.project_manager:
            project_path, _ = self.project_manager.get_current_project()
            if project_path:
                default_path = os.path.join(project_path, "features")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Feature CSV File",
            default_path,
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.load_csv(file_path)

    def load_csv(self, csv_path: str):
        """
        Load a CSV file into the editor.

        Args:
            csv_path: Path to the CSV file
        """
        try:
            self.csv_path = csv_path
            self.df = pd.read_csv(csv_path)

            # Verify ground_truth column exists
            if 'ground_truth' not in self.df.columns:
                QMessageBox.warning(
                    self,
                    "Invalid CSV",
                    "This CSV file does not have a 'ground_truth' column."
                )
                return

            # Convert ground_truth to string to handle mixed types
            self.df['ground_truth'] = self.df['ground_truth'].astype(str)

            # Update UI
            self.file_label.setText(f"📄 {os.path.basename(csv_path)}")
            self.save_btn.setEnabled(True)

            # Populate table
            self.populate_table()

            # Update progress
            self.update_progress()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading CSV", f"Failed to load CSV file:\n{str(e)}")

    def populate_table(self):
        """Populate the table widget with CSV data."""
        if self.df is None:
            return

        # Apply filter
        if self.show_unlabeled_only.isChecked():
            display_df = self.df[self.df['ground_truth'] == self.unlabeled_value].copy()
        else:
            display_df = self.df.copy()

        # Set up table
        self.table.clear()
        self.table.setRowCount(len(display_df))

        # Show key columns + ground_truth
        display_columns = ['start_frame', 'end_frame', 'duration_frames', 'ground_truth']
        # Add a few feature columns for context
        feature_cols = [c for c in self.df.columns if c not in display_columns and c not in ['Cohort', 'Dose', 'Rat ID']]
        display_columns.extend(feature_cols[:5])  # Show first 5 features

        self.table.setColumnCount(len(display_columns))
        self.table.setHorizontalHeaderLabels(display_columns)

        # Populate rows
        for row_idx, (df_idx, row) in enumerate(display_df.iterrows()):
            for col_idx, col_name in enumerate(display_columns):
                value = str(row[col_name])

                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, df_idx)  # Store original dataframe index

                # Color-code ground_truth column
                if col_name == 'ground_truth':
                    if value == '1' or value == '1.0':
                        item.setBackground(QColor("#d4edda"))  # Green - HTR
                        item.setForeground(QColor("#155724"))
                    elif value == '0' or value == '0.0':
                        item.setBackground(QColor("#f8d7da"))  # Red - Not HTR
                        item.setForeground(QColor("#721c24"))
                    else:  # Unlabeled
                        item.setBackground(QColor("#fff3cd"))  # Yellow - Unlabeled
                        item.setForeground(QColor("#856404"))
                    item.setFont(QFont("Arial", 9, QFont.Bold))

                self.table.setItem(row_idx, col_idx, item)

        # Resize columns
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

    def table_key_press_event(self, event):
        """Handle keyboard shortcuts in the table."""
        current_row = self.table.currentRow()
        if current_row < 0:
            return

        key = event.key()

        if key == Qt.Key_1:
            self.set_current_row_label('1')
        elif key == Qt.Key_0:
            self.set_current_row_label('0')
        elif key == Qt.Key_Space:
            self.toggle_current_row_label()
        elif key == Qt.Key_Up or key == Qt.Key_Down:
            # Let default behavior handle navigation
            QTableWidget.keyPressEvent(self.table, event)
        else:
            QTableWidget.keyPressEvent(self.table, event)

    def cell_double_clicked(self, row, column):
        """Handle double-click on a cell (toggle label)."""
        # Only toggle if ground_truth column
        if self.table.horizontalHeaderItem(column).text() == 'ground_truth':
            self.toggle_current_row_label()

    def set_current_row_label(self, label: str):
        """Set the label for the currently selected row."""
        current_row = self.table.currentRow()
        if current_row < 0:
            return

        # Get the original dataframe index from UserRole
        ground_truth_col_idx = self.get_column_index('ground_truth')
        if ground_truth_col_idx < 0:
            return

        item = self.table.item(current_row, ground_truth_col_idx)
        df_idx = item.data(Qt.UserRole)

        # Update dataframe
        self.df.at[df_idx, 'ground_truth'] = label

        # Update table display
        self.populate_table()

        # Move to next row
        if current_row < self.table.rowCount() - 1:
            self.table.setCurrentCell(current_row + 1, ground_truth_col_idx)

        # Update progress
        self.update_progress()

        # Emit signal
        self.labels_changed.emit()

    def toggle_current_row_label(self):
        """Toggle the current row's label between 0 and 1."""
        current_row = self.table.currentRow()
        if current_row < 0:
            return

        ground_truth_col_idx = self.get_column_index('ground_truth')
        if ground_truth_col_idx < 0:
            return

        item = self.table.item(current_row, ground_truth_col_idx)
        df_idx = item.data(Qt.UserRole)

        current_value = self.df.at[df_idx, 'ground_truth']

        # Toggle: __ -> 1, 1 -> 0, 0 -> 1
        if current_value == self.unlabeled_value:
            new_value = '1'
        elif current_value == '1' or current_value == '1.0':
            new_value = '0'
        else:
            new_value = '1'

        self.df.at[df_idx, 'ground_truth'] = new_value

        # Update display
        self.populate_table()

        # Move to next row
        if current_row < self.table.rowCount() - 1:
            self.table.setCurrentCell(current_row + 1, ground_truth_col_idx)

        # Update progress
        self.update_progress()

        # Emit signal
        self.labels_changed.emit()

    def get_column_index(self, column_name: str) -> int:
        """Get the index of a column by name."""
        for col_idx in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(col_idx).text() == column_name:
                return col_idx
        return -1

    def apply_filter(self):
        """Apply the unlabeled-only filter."""
        self.populate_table()

    def update_progress(self):
        """Update progress bar and label."""
        if self.df is None:
            return

        total_rows = len(self.df)
        labeled_rows = len(self.df[self.df['ground_truth'] != self.unlabeled_value])

        percentage = int((labeled_rows / total_rows) * 100) if total_rows > 0 else 0

        self.progress_label.setText(f"Labeled: {labeled_rows} / {total_rows} ({percentage}%)")
        self.progress_bar.setValue(percentage)

        # Emit progress signal
        self.progress_updated.emit(labeled_rows, total_rows)

    def save_csv(self):
        """Save changes and move to training folder if labels exist."""
        if self.df is None or self.csv_path is None:
            return

        try:
            # Save changes to current location first
            self.df.to_csv(self.csv_path, index=False)

            # Check if file has ANY labels (not just __)
            labeled_count = len(self.df[self.df['ground_truth'] != self.unlabeled_value])

            if labeled_count > 0 and self.project_manager:
                # File has labels - attempt to move to training folder
                moved = self._move_to_training_folder()
                if moved:
                    QMessageBox.information(
                        self,
                        "Saved",
                        f"Ground truth labels saved successfully!\n\n"
                        f"✓ {labeled_count} events labeled\n"
                        f"✓ File moved to training/ folder"
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Saved",
                        f"Ground truth labels saved successfully!\n\n"
                        f"{labeled_count} events labeled"
                    )
            else:
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Ground truth labels saved successfully to:\n{os.path.basename(self.csv_path)}"
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving",
                f"Failed to save CSV file:\n{str(e)}"
            )

    def _move_to_training_folder(self) -> bool:
        """
        Move CSV file from features/ to training/ folder.

        Returns:
            True if file was moved, False if not (already in training or error)
        """
        if not self.project_manager:
            return False

        project_path, _ = self.project_manager.get_current_project()
        if not project_path:
            return False

        # Get paths
        features_folder = os.path.join(project_path, "features")
        training_folder = os.path.join(project_path, "training")

        # Check if file is in features folder
        if not self.csv_path.startswith(features_folder):
            # File is not in features folder (might already be in training or elsewhere)
            return False

        # Calculate destination path
        filename = os.path.basename(self.csv_path)
        dest_path = os.path.join(training_folder, filename)

        # Check if destination already exists
        if os.path.exists(dest_path):
            # File already exists in training, just update it in place
            import shutil
            shutil.copy2(self.csv_path, dest_path)
            os.remove(self.csv_path)
        else:
            # Move file
            import shutil
            shutil.move(self.csv_path, dest_path)

        # Update internal path
        self.csv_path = dest_path

        # Update project config with training stats
        stats = self.get_labeling_stats()
        self.project_manager.add_training_file(
            filename,
            stats['htr_count'],
            stats['non_htr_count']
        )

        # Update file label to show new location
        self.file_label.setText(f"📄 {filename} (Training Data)")

        return True

    def get_labeling_stats(self) -> dict:
        """
        Get labeling statistics.

        Returns:
            Dict with stats: total, labeled, unlabeled, htr_count, non_htr_count
        """
        if self.df is None:
            return {
                'total': 0,
                'labeled': 0,
                'unlabeled': 0,
                'htr_count': 0,
                'non_htr_count': 0,
                'percentage_labeled': 0
            }

        total = len(self.df)
        unlabeled = len(self.df[self.df['ground_truth'] == self.unlabeled_value])
        labeled = total - unlabeled
        htr_count = len(self.df[self.df['ground_truth'].isin(['1', '1.0'])])
        non_htr_count = len(self.df[self.df['ground_truth'].isin(['0', '0.0'])])
        percentage = int((labeled / total) * 100) if total > 0 else 0

        return {
            'total': total,
            'labeled': labeled,
            'unlabeled': unlabeled,
            'htr_count': htr_count,
            'non_htr_count': non_htr_count,
            'percentage_labeled': percentage
        }
