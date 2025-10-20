"""
Video Inspector Widget - Video display and H5 signal visualization.

Handles video playback, H5 file loading, node mapping, and signal calculation
for the Tune Parameters tab.
"""

import os
import cv2
import h5py
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QLineEdit, QFileDialog, QMessageBox, QDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QIntValidator

from .node_mapping_dialog import NodeMappingDialog
from .detection_utils import (
    calculate_ear_distances,
    calculate_head_signal,
    normalize_sleap_tracks,
    normalize_sleap_scores
)


class VideoInspectorWidget(QWidget):
    """
    Video display widget with H5 signal calculation and frame navigation.

    Provides controls for loading video and H5 files, configuring node mapping,
    and navigating through frames. Calculates tracking signals once on H5 load,
    then allows fast frame-by-frame navigation.
    """

    # Signals
    signals_calculated = Signal(pd.DataFrame)  # Emitted after H5 signal calculation
    video_loaded = Signal()                    # Emitted after video loads
    frame_changed = Signal(int)                # Emitted when current frame changes
    h5_path_changed = Signal(str)              # Emitted when H5 file path changes

    def __init__(self, parent=None):
        """Initialize video inspector widget."""
        super().__init__(parent)

        # Video state
        self.video_path = None
        self.video_capture = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0

        # H5 state
        self.h5_path = None
        self.tracking_data = None      # Raw SLEAP tracks (n_frames, n_nodes, 2)
        self.point_scores = None       # Confidence scores (n_frames, n_nodes)
        self.signals_df = None         # Calculated signals DataFrame
        self.node_mapping = {          # Default node mapping
            'left_ear': 0,
            'right_ear': 1,
            'back': 2,
            'nose': 3,
            'head': 4
        }

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # File loading buttons
        self.create_file_controls(layout)

        # Video display area
        self.video_label = QLabel("Load H5 and Video files to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #aaaaaa;
                border: 2px solid #333333;
                border-radius: 4px;
                font-size: 14px;
            }
        """)
        self.video_label.setMinimumHeight(400)
        # Prevent label from expanding when pixmap is set - use Ignored horizontal policy
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        # Navigation controls
        self.create_navigation_controls(layout)

    def create_file_controls(self, parent_layout):
        """Create file loading buttons."""
        file_layout = QHBoxLayout()

        self.load_h5_btn = QPushButton("📁 Load H5 File...")
        self.load_h5_btn.setFont(QFont("Arial", 9))
        self.load_h5_btn.setToolTip("Load SLEAP tracking data (.h5 file)")
        self.load_h5_btn.clicked.connect(self.load_h5_file)
        file_layout.addWidget(self.load_h5_btn)

        self.load_video_btn = QPushButton("🎬 Load Video...")
        self.load_video_btn.setFont(QFont("Arial", 9))
        self.load_video_btn.setToolTip("Load video file (.mp4, .avi, etc.)")
        self.load_video_btn.clicked.connect(self.load_video_file)
        file_layout.addWidget(self.load_video_btn)

        file_layout.addStretch()

        parent_layout.addLayout(file_layout)

    def create_navigation_controls(self, parent_layout):
        """Create frame navigation controls."""
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        parent_layout.addWidget(self.frame_slider)

        # Navigation buttons and frame display
        nav_layout = QHBoxLayout()

        # Previous frame buttons
        self.prev_5_btn = QPushButton("◀◀ 5")
        self.prev_5_btn.setMaximumWidth(60)
        self.prev_5_btn.setToolTip("Back 5 frames")
        self.prev_5_btn.clicked.connect(lambda: self.navigate_frames(-5))
        nav_layout.addWidget(self.prev_5_btn)

        self.prev_1_btn = QPushButton("◀ 1")
        self.prev_1_btn.setMaximumWidth(60)
        self.prev_1_btn.setToolTip("Back 1 frame")
        self.prev_1_btn.clicked.connect(lambda: self.navigate_frames(-1))
        nav_layout.addWidget(self.prev_1_btn)

        # Frame display
        self.frame_display = QLabel("Frame: 0 / 0")
        self.frame_display.setAlignment(Qt.AlignCenter)
        self.frame_display.setFont(QFont("Arial", 9))
        self.frame_display.setMinimumWidth(120)
        nav_layout.addWidget(self.frame_display)

        # Next frame buttons
        self.next_1_btn = QPushButton("1 ▶")
        self.next_1_btn.setMaximumWidth(60)
        self.next_1_btn.setToolTip("Forward 1 frame")
        self.next_1_btn.clicked.connect(lambda: self.navigate_frames(1))
        nav_layout.addWidget(self.next_1_btn)

        self.next_5_btn = QPushButton("5 ▶▶")
        self.next_5_btn.setMaximumWidth(60)
        self.next_5_btn.setToolTip("Forward 5 frames")
        self.next_5_btn.clicked.connect(lambda: self.navigate_frames(5))
        nav_layout.addWidget(self.next_5_btn)

        nav_layout.addSpacing(20)

        # Jump to frame
        jump_label = QLabel("Go to:")
        jump_label.setFont(QFont("Arial", 9))
        nav_layout.addWidget(jump_label)

        self.jump_input = QLineEdit()
        self.jump_input.setMaximumWidth(80)
        self.jump_input.setPlaceholderText("frame #")
        self.jump_input.setValidator(QIntValidator(0, 999999))
        self.jump_input.returnPressed.connect(self.jump_to_frame)
        nav_layout.addWidget(self.jump_input)

        self.jump_btn = QPushButton("Go")
        self.jump_btn.setMaximumWidth(50)
        self.jump_btn.clicked.connect(self.jump_to_frame)
        nav_layout.addWidget(self.jump_btn)

        nav_layout.addStretch()

        parent_layout.addLayout(nav_layout)

        # Initially disable navigation until files are loaded
        self.set_navigation_enabled(False)

    def set_navigation_enabled(self, enabled: bool):
        """Enable or disable navigation controls."""
        self.frame_slider.setEnabled(enabled)
        self.prev_5_btn.setEnabled(enabled)
        self.prev_1_btn.setEnabled(enabled)
        self.next_1_btn.setEnabled(enabled)
        self.next_5_btn.setEnabled(enabled)
        self.jump_input.setEnabled(enabled)
        self.jump_btn.setEnabled(enabled)

    # ==================== H5 Loading ====================

    def load_h5_file(self):
        """Load H5 file with node mapping configuration."""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SLEAP H5 File",
            "",
            "H5 Files (*.h5 *.hdf5);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Read node names from H5 file
            with h5py.File(file_path, 'r') as f:
                # Try to read node names
                node_names = []
                if 'node_names' in f:
                    raw_names = f['node_names'][:]
                    node_names = [
                        n.decode('utf-8') if isinstance(n, (bytes, bytearray)) else str(n)
                        for n in raw_names
                    ]
                else:
                    # Fallback: use indices if node names not available
                    tracks = f['tracks'][:]
                    n_nodes = self._get_num_nodes(tracks)
                    node_names = [f"Node {i}" for i in range(n_nodes)]

            # Show node mapping dialog
            dialog = NodeMappingDialog(node_names, self.node_mapping, self)

            if dialog.exec() != 1:  # QDialog.Accepted = 1
                return  # User cancelled

            # Get configured mapping
            self.node_mapping = dialog.get_mapping()

            # Load H5 data and calculate signals
            self.h5_path = file_path
            self.h5_path_changed.emit(file_path)
            self._load_h5_data()
            self._calculate_signals()

            # Update status
            filename = os.path.basename(file_path)
            self.video_label.setText(f"H5 Loaded: {filename}\n\nLoad a video file to display frames")

            QMessageBox.information(
                self,
                "H5 Loaded",
                f"Successfully loaded H5 file and calculated signals.\n\n"
                f"Total frames: {len(self.signals_df)}\n"
                f"Nodes configured: {len(self.node_mapping)}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading H5",
                f"Failed to load H5 file:\n\n{str(e)}"
            )

    def _get_num_nodes(self, tracks_array):
        """Determine number of nodes from tracks array shape."""
        arr = np.squeeze(tracks_array)
        # Try to find dimension that represents nodes (not 2, not largest)
        for dim_size in arr.shape:
            if dim_size != 2 and dim_size < max(arr.shape):
                return dim_size
        return 5  # Default fallback

    def _load_h5_data(self):
        """Load tracking data and point scores from H5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            # Load tracks
            tracks_raw = f['tracks'][:]
            self.tracking_data = normalize_sleap_tracks(tracks_raw)

            # Load point scores if available
            if 'point_scores' in f:
                scores_raw = f['point_scores'][:]
                n_frames, n_nodes = self.tracking_data.shape[0], self.tracking_data.shape[1]
                self.point_scores = normalize_sleap_scores(scores_raw, n_frames, n_nodes)
            else:
                # Create dummy scores (all 1.0)
                n_frames, n_nodes = self.tracking_data.shape[0], self.tracking_data.shape[1]
                self.point_scores = np.ones((n_frames, n_nodes))

        self.total_frames = self.tracking_data.shape[0]

    def _calculate_signals(self):
        """Calculate ear distances and head signal from tracking data."""
        # Extract node locations based on mapping
        left_ear_locs = self.tracking_data[:, self.node_mapping['left_ear'], :]
        right_ear_locs = self.tracking_data[:, self.node_mapping['right_ear'], :]
        back_locs = self.tracking_data[:, self.node_mapping['back'], :]
        nose_locs = self.tracking_data[:, self.node_mapping['nose'], :]
        head_locs = self.tracking_data[:, self.node_mapping['head'], :]

        # Calculate ear distances
        left_distances, right_distances = calculate_ear_distances(
            left_ear_locs, right_ear_locs, back_locs, nose_locs
        )

        # Calculate head signal
        head_distances = calculate_head_signal(head_locs, back_locs, nose_locs)

        # Create DataFrame
        self.signals_df = pd.DataFrame({
            'frame': np.arange(len(left_distances)),
            'left_ear_dist': left_distances,
            'right_ear_dist': right_distances,
            'head_dist': head_distances
        })

        # Emit signal that calculations are complete
        self.signals_calculated.emit(self.signals_df)

    # ==================== Video Loading ====================

    def load_video_file(self):
        """Load video file for display."""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Release existing video if any
            if self.video_capture is not None:
                self.video_capture.release()

            # Open video
            self.video_capture = cv2.VideoCapture(file_path)

            if not self.video_capture.isOpened():
                raise Exception("Could not open video file")

            self.video_path = file_path

            # Get video properties
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0 or np.isnan(self.fps):
                self.fps = 30.0  # Default fallback

            video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # If H5 is loaded, use minimum of video and H5 frames
            if self.signals_df is not None:
                self.total_frames = min(video_frame_count, len(self.signals_df))
            else:
                self.total_frames = video_frame_count

            # Update slider range
            self.frame_slider.setMaximum(max(0, self.total_frames - 1))

            # Enable navigation
            self.set_navigation_enabled(True)

            # Display first frame
            self.current_frame = 0
            self.update_video_frame()

            # Emit signal
            self.video_loaded.emit()

            filename = os.path.basename(file_path)
            QMessageBox.information(
                self,
                "Video Loaded",
                f"Successfully loaded video:\n\n{filename}\n\n"
                f"Total frames: {self.total_frames}\n"
                f"FPS: {self.fps:.2f}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Video",
                f"Failed to load video file:\n\n{str(e)}"
            )

    # ==================== Frame Navigation ====================

    def navigate_frames(self, delta: int):
        """Navigate forward or backward by delta frames."""
        new_frame = self.current_frame + delta
        new_frame = max(0, min(self.total_frames - 1, new_frame))
        self.set_frame(new_frame)

    def jump_to_frame(self):
        """Jump to frame number entered in jump_input."""
        try:
            frame_num = int(self.jump_input.text())
            frame_num = max(0, min(self.total_frames - 1, frame_num))
            self.set_frame(frame_num)
            self.jump_input.clear()
        except ValueError:
            pass

    def on_slider_changed(self, value: int):
        """Handle slider value change."""
        if value != self.current_frame:
            self.set_frame(value)

    def set_frame(self, frame_num: int):
        """Set current frame and update display."""
        self.current_frame = frame_num
        self.update_video_frame()
        self.update_frame_display()

        # Update slider without triggering signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)

        # Emit frame changed signal
        self.frame_changed.emit(self.current_frame)

    def update_frame_display(self):
        """Update frame number display label."""
        self.frame_display.setText(f"Frame: {self.current_frame} / {self.total_frames - 1}")

    def update_video_frame(self):
        """Update video display with current frame."""
        if self.video_capture is None:
            return

        # Seek to frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.video_capture.read()

        if not ret:
            return

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    # ==================== Cleanup ====================

    def closeEvent(self, event):
        """Clean up resources when widget is closed."""
        if self.video_capture is not None:
            self.video_capture.release()
        super().closeEvent(event)
