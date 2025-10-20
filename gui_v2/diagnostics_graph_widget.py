"""
Diagnostics Graph Widget - Signal visualization with event overlays.

Displays ear distance and head signal plots with detected event overlays
for real-time parameter tuning feedback.
"""

import pandas as pd
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Signal
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class DiagnosticsGraphWidget(QWidget):
    """
    Matplotlib canvas widget for displaying tracking signals and detection results.

    Shows three signal lines (left ear, right ear, head) with colored event overlays
    and a frame cursor for synchronized video/graph navigation.
    """

    # Signals
    view_range_changed = Signal(int, int)  # Emitted when user pans/zooms graph

    def __init__(self, parent=None):
        """Initialize diagnostics graph widget."""
        super().__init__(parent)

        # State
        self.signals_df = None        # Base signal data
        self.current_frame = 0        # Current frame cursor position
        self.view_start = 0           # X-axis view start
        self.view_end = 100           # X-axis view end
        self.event_spans = []         # List of matplotlib span patches

        # Signal line references (for efficient updates)
        self.left_ear_line = None
        self.right_ear_line = None
        self.head_line = None
        self.frame_cursor = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create matplotlib figure and canvas matching legacy implementation
        # figsize=(8, 6) provides proper proportions for text/margins/legend
        # dpi=150 gives good quality without extreme resource usage
        self.figure = Figure(figsize=(8, 6), dpi=150)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create axes
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Frame", fontsize=9)
        self.ax.set_ylabel("Distance (pixels)", fontsize=9)
        self.ax.set_title("Tracking Signals", fontsize=10, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # Invert y-axis (signals typically go down when head moves)
        self.ax.invert_yaxis()

        # Tight layout
        self.figure.tight_layout()

        layout.addWidget(self.canvas)

    def set_signals(self, signals_df: pd.DataFrame):
        """
        Set base signal data and plot initial lines.

        This is called once when H5 file is loaded. Subsequent updates
        only modify event overlays and frame cursor, not signal lines.

        Args:
            signals_df: DataFrame with columns ['frame', 'left_ear_dist', 'right_ear_dist', 'head_dist']
        """
        self.signals_df = signals_df

        # Clear axes
        self.ax.clear()

        # Plot signal lines
        frames = signals_df['frame'].values
        left_ear = signals_df['left_ear_dist'].values
        right_ear = signals_df['right_ear_dist'].values
        head = signals_df['head_dist'].values

        self.left_ear_line, = self.ax.plot(
            frames, left_ear,
            color='#1f77b4', linewidth=1.5, label='Left Ear',
            alpha=0.8
        )

        self.right_ear_line, = self.ax.plot(
            frames, right_ear,
            color='#ff7f0e', linewidth=1.5, label='Right Ear',
            alpha=0.8
        )

        self.head_line, = self.ax.plot(
            frames, head,
            color='#2ca02c', linewidth=1.5, label='Head→Midline',
            linestyle='--', alpha=0.8
        )

        # Add frame cursor (vertical line at frame 0)
        self.frame_cursor = self.ax.axvline(
            x=0, color='red', linestyle='--', linewidth=2, alpha=0.7
        )

        # Configure axes
        self.ax.set_xlabel("Frame", fontsize=9)
        self.ax.set_ylabel("Distance (pixels)", fontsize=9)
        self.ax.set_title("Tracking Signals", fontsize=10, fontweight='bold')

        # Add signal legend
        signal_legend = self.ax.legend(loc='upper right', fontsize=8)

        # Add compact event detection legend below signal legend
        from matplotlib.patches import Patch
        event_legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.25, label='Combined'),
            Patch(facecolor='#ff7f0e', alpha=0.25, label='Ear Only'),
            Patch(facecolor='#d62728', alpha=0.25, label='Head Only')
        ]
        event_legend = self.ax.legend(
            handles=event_legend_elements,
            loc='upper left',
            fontsize=7,
            title='Detection',
            title_fontsize=7,
            framealpha=0.9
        )

        # Add signal legend back (matplotlib only keeps one legend by default)
        self.ax.add_artist(signal_legend)

        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.invert_yaxis()

        # Set initial view range
        total_frames = len(frames)
        self.view_start = 0
        self.view_end = min(total_frames, 100)  # Initial window of 100 frames
        self.ax.set_xlim(self.view_start, self.view_end)

        # Auto-scale y-axis
        self._autoscale_y()

        self.figure.tight_layout()
        self.canvas.draw()

    def update_frame_cursor(self, frame_num: int):
        """
        Update frame cursor position (fast operation).

        Args:
            frame_num: Frame number to position cursor at
        """
        if self.frame_cursor is None:
            return

        self.current_frame = frame_num

        # Update cursor line position
        self.frame_cursor.set_xdata([frame_num])

        # Auto-pan view if cursor moves outside visible range
        if frame_num < self.view_start or frame_num > self.view_end:
            self._auto_pan_to_frame(frame_num)

        # Redraw canvas (fast, no full replot)
        self.canvas.draw_idle()

    def update_events(self, events_df: pd.DataFrame):
        """
        Update event detection overlays.

        Args:
            events_df: DataFrame with columns ['start_frame', 'end_frame', 'confidence', 'detection_method']
                      confidence: 'High', 'Medium', 'Low'
        """
        # Remove old event spans
        for span in self.event_spans:
            span.remove()
        self.event_spans.clear()

        if events_df is None or len(events_df) == 0:
            self.canvas.draw()
            return

        # Color mapping for confidence levels
        colors = {
            'High': '#2ca02c',    # Green
            'Medium': '#ff7f0e',  # Orange
            'Low': '#d62728'      # Red
        }

        # Add event spans
        for _, event in events_df.iterrows():
            start = event['start_frame']
            end = event['end_frame']
            confidence = event.get('confidence', 'Low')
            color = colors.get(confidence, '#888888')

            # Create span patch
            span = self.ax.axvspan(
                start, end,
                color=color,
                alpha=0.25,
                zorder=1  # Behind lines
            )
            self.event_spans.append(span)

        # Redraw canvas
        self.canvas.draw()

    def clear_events(self):
        """Remove all event overlays."""
        for span in self.event_spans:
            span.remove()
        self.event_spans.clear()
        self.canvas.draw()

    def set_view_range(self, start: int, end: int):
        """
        Set x-axis view range (for zoom and pan).

        Args:
            start: Start frame of visible range
            end: End frame of visible range
        """
        if self.signals_df is None:
            return

        total_frames = len(self.signals_df)

        # Clamp to valid range
        start = max(0, min(start, total_frames - 1))
        end = max(start + 1, min(end, total_frames))

        self.view_start = start
        self.view_end = end

        # Update axes limits
        self.ax.set_xlim(start, end)
        self._autoscale_y()

        # Redraw canvas
        self.canvas.draw()

        # Emit signal
        self.view_range_changed.emit(start, end)

    def _auto_pan_to_frame(self, frame_num: int):
        """
        Automatically pan view to keep frame in center of visible range.

        Args:
            frame_num: Frame to center on
        """
        if self.signals_df is None:
            return

        total_frames = len(self.signals_df)
        window_size = self.view_end - self.view_start

        # Calculate new view range centered on frame
        half_window = window_size // 2
        new_start = max(0, frame_num - half_window)
        new_end = min(total_frames, new_start + window_size)

        # Adjust start if we hit the end boundary
        if new_end == total_frames:
            new_start = max(0, new_end - window_size)

        self.set_view_range(new_start, new_end)

    def _autoscale_y(self):
        """Auto-scale y-axis based on visible data range."""
        if self.signals_df is None:
            return

        # Get visible data
        visible_data = self.signals_df[
            (self.signals_df['frame'] >= self.view_start) &
            (self.signals_df['frame'] <= self.view_end)
        ]

        if len(visible_data) == 0:
            return

        # Find min/max of all signals in visible range
        all_values = np.concatenate([
            visible_data['left_ear_dist'].values,
            visible_data['right_ear_dist'].values,
            visible_data['head_dist'].values
        ])

        # Filter out NaN and inf
        all_values = all_values[np.isfinite(all_values)]

        if len(all_values) == 0:
            return

        y_min = np.min(all_values)
        y_max = np.max(all_values)

        # Add 10% padding
        y_range = y_max - y_min
        if y_range > 0:
            padding = y_range * 0.1
            self.ax.set_ylim(y_max + padding, y_min - padding)  # Inverted axis

    def zoom_in(self):
        """Zoom in by 50%."""
        window_size = self.view_end - self.view_start
        new_window = max(10, window_size // 2)  # Minimum 10 frames
        center = (self.view_start + self.view_end) // 2
        new_start = center - new_window // 2
        new_end = new_start + new_window
        self.set_view_range(new_start, new_end)

    def zoom_out(self):
        """Zoom out by 2x."""
        window_size = self.view_end - self.view_start
        new_window = window_size * 2
        center = (self.view_start + self.view_end) // 2
        new_start = center - new_window // 2
        new_end = new_start + new_window
        self.set_view_range(new_start, new_end)

    def zoom_to_full(self):
        """Zoom to show all data."""
        if self.signals_df is None:
            return
        total_frames = len(self.signals_df)
        self.set_view_range(0, total_frames)

    def get_view_range(self):
        """
        Get current visible frame range.

        Returns:
            Tuple of (start_frame, end_frame)
        """
        return (self.view_start, self.view_end)
