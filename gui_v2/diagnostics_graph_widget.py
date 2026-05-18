"""
Diagnostics Graph Widget - Signal visualization with event overlays.

Displays ear distance and head signal plots with detected event overlays
for real-time parameter tuning feedback.
"""

import pandas as pd
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSizePolicy
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from .theme import Fonts
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks, savgol_filter


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

        # Peak/valley marker state
        self.show_peaks = False
        self.show_cycle_lines = False  # Toggle for connector lines (Option B)
        self.show_oscillation_numbers = False  # Toggle for oscillation group numbering
        self.peak_params = {
            # Head detector params (uses prominence)
            'head_prominence': 1,
            'head_distance': 1,
            'head_amplitude_threshold': 2,  # For cycle pass/fail visualization
            'head_max_cycle_gap': 5,  # For grouping cycles into oscillation groups
            'head_min_oscillations': 3,  # Minimum oscillations for valid group
            'head_use_smoothing': True,  # Apply smoothing to head signal (matches detector)
            'head_smoothing_window': 5,  # Savitzky-Golay window size
            'head_smoothing_polyorder': 2,  # Savitzky-Golay polynomial order
            # Ear detector mode and params
            'ear_use_prominence_mode': False,  # If True, use prominence instead of height
            'ear_prominence': 5,  # Prominence value (used when ear_use_prominence_mode=True)
            'ear_peak_height': 10,  # Height threshold (used when ear_use_prominence_mode=False)
            'ear_valley_height': 10,  # Height threshold (used when ear_use_prominence_mode=False)
            'ear_distance': 2,  # max_gap in detector
            'ear_quick_gap': 5,  # For crisscross pairing
            'ear_between_unit_gap': 15,  # For grouping crisscrosses
            'ear_min_crisscrosses': 2  # Minimum crisscrosses for valid group
        }
        # Cached smoothed head signal (computed in _calculate_peaks)
        self.head_signal_smoothed = None
        # Peak/valley scatter plot references
        self.left_ear_peaks_scatter = None
        self.left_ear_valleys_scatter = None
        self.right_ear_peaks_scatter = None
        self.right_ear_valleys_scatter = None
        self.head_peaks_scatter = None
        self.head_valleys_scatter = None
        self.head_peaks_failing_scatter = None  # Separate scatter for failing peaks
        self.head_valleys_failing_scatter = None  # Separate scatter for failing valleys
        # Cached peak/valley indices
        self.left_ear_peak_indices = np.array([])
        self.left_ear_valley_indices = np.array([])
        self.right_ear_peak_indices = np.array([])
        self.right_ear_valley_indices = np.array([])
        self.head_peak_indices = np.array([])
        self.head_valley_indices = np.array([])
        # Head cycle data (for amplitude threshold visualization)
        self.head_cycles = []  # List of {'start_idx', 'end_idx', 'amplitude', 'passes'}
        self.cycle_lines = []  # Matplotlib line objects for connector lines
        # Oscillation group data
        self.head_oscillation_groups = []  # List of groups, each group is list of cycles with numbers
        self.ear_crisscross_groups = []  # List of crisscross groups with numbers
        self.oscillation_annotations = []  # Matplotlib text annotations

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Controls row with toggle checkbox
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 2, 5, 2)

        self.show_peaks_checkbox = QCheckBox("Show Peaks/Valleys")
        self.show_peaks_checkbox.setFont(QFont(Fonts.FAMILY, 8))
        self.show_peaks_checkbox.setToolTip(
            "Show detected peaks and valleys on signal lines.\n"
            "Filled circles = peaks, hollow circles = valleys.\n"
            "Green = passes amplitude threshold, Gray = fails.\n"
            "Updates when you click Reanalyze.\n\n"
            "Ear signals: controlled by Peak/Valley Thresh and Max Gap\n"
            "Head signal: controlled by Peak Prominence and Peak Distance"
        )
        self.show_peaks_checkbox.setChecked(False)
        self.show_peaks_checkbox.stateChanged.connect(self._on_show_peaks_changed)
        controls_layout.addWidget(self.show_peaks_checkbox)

        self.show_cycles_checkbox = QCheckBox("Show Cycle Lines")
        self.show_cycles_checkbox.setFont(QFont(Fonts.FAMILY, 8))
        self.show_cycles_checkbox.setToolTip(
            "Draw connector lines between peak-valley pairs (cycles).\n"
            "Green line = cycle amplitude passes Amp Thresh\n"
            "Gray line = cycle amplitude fails Amp Thresh\n"
            "Line length shows the amplitude of each cycle."
        )
        self.show_cycles_checkbox.setChecked(False)
        self.show_cycles_checkbox.stateChanged.connect(self._on_show_cycles_changed)
        controls_layout.addWidget(self.show_cycles_checkbox)

        self.show_osc_numbers_checkbox = QCheckBox("Show Osc #")
        self.show_osc_numbers_checkbox.setFont(QFont(Fonts.FAMILY, 8))
        self.show_osc_numbers_checkbox.setToolTip(
            "Show oscillation numbers within groups.\n"
            "Each cycle in a group is numbered (1, 2, 3...).\n"
            "Only shown for groups with 3+ oscillations.\n"
            "Green = valid group, Gray = group too small.\n\n"
            "Head: grouped by Max Cycle Gap, needs Min Oscillations\n"
            "Ears: grouped by Unit Gap, needs Min Crisscrosses"
        )
        self.show_osc_numbers_checkbox.setChecked(False)
        self.show_osc_numbers_checkbox.stateChanged.connect(self._on_show_osc_numbers_changed)
        controls_layout.addWidget(self.show_osc_numbers_checkbox)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

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
            try:
                span.remove()
            except ValueError:
                pass  # Artist already removed
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
            try:
                span.remove()
            except ValueError:
                pass  # Artist already removed
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

    # ==================== Peak Marker Methods ====================

    def _on_show_peaks_changed(self, state):
        """Handle show peaks checkbox state change."""
        self.show_peaks = bool(state)
        if self.show_peaks:
            self._draw_peaks()
        else:
            self._hide_peaks()
        self.canvas.draw_idle()

    def update_peaks(self, head_prominence: int = 1, head_distance: int = 1,
                     head_amplitude_threshold: int = 2,
                     head_max_cycle_gap: int = 5, head_min_oscillations: int = 3,
                     head_use_smoothing: bool = True, head_smoothing_window: int = 5,
                     head_smoothing_polyorder: int = 2,
                     ear_use_prominence_mode: bool = False, ear_prominence: int = 5,
                     ear_peak_height: int = 10, ear_valley_height: int = 10,
                     ear_distance: int = 2, ear_quick_gap: int = 5,
                     ear_between_unit_gap: int = 15, ear_min_crisscrosses: int = 2):
        """
        Recalculate peaks/valleys with new parameters and update display.

        Called after reanalysis to reflect current parameter settings.
        Parameters match the actual detector logic in core/detectors.py.

        Args:
            head_prominence: Peak prominence for head signal (scipy.signal.find_peaks prominence param)
            head_distance: Minimum distance between peaks for head signal (peak_distance param)
            head_amplitude_threshold: Minimum amplitude for head cycles (amplitude_threshold param)
            head_max_cycle_gap: Maximum gap between cycles to group them (max_cycle_gap param)
            head_min_oscillations: Minimum oscillations for valid group (min_oscillations param)
            head_use_smoothing: If True, apply Savitzky-Golay smoothing to head signal (use_smoothing param)
            head_smoothing_window: Window size for Savitzky-Golay filter (smoothing_window param)
            head_smoothing_polyorder: Polynomial order for Savitzky-Golay filter (smoothing_polyorder param)
            ear_use_prominence_mode: If True, use prominence-based detection for ears
            ear_prominence: Prominence value for ear peaks/valleys (used when ear_use_prominence_mode=True)
            ear_peak_height: Minimum height for ear peaks (peak_threshold param, used when ear_use_prominence_mode=False)
            ear_valley_height: Minimum depth for ear valleys (valley_threshold param, used when ear_use_prominence_mode=False)
            ear_distance: Minimum distance between ear peaks/valleys (max_gap param)
            ear_quick_gap: Maximum gap for crisscross pairing (quick_gap param)
            ear_between_unit_gap: Maximum gap between crisscrosses to group them (between_unit_gap param)
            ear_min_crisscrosses: Minimum crisscrosses for valid group (min_crisscrosses param)
        """
        self.peak_params = {
            'head_prominence': head_prominence,
            'head_distance': head_distance,
            'head_amplitude_threshold': head_amplitude_threshold,
            'head_max_cycle_gap': head_max_cycle_gap,
            'head_min_oscillations': head_min_oscillations,
            'head_use_smoothing': head_use_smoothing,
            'head_smoothing_window': head_smoothing_window,
            'head_smoothing_polyorder': head_smoothing_polyorder,
            'ear_use_prominence_mode': ear_use_prominence_mode,
            'ear_prominence': ear_prominence,
            'ear_peak_height': ear_peak_height,
            'ear_valley_height': ear_valley_height,
            'ear_distance': ear_distance,
            'ear_quick_gap': ear_quick_gap,
            'ear_between_unit_gap': ear_between_unit_gap,
            'ear_min_crisscrosses': ear_min_crisscrosses
        }
        self._calculate_peaks()
        self._calculate_head_cycles()  # Calculate cycles and pass/fail status
        self._calculate_oscillation_groups()  # Group cycles/crisscrosses for numbering
        if self.show_peaks:
            self._draw_peaks()
        if self.show_cycle_lines:
            self._draw_cycle_lines()
        if self.show_oscillation_numbers:
            self._draw_oscillation_numbers()
        self.canvas.draw_idle()

    def _calculate_peaks(self):
        """
        Calculate peak and valley indices for all signals using current parameters.

        This matches the actual detector logic in core/detectors.py:
        - EarsDetector uses height threshold and max_gap distance
        - HeadDetector uses prominence and peak_distance
        """
        if self.signals_df is None:
            return

        # Get signal arrays
        left_ear = self.signals_df['left_ear_dist'].values
        right_ear = self.signals_df['right_ear_dist'].values
        head = self.signals_df['head_dist'].values

        # === EAR SIGNALS ===
        # Matches EarsDetector._detect_crisscross_units() logic:
        # If use_prominence_mode: find_peaks(signal, prominence=ear_prominence, distance=max_gap)
        # Otherwise: find_peaks(signal, height=peak_threshold, distance=max_gap)

        ear_distance = self.peak_params['ear_distance']
        ear_use_prominence = self.peak_params['ear_use_prominence_mode']
        ear_prominence = self.peak_params['ear_prominence']
        ear_peak_height = self.peak_params['ear_peak_height']
        ear_valley_height = self.peak_params['ear_valley_height']

        if ear_use_prominence:
            # Prominence-based detection (relative movement)
            # Left ear peaks and valleys
            try:
                self.left_ear_peak_indices, _ = find_peaks(
                    left_ear,
                    prominence=ear_prominence,
                    distance=ear_distance
                )
            except Exception:
                self.left_ear_peak_indices = np.array([])

            try:
                self.left_ear_valley_indices, _ = find_peaks(
                    -left_ear,
                    prominence=ear_prominence,
                    distance=ear_distance
                )
            except Exception:
                self.left_ear_valley_indices = np.array([])

            # Right ear peaks and valleys
            try:
                self.right_ear_peak_indices, _ = find_peaks(
                    right_ear,
                    prominence=ear_prominence,
                    distance=ear_distance
                )
            except Exception:
                self.right_ear_peak_indices = np.array([])

            try:
                self.right_ear_valley_indices, _ = find_peaks(
                    -right_ear,
                    prominence=ear_prominence,
                    distance=ear_distance
                )
            except Exception:
                self.right_ear_valley_indices = np.array([])
        else:
            # Absolute height threshold detection (original method)
            # Left ear peaks and valleys
            try:
                self.left_ear_peak_indices, _ = find_peaks(
                    left_ear,
                    height=ear_peak_height,
                    distance=ear_distance
                )
            except Exception:
                self.left_ear_peak_indices = np.array([])

            try:
                self.left_ear_valley_indices, _ = find_peaks(
                    -left_ear,
                    height=-ear_valley_height,
                    distance=ear_distance
                )
            except Exception:
                self.left_ear_valley_indices = np.array([])

            # Right ear peaks and valleys
            try:
                self.right_ear_peak_indices, _ = find_peaks(
                    right_ear,
                    height=ear_peak_height,
                    distance=ear_distance
                )
            except Exception:
                self.right_ear_peak_indices = np.array([])

            try:
                self.right_ear_valley_indices, _ = find_peaks(
                    -right_ear,
                    height=-ear_valley_height,
                    distance=ear_distance
                )
            except Exception:
                self.right_ear_valley_indices = np.array([])

        # === HEAD SIGNAL ===
        # Matches HeadDetector logic:
        # 1. Apply Savitzky-Golay smoothing if use_smoothing=True
        # 2. find_peaks(signal, prominence=prom, distance=dist)
        # 3. find_peaks(-signal, prominence=prom, distance=dist)

        head_prominence = self.peak_params['head_prominence']
        head_distance = self.peak_params['head_distance']
        use_smoothing = self.peak_params['head_use_smoothing']
        smoothing_window = self.peak_params['head_smoothing_window']
        smoothing_polyorder = self.peak_params['head_smoothing_polyorder']

        # Apply smoothing if enabled (matches HeadDetector._apply_smoothing)
        if use_smoothing and smoothing_window > smoothing_polyorder and len(head) > smoothing_window:
            try:
                self.head_signal_smoothed = savgol_filter(
                    head,
                    window_length=smoothing_window,
                    polyorder=smoothing_polyorder
                )
            except Exception:
                self.head_signal_smoothed = head.copy()
        else:
            self.head_signal_smoothed = head.copy()

        # Use smoothed signal for peak detection (matches detector behavior)
        head_for_peaks = self.head_signal_smoothed

        try:
            self.head_peak_indices, _ = find_peaks(
                head_for_peaks,
                prominence=head_prominence,
                distance=head_distance
            )
        except Exception:
            self.head_peak_indices = np.array([])

        try:
            self.head_valley_indices, _ = find_peaks(
                -head_for_peaks,
                prominence=head_prominence,
                distance=head_distance
            )
        except Exception:
            self.head_valley_indices = np.array([])

    def _calculate_head_cycles(self):
        """
        Calculate head cycles from peaks/valleys and determine pass/fail status.

        Matches HeadDetector._build_cycles_from_events() logic:
        - Cycles are formed between consecutive peak-valley pairs
        - Amplitude = abs(signal[peak] - signal[valley])
        - Passes if amplitude >= amplitude_threshold

        Optimized to only calculate for visible range + small buffer.
        """
        self.head_cycles = []

        if self.signals_df is None:
            return

        # Use smoothed signal for amplitude calculation (matches detector behavior)
        if self.head_signal_smoothed is not None:
            head = self.head_signal_smoothed
        else:
            head = self.signals_df['head_dist'].values
        amp_threshold = self.peak_params['head_amplitude_threshold']

        # Add buffer around visible range to ensure we have complete cycles at edges
        buffer = 50  # frames
        calc_start = max(0, self.view_start - buffer)
        calc_end = min(len(head), self.view_end + buffer)

        # Filter peaks/valleys to calculation range
        visible_peaks = self.head_peak_indices[
            (self.head_peak_indices >= calc_start) & (self.head_peak_indices <= calc_end)
        ]
        visible_valleys = self.head_valley_indices[
            (self.head_valley_indices >= calc_start) & (self.head_valley_indices <= calc_end)
        ]

        # Combine peaks and valleys into sorted oscillation events
        # Format: (frame_index, 'peak'/'valley')
        oscillation_events = []
        for idx in visible_peaks:
            oscillation_events.append((idx, 'peak'))
        for idx in visible_valleys:
            oscillation_events.append((idx, 'valley'))

        # Sort by frame index
        oscillation_events.sort(key=lambda x: x[0])

        # Build cycles between consecutive different-type events
        for i in range(len(oscillation_events) - 1):
            idx1, type1 = oscillation_events[i]
            idx2, type2 = oscillation_events[i + 1]

            # Only create cycles between different event types
            if type1 == type2:
                continue

            # Calculate amplitude
            amplitude = abs(head[idx1] - head[idx2])

            # Determine pass/fail
            passes = amplitude >= amp_threshold

            self.head_cycles.append({
                'start_idx': idx1,
                'end_idx': idx2,
                'start_type': type1,
                'end_type': type2,
                'amplitude': amplitude,
                'passes': passes
            })

    def _draw_peaks(self):
        """Draw or update peak and valley scatter plots (only for visible range)."""
        if self.signals_df is None:
            return

        frames = self.signals_df['frame'].values
        left_ear = self.signals_df['left_ear_dist'].values
        right_ear = self.signals_df['right_ear_dist'].values
        head = self.signals_df['head_dist'].values

        # Remove existing scatter plots if any
        self._remove_peak_scatters()

        # Filter to only visible range for performance
        view_start, view_end = self.view_start, self.view_end

        def filter_visible(indices):
            """Filter indices to only those within visible range."""
            if len(indices) == 0:
                return np.array([])
            return indices[(indices >= view_start) & (indices <= view_end)]

        # Marker styles: filled circles for peaks, hollow circles for valleys
        peak_style = {'s': 30, 'zorder': 5, 'marker': 'o', 'edgecolors': 'white', 'linewidths': 0.5}
        valley_style = {'s': 30, 'zorder': 5, 'marker': 'o', 'facecolors': 'none', 'linewidths': 1.5}

        # === LEFT EAR (blue) - filtered to visible range ===
        visible_left_peaks = filter_visible(self.left_ear_peak_indices)
        visible_left_valleys = filter_visible(self.left_ear_valley_indices)

        if len(visible_left_peaks) > 0:
            self.left_ear_peaks_scatter = self.ax.scatter(
                frames[visible_left_peaks],
                left_ear[visible_left_peaks],
                c='#1f77b4', **peak_style
            )
        if len(visible_left_valleys) > 0:
            self.left_ear_valleys_scatter = self.ax.scatter(
                frames[visible_left_valleys],
                left_ear[visible_left_valleys],
                edgecolors='#1f77b4', **valley_style
            )

        # === RIGHT EAR (orange) - filtered to visible range ===
        visible_right_peaks = filter_visible(self.right_ear_peak_indices)
        visible_right_valleys = filter_visible(self.right_ear_valley_indices)

        if len(visible_right_peaks) > 0:
            self.right_ear_peaks_scatter = self.ax.scatter(
                frames[visible_right_peaks],
                right_ear[visible_right_peaks],
                c='#ff7f0e', **peak_style
            )
        if len(visible_right_valleys) > 0:
            self.right_ear_valleys_scatter = self.ax.scatter(
                frames[visible_right_valleys],
                right_ear[visible_right_valleys],
                edgecolors='#ff7f0e', **valley_style
            )

        # === HEAD (green, with pass/fail coloring based on amplitude threshold) ===
        # Filter to visible range first
        visible_head_peaks = filter_visible(self.head_peak_indices)
        visible_head_valleys = filter_visible(self.head_valley_indices)

        # Determine which peaks/valleys are part of passing cycles
        passing_peak_indices = set()
        passing_valley_indices = set()
        for cycle in self.head_cycles:
            if cycle['passes']:
                if cycle['start_type'] == 'peak':
                    passing_peak_indices.add(cycle['start_idx'])
                else:
                    passing_valley_indices.add(cycle['start_idx'])
                if cycle['end_type'] == 'peak':
                    passing_peak_indices.add(cycle['end_idx'])
                else:
                    passing_valley_indices.add(cycle['end_idx'])

        # Split visible peaks into passing and failing
        head_peaks_pass = np.array([i for i in visible_head_peaks if i in passing_peak_indices])
        head_peaks_fail = np.array([i for i in visible_head_peaks if i not in passing_peak_indices])
        head_valleys_pass = np.array([i for i in visible_head_valleys if i in passing_valley_indices])
        head_valleys_fail = np.array([i for i in visible_head_valleys if i not in passing_valley_indices])

        # Draw passing peaks/valleys (green)
        if len(head_peaks_pass) > 0:
            self.head_peaks_scatter = self.ax.scatter(
                frames[head_peaks_pass],
                head[head_peaks_pass],
                c='#2ca02c', **peak_style
            )
        if len(head_valleys_pass) > 0:
            self.head_valleys_scatter = self.ax.scatter(
                frames[head_valleys_pass],
                head[head_valleys_pass],
                edgecolors='#2ca02c', **valley_style
            )

        # Draw failing peaks/valleys (gray)
        failing_peak_style = {'s': 30, 'zorder': 4, 'marker': 'o', 'edgecolors': 'white', 'linewidths': 0.5, 'alpha': 0.5}
        failing_valley_style = {'s': 30, 'zorder': 4, 'marker': 'o', 'facecolors': 'none', 'linewidths': 1.5, 'alpha': 0.5}
        if len(head_peaks_fail) > 0:
            self.head_peaks_failing_scatter = self.ax.scatter(
                frames[head_peaks_fail],
                head[head_peaks_fail],
                c='#888888', **failing_peak_style
            )
        if len(head_valleys_fail) > 0:
            self.head_valleys_failing_scatter = self.ax.scatter(
                frames[head_valleys_fail],
                head[head_valleys_fail],
                edgecolors='#888888', **failing_valley_style
            )

    def _hide_peaks(self):
        """Hide peak scatter plots without removing cached peak data."""
        self._remove_peak_scatters()

    def _remove_peak_scatters(self):
        """Remove existing peak and valley scatter plot objects."""
        for attr in (
            'left_ear_peaks_scatter', 'left_ear_valleys_scatter',
            'right_ear_peaks_scatter', 'right_ear_valleys_scatter',
            'head_peaks_scatter', 'head_valleys_scatter',
            'head_peaks_failing_scatter', 'head_valleys_failing_scatter',
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except ValueError:
                    pass  # Artist already removed
                setattr(self, attr, None)

    # ==================== Cycle Line Methods ====================

    def _on_show_cycles_changed(self, state):
        """Handle show cycle lines checkbox state change."""
        self.show_cycle_lines = bool(state)
        if self.show_cycle_lines:
            self._draw_cycle_lines()
        else:
            self._remove_cycle_lines()
        self.canvas.draw_idle()

    def _draw_cycle_lines(self):
        """Draw connector lines between peak-valley pairs showing cycle amplitudes (visible range only)."""
        if self.signals_df is None or not self.head_cycles:
            return

        # Remove existing cycle lines
        self._remove_cycle_lines()

        frames = self.signals_df['frame'].values
        head = self.signals_df['head_dist'].values
        view_start, view_end = self.view_start, self.view_end

        for cycle in self.head_cycles:
            start_idx = cycle['start_idx']
            end_idx = cycle['end_idx']

            # Skip cycles outside visible range
            if end_idx < view_start or start_idx > view_end:
                continue

            passes = cycle['passes']

            # Get coordinates
            x1, y1 = frames[start_idx], head[start_idx]
            x2, y2 = frames[end_idx], head[end_idx]

            # Draw vertical connector line
            # Use the x-coordinate midpoint for the vertical line
            x_mid = (x1 + x2) / 2
            color = '#2ca02c' if passes else '#888888'
            alpha = 0.8 if passes else 0.4
            linewidth = 2 if passes else 1

            # Draw a vertical line showing amplitude, positioned at midpoint
            line, = self.ax.plot(
                [x_mid, x_mid], [y1, y2],
                color=color, linewidth=linewidth, alpha=alpha,
                linestyle='-', zorder=3
            )
            self.cycle_lines.append(line)

            # Also draw small horizontal ticks at the endpoints to connect to the signal
            tick_width = (x2 - x1) * 0.15  # 15% of cycle width
            tick1, = self.ax.plot(
                [x_mid - tick_width/2, x_mid + tick_width/2], [y1, y1],
                color=color, linewidth=linewidth, alpha=alpha, zorder=3
            )
            tick2, = self.ax.plot(
                [x_mid - tick_width/2, x_mid + tick_width/2], [y2, y2],
                color=color, linewidth=linewidth, alpha=alpha, zorder=3
            )
            self.cycle_lines.append(tick1)
            self.cycle_lines.append(tick2)

    def _remove_cycle_lines(self):
        """Remove all cycle connector lines."""
        for line in self.cycle_lines:
            try:
                line.remove()
            except ValueError:
                pass  # Line already removed
        self.cycle_lines = []

    # ==================== Oscillation Numbering Methods ====================

    def _on_show_osc_numbers_changed(self, state):
        """Handle show oscillation numbers checkbox state change."""
        self.show_oscillation_numbers = bool(state)
        if self.show_oscillation_numbers:
            self._draw_oscillation_numbers()
        else:
            self._remove_oscillation_numbers()
        self.canvas.draw_idle()

    def _calculate_oscillation_groups(self):
        """
        Group cycles into oscillation groups for numbering.

        For head: Groups consecutive passing cycles within max_cycle_gap.
        For ears: Groups crisscross events within between_unit_gap.

        Only processes visible range + buffer for performance.
        """
        self.head_oscillation_groups = []
        self.ear_crisscross_groups = []

        if self.signals_df is None:
            return

        # === HEAD OSCILLATION GROUPS ===
        # Group passing cycles that are within max_cycle_gap of each other
        # Matches HeadDetector._group_cycles_into_headshakes()

        max_cycle_gap = self.peak_params['head_max_cycle_gap']
        min_oscillations = self.peak_params['head_min_oscillations']

        # Filter to passing cycles only (for grouping)
        passing_cycles = [c for c in self.head_cycles if c['passes']]

        if passing_cycles:
            current_group = [passing_cycles[0]]

            for cycle in passing_cycles[1:]:
                prev_cycle = current_group[-1]
                gap = cycle['start_idx'] - prev_cycle['end_idx']

                if gap <= max_cycle_gap:
                    current_group.append(cycle)
                else:
                    # Save current group and start new one
                    if len(current_group) >= 1:  # Save all groups, mark validity later
                        self.head_oscillation_groups.append({
                            'cycles': current_group.copy(),
                            'valid': len(current_group) >= min_oscillations
                        })
                    current_group = [cycle]

            # Don't forget the last group
            if current_group:
                self.head_oscillation_groups.append({
                    'cycles': current_group.copy(),
                    'valid': len(current_group) >= min_oscillations
                })

        # === EAR CRISSCROSS GROUPS ===
        # First detect crisscross events, then group them
        # Matches EarsDetector._detect_crisscross_units() and _group_into_headshakes()

        self._calculate_ear_crisscrosses()

    def _calculate_ear_crisscrosses(self):
        """
        Detect and group ear crisscross events.

        A crisscross is when a left ear peak/valley pairs with an opposite
        right ear valley/peak within quick_gap frames.
        """
        if self.signals_df is None:
            return

        frames = self.signals_df['frame'].values
        left_ear = self.signals_df['left_ear_dist'].values
        right_ear = self.signals_df['right_ear_dist'].values

        quick_gap = self.peak_params['ear_quick_gap']
        between_unit_gap = self.peak_params['ear_between_unit_gap']
        min_crisscrosses = self.peak_params['ear_min_crisscrosses']

        # Buffer around visible range
        buffer = 50
        calc_start = max(0, self.view_start - buffer)
        calc_end = min(len(frames), self.view_end + buffer)

        # Filter peaks/valleys to calculation range
        left_peaks = self.left_ear_peak_indices[
            (self.left_ear_peak_indices >= calc_start) & (self.left_ear_peak_indices <= calc_end)
        ]
        left_valleys = self.left_ear_valley_indices[
            (self.left_ear_valley_indices >= calc_start) & (self.left_ear_valley_indices <= calc_end)
        ]
        right_peaks = self.right_ear_peak_indices[
            (self.right_ear_peak_indices >= calc_start) & (self.right_ear_peak_indices <= calc_end)
        ]
        right_valleys = self.right_ear_valley_indices[
            (self.right_ear_valley_indices >= calc_start) & (self.right_ear_valley_indices <= calc_end)
        ]

        # Combine all events with their types
        # Format: (frame_idx, type, value)
        all_events = []
        for idx in left_peaks:
            all_events.append((idx, 'LP', left_ear[idx]))
        for idx in left_valleys:
            all_events.append((idx, 'LV', left_ear[idx]))
        for idx in right_peaks:
            all_events.append((idx, 'RP', right_ear[idx]))
        for idx in right_valleys:
            all_events.append((idx, 'RV', right_ear[idx]))

        # Sort by frame
        all_events.sort(key=lambda x: x[0])

        # Find crisscross patterns (LP+RV or RP+LV within quick_gap)
        crisscrosses = []
        i = 0
        while i < len(all_events) - 1:
            frame1, type1, val1 = all_events[i]
            frame2, type2, val2 = all_events[i + 1]

            if abs(frame2 - frame1) <= quick_gap:
                if (type1 == 'LP' and type2 == 'RV') or (type1 == 'RV' and type2 == 'LP'):
                    crisscrosses.append({
                        'frame1': frame1,
                        'frame2': frame2,
                        'type': 'LPRV',
                        'events': (all_events[i], all_events[i + 1])
                    })
                    i += 2
                    continue
                elif (type1 == 'RP' and type2 == 'LV') or (type1 == 'LV' and type2 == 'RP'):
                    crisscrosses.append({
                        'frame1': frame1,
                        'frame2': frame2,
                        'type': 'RPLV',
                        'events': (all_events[i], all_events[i + 1])
                    })
                    i += 2
                    continue
            i += 1

        # Group crisscrosses by proximity and alternating type
        if not crisscrosses:
            return

        current_group = [crisscrosses[0]]

        for cc in crisscrosses[1:]:
            prev_cc = current_group[-1]
            gap = min(cc['frame1'], cc['frame2']) - max(prev_cc['frame1'], prev_cc['frame2'])

            # Check if close enough and alternating type
            if gap <= between_unit_gap and cc['type'] != prev_cc['type']:
                current_group.append(cc)
            else:
                # Save current group
                if len(current_group) >= 1:
                    self.ear_crisscross_groups.append({
                        'crisscrosses': current_group.copy(),
                        'valid': len(current_group) >= min_crisscrosses
                    })
                current_group = [cc]

        # Don't forget last group
        if current_group:
            self.ear_crisscross_groups.append({
                'crisscrosses': current_group.copy(),
                'valid': len(current_group) >= min_crisscrosses
            })

    def _draw_oscillation_numbers(self):
        """Draw numbered labels on oscillations within groups (visible range only, 3+ only)."""
        if self.signals_df is None:
            return

        # Remove existing annotations
        self._remove_oscillation_numbers()

        frames = self.signals_df['frame'].values
        head = self.signals_df['head_dist'].values
        view_start, view_end = self.view_start, self.view_end

        # === HEAD OSCILLATION NUMBERS ===
        min_to_show = 3  # Only show numbers for groups with 3+ oscillations

        for group in self.head_oscillation_groups:
            cycles = group['cycles']
            valid = group['valid']

            # Only show for groups with 3+ cycles
            if len(cycles) < min_to_show:
                continue

            for i, cycle in enumerate(cycles):
                start_idx = cycle['start_idx']
                end_idx = cycle['end_idx']

                # Skip if outside visible range
                mid_idx = (start_idx + end_idx) // 2
                if mid_idx < view_start or mid_idx > view_end:
                    continue

                # Position label at the midpoint of the cycle, above the signal
                x = frames[mid_idx]
                y = min(head[start_idx], head[end_idx])  # Top of the cycle (remember Y is inverted)

                # Number within group (1-indexed)
                num = i + 1
                color = '#2ca02c' if valid else '#888888'

                annotation = self.ax.annotate(
                    str(num),
                    xy=(x, y),
                    xytext=(0, -12),  # Offset above (negative because Y inverted)
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='circle,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.9),
                    zorder=10
                )
                self.oscillation_annotations.append(annotation)

        # === EAR CRISSCROSS NUMBERS ===
        left_ear = self.signals_df['left_ear_dist'].values
        right_ear = self.signals_df['right_ear_dist'].values

        for group in self.ear_crisscross_groups:
            crisscrosses = group['crisscrosses']
            valid = group['valid']

            # Only show for groups with 3+ crisscrosses
            if len(crisscrosses) < min_to_show:
                continue

            for i, cc in enumerate(crisscrosses):
                mid_frame = (cc['frame1'] + cc['frame2']) // 2

                # Skip if outside visible range
                if mid_frame < view_start or mid_frame > view_end:
                    continue

                # Position between the two ear signals
                x = frames[mid_frame]
                # Get ear values at this position
                y_left = left_ear[mid_frame] if mid_frame < len(left_ear) else 0
                y_right = right_ear[mid_frame] if mid_frame < len(right_ear) else 0
                y = (y_left + y_right) / 2  # Middle between the two signals

                # Number within group (1-indexed)
                num = i + 1
                color = '#ff7f0e' if valid else '#888888'  # Orange for ears

                annotation = self.ax.annotate(
                    str(num),
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center', va='center',
                    fontsize=7, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='circle,pad=0.15', facecolor=color,
                              edgecolor='white', alpha=0.9),
                    zorder=10
                )
                self.oscillation_annotations.append(annotation)

    def _remove_oscillation_numbers(self):
        """Remove all oscillation number annotations."""
        for annotation in self.oscillation_annotations:
            try:
                annotation.remove()
            except ValueError:
                pass  # Already removed
        self.oscillation_annotations = []
