# Tune Parameters Tab - Video Inspector Implementation

**Status:** In Progress
**Created:** 2025-10-16
**Last Updated:** 2025-10-16

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Decisions](#architecture-decisions)
3. [User Workflow](#user-workflow)
4. [Component Specifications](#component-specifications)
5. [Signal Calculation Strategy](#signal-calculation-strategy)
6. [Node Mapping Handling](#node-mapping-handling)
7. [Implementation Phases](#implementation-phases)
8. [Legacy Code Analysis](#legacy-code-analysis)
9. [Known Issues & Future Enhancements](#known-issues--future-enhancements)

---

## Overview

### Goal
Create an integrated video inspection interface for the "Tune Parameters" tab that allows users to:
- Load video + H5 tracking data
- Visualize tracking signals and detected events in real-time
- Tune detection parameters interactively
- Validate parameter changes before applying them to full analysis

### Key Requirements
- **Performance**: Calculate signals once, re-run detection only when requested
- **User Control**: No hot-reload on parameter changes; explicit "Reanalyze" actions
- **Visual Feedback**: Side-by-side video and graph with synchronized frame cursor
- **Node Mapping**: User must configure node indices before signal calculation
- **Two Analysis Modes**:
  1. Current view (fast, for tuning)
  2. Entire video (slower, for validation)

---

## Architecture Decisions

### Component Structure
```
gui_v2/
├── video_inspector_widget.py       [NEW] Video display + navigation + H5 loading
├── diagnostics_graph_widget.py     [NEW] Signal plotting + event visualization
├── parameter_panel.py              [MODIFY] Add reanalysis buttons
└── main_window.py                  [MODIFY] Integration + layout
```

### Why This Architecture?
- **Separation of Concerns**: Video logic, graph logic, and parameter logic are independent
- **Reusability**: VideoInspectorWidget could be used elsewhere if needed
- **Maintainability**: Each component has clear responsibilities
- **Testability**: Can test each widget independently

### Data Flow
```
User Action          Component              Signal/Method              Result
────────────────────────────────────────────────────────────────────────────
Load H5          →   VideoInspector    →   Prompt node mapping    →   Show node selector
Set Nodes        →   VideoInspector    →   Calculate signals      →   signals_calculated(df)
                 →   GraphWidget       ←   Receive signals        →   Plot base signals

Load Video       →   VideoInspector    →   Open VideoCapture      →   Display first frame

Navigate Frame   →   VideoInspector    →   frame_changed(num)     →   Update cursor on graph

Change Param     →   ParameterPanel    →   (No immediate action)  →   Parameters stored

Click "Reanalyze →   ParameterPanel    →   reanalyze_view()       →   Run detection on visible range
Current View"    →   GraphWidget       ←   Update event overlays  →   Show colored spans

Click "Reanalyze →   ParameterPanel    →   reanalyze_full()       →   Run detection on entire video
Full Video"      →   GraphWidget       ←   Update event overlays  →   Show colored spans
```

---

## User Workflow

### Initial Setup
1. User opens "Tune Parameters" tab
2. Left side shows placeholder: "Load H5 and Video to begin"
3. Right side shows empty graph canvas above, parameter panel below

### Loading Data
1. User clicks **"Load H5 File..."**
2. System reads `node_names` from H5 file
3. **Node Mapping Dialog** appears:
   ```
   ┌─────────────────────────────────────────┐
   │  Configure Node Mapping                 │
   ├─────────────────────────────────────────┤
   │  Detected nodes in H5 file:             │
   │    0: left-ear                          │
   │    1: right-ear                         │
   │    2: back                              │
   │    3: nose                              │
   │    4: head                              │
   │                                         │
   │  Map to detection indices:              │
   │  Left Ear:    [0 ▼]                    │
   │  Right Ear:   [1 ▼]                    │
   │  Back:        [2 ▼]                    │
   │  Nose:        [3 ▼]                    │
   │  Head:        [4 ▼]                    │
   │                                         │
   │  [Use Defaults]  [Cancel]  [Confirm]   │
   └─────────────────────────────────────────┘
   ```
4. User configures mapping (or uses defaults)
5. System calculates **ear distances** and **head signal** (one-time heavy computation)
6. Graph displays three signal lines:
   - Left ear distance (blue)
   - Right ear distance (orange)
   - Head-to-midline distance (green)

7. User clicks **"Load Video..."**
8. Video appears in left panel, synced to frame 0

### Parameter Tuning Workflow
1. User navigates to a frame with a known headtwitch event (using slider, arrows, or jump-to-frame)
2. User notes the approximate frame range (e.g., frames 500-550)
3. User adjusts parameters in the bottom-right panel
4. User clicks **"🔍 Reanalyze Current View"**
5. System re-runs detection ONLY on the visible x-axis range in the graph
6. Event overlays update immediately:
   - Green span = High confidence (detected by both methods)
   - Orange span = Medium confidence (ears only)
   - Red span = Low confidence (head only)
7. User sees if the event is detected correctly
8. If not, user tweaks parameters and clicks "Reanalyze Current View" again
9. Repeat until satisfied

### Validation Workflow
1. Once parameters look good for one event, user clicks **"📊 Reanalyze Entire Video"**
2. Progress bar shows analysis progress
3. All events across the full video are displayed on graph
4. User can scrub through video to spot-check other events
5. If issues found, user zooms to that frame range and repeats tuning workflow
6. When satisfied, user clicks **"💾 Save Parameters"** to save to project

### Export & Next Steps
1. User can export detected events to CSV
2. User can export GIF of video + graph for specific frame range
3. User proceeds to "Prepare Data" tab to continue workflow

---

## Component Specifications

### 1. VideoInspectorWidget (`video_inspector_widget.py`)

**Purpose:** Handle video display, H5 loading, node mapping, and signal calculation

**UI Layout:**
```
┌───────────────────────────────────────┐
│  [Load H5 File...] [Load Video...]   │
├───────────────────────────────────────┤
│                                       │
│                                       │
│         VIDEO DISPLAY AREA            │
│           (QLabel)                    │
│                                       │
│                                       │
├───────────────────────────────────────┤
│  [◀◀] [◀] ████████████░░ [▶] [▶▶]   │
│  Frame: 0 / 0    [Go to: ___] [Go]   │
└───────────────────────────────────────┘
```

**Key Attributes:**
```python
self.h5_path: str = None
self.video_capture: cv2.VideoCapture = None
self.current_frame: int = 0
self.total_frames: int = 0
self.fps: float = 30.0

# Signal data (calculated once on H5 load)
self.signals_df: pd.DataFrame = None  # Columns: frame, left_ear_dist, right_ear_dist, head_dist
self.tracking_data: np.ndarray = None  # Raw SLEAP tracks
self.point_scores: np.ndarray = None   # Confidence scores

# Node mapping (set by user during H5 load)
self.node_mapping: dict = {
    'left_ear': 0,
    'right_ear': 1,
    'back': 2,
    'nose': 3,
    'head': 4
}
```

**Key Methods:**
```python
def load_h5_file(self):
    """Open file dialog, read H5, prompt node mapping, calculate signals."""
    # 1. Open file dialog
    # 2. Read node_names from H5
    # 3. Show NodeMappingDialog
    # 4. If confirmed, load tracks and point_scores
    # 5. Calculate ear distances and head signal
    # 6. Emit signals_calculated(signals_df)

def load_video_file(self):
    """Open file dialog and load video with OpenCV."""
    # 1. Open file dialog
    # 2. Create cv2.VideoCapture
    # 3. Read total frames and FPS
    # 4. Update slider range
    # 5. Display first frame
    # 6. Emit video_loaded()

def calculate_signals(self):
    """Calculate ear distances and head-to-midline distance."""
    # Uses helper functions from legacy code:
    # - _line_intersection()
    # - point_line_distance()
    # Returns DataFrame with columns: frame, left_ear_dist, right_ear_dist, head_dist

def update_video_frame(self, frame_num: int):
    """Seek to frame and display in QLabel."""

def navigate_frames(self, delta: int):
    """Move forward/backward by delta frames."""

def jump_to_frame(self, frame_num: int):
    """Jump directly to specified frame."""
```

**Signals Emitted:**
```python
signals_calculated = Signal(pd.DataFrame)  # Emitted after H5 signal calculation
video_loaded = Signal()                    # Emitted after video loads
frame_changed = Signal(int)                # Emitted when current frame changes
view_range_changed = Signal(int, int)      # Emitted when graph view changes (for reanalysis scope)
```

---

### 2. DiagnosticsGraphWidget (`diagnostics_graph_widget.py`)

**Purpose:** Display signal plots and event detection overlays

**UI Layout:**
```
┌───────────────────────────────────────┐
│                                       │
│  [Matplotlib Canvas]                  │
│   • Blue line: Left ear distance      │
│   • Orange line: Right ear distance   │
│   • Green dashed: Head signal         │
│   • Red vertical: Current frame       │
│   • Colored spans: Detected events    │
│                                       │
└───────────────────────────────────────┘
```

**Key Attributes:**
```python
self.signals_df: pd.DataFrame = None     # Base signal data (set once)
self.current_frame: int = 0              # Current frame cursor position
self.view_start: int = 0                 # X-axis view start
self.view_end: int = 100                 # X-axis view end
self.event_spans: list = []              # List of matplotlib span patches
```

**Key Methods:**
```python
def set_signals(self, signals_df: pd.DataFrame):
    """Set base signal data and plot initial lines (done once)."""
    # 1. Store signals_df
    # 2. Clear axes
    # 3. Plot three lines (left ear, right ear, head)
    # 4. Add legend
    # 5. Set y-axis limits
    # 6. Add frame cursor line (initially at frame 0)
    # 7. Draw canvas

def update_frame_cursor(self, frame_num: int):
    """Move red vertical line to show current frame."""
    # 1. Update frame cursor line x-position
    # 2. Redraw canvas (fast, no full replot)

def update_events(self, events_df: pd.DataFrame):
    """Update event detection overlays."""
    # 1. Remove old span patches
    # 2. Iterate through events_df
    # 3. For each event, add axvspan with color based on confidence:
    #    - High (both detectors): green, alpha=0.3
    #    - Medium (ears only): orange, alpha=0.3
    #    - Low (head only): red, alpha=0.3
    # 4. Redraw canvas

def set_view_range(self, start: int, end: int):
    """Set x-axis view range (for zoom and pan)."""
    # 1. Update view_start and view_end
    # 2. Set axes xlim
    # 3. Redraw canvas

def clear_events(self):
    """Remove all event overlays."""
```

**Signals Emitted:**
```python
view_range_changed = Signal(int, int)  # User panned/zoomed graph
```

---

### 3. ParameterPanel Modifications (`parameter_panel.py`)

**New UI Elements to Add:**

Add after the "Parameter Management" section (around line 135):

```python
# ===== ANALYSIS CONTROL SECTION =====
analysis_group = QGroupBox("Detection Analysis")
analysis_group.setMaximumHeight(100)
analysis_layout = QVBoxLayout(analysis_group)

# Info label
info_label = QLabel("Adjust parameters above, then reanalyze to see results.")
info_label.setFont(QFont("Arial", 8))
info_label.setStyleSheet("color: #666;")
analysis_layout.addWidget(info_label)

# Buttons row
button_row = QHBoxLayout()

self.reanalyze_view_btn = QPushButton("🔍 Reanalyze Current View")
self.reanalyze_view_btn.setFont(QFont("Arial", 9, QFont.Bold))
self.reanalyze_view_btn.setStyleSheet("""
    QPushButton {
        background-color: #007bff;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #0056b3;
    }
""")
self.reanalyze_view_btn.clicked.connect(self.reanalyze_current_view)
button_row.addWidget(self.reanalyze_view_btn)

self.reanalyze_full_btn = QPushButton("📊 Reanalyze Entire Video")
self.reanalyze_full_btn.setFont(QFont("Arial", 9))
self.reanalyze_full_btn.setStyleSheet("""
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
self.reanalyze_full_btn.clicked.connect(self.reanalyze_full_video)
button_row.addWidget(self.reanalyze_full_btn)

analysis_layout.addLayout(button_row)
parent_layout.addWidget(analysis_group)
```

**New Signals:**
```python
reanalyze_view_requested = Signal()      # User wants to reanalyze visible range
reanalyze_full_requested = Signal()      # User wants to reanalyze entire video
```

**New Methods:**
```python
def reanalyze_current_view(self):
    """Emit signal to reanalyze the current visible graph range."""
    self.apply_parameters_to_config()  # Ensure config is up-to-date
    self.reanalyze_view_requested.emit()

def reanalyze_full_video(self):
    """Emit signal to reanalyze the entire video."""
    self.apply_parameters_to_config()  # Ensure config is up-to-date
    self.reanalyze_full_requested.emit()
```

---

### 4. MainWindow Integration (`main_window.py`)

**Modified `create_tune_parameters_tab()` method:**

```python
def create_tune_parameters_tab(self):
    """Tab 2: Tune Parameters with video feedback."""
    param_widget = QWidget()
    param_layout = QHBoxLayout(param_widget)
    param_layout.setContentsMargins(5, 5, 5, 5)

    # LEFT SIDE: Video Inspector (690px)
    self.video_inspector = VideoInspectorWidget(parent=self)
    self.video_inspector.setMaximumWidth(690)
    param_layout.addWidget(self.video_inspector)

    # RIGHT SIDE: Graph + Parameter Panel (690px)
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(5)

    # Top: Diagnostics Graph (~350px)
    self.diagnostics_graph = DiagnosticsGraphWidget(parent=self)
    self.diagnostics_graph.setMaximumHeight(350)
    right_layout.addWidget(self.diagnostics_graph)

    # Bottom: Parameter Panel (scrollable, remaining space)
    self.parameter_panel = ParameterPanel(parent=self, project_manager=self.project_manager)
    right_layout.addWidget(self.parameter_panel)

    right_widget.setMaximumWidth(690)
    param_layout.addWidget(right_widget)

    # WIRE UP SIGNALS
    # H5 signals calculated -> plot on graph
    self.video_inspector.signals_calculated.connect(self.diagnostics_graph.set_signals)

    # Frame changed -> update cursor
    self.video_inspector.frame_changed.connect(self.diagnostics_graph.update_frame_cursor)

    # Reanalyze requests -> run detection
    self.parameter_panel.reanalyze_view_requested.connect(self.reanalyze_current_view)
    self.parameter_panel.reanalyze_full_requested.connect(self.reanalyze_full_video)

    self.tab_widget.addTab(param_widget, "Tune Parameters")

def reanalyze_current_view(self):
    """Run detection on the visible graph range."""
    if not self.video_inspector.signals_df:
        QMessageBox.warning(self, "No Data", "Please load an H5 file first.")
        return

    # Get visible range from graph
    start, end = self.diagnostics_graph.view_start, self.diagnostics_graph.view_end

    # Run detection on this range using current parameters
    events_df = self.run_detection_on_range(start, end)

    # Update graph overlays
    self.diagnostics_graph.update_events(events_df)

def reanalyze_full_video(self):
    """Run detection on entire video."""
    if not self.video_inspector.signals_df:
        QMessageBox.warning(self, "No Data", "Please load an H5 file first.")
        return

    # Show progress dialog
    progress = QProgressDialog("Analyzing entire video...", "Cancel", 0, 100, self)
    progress.setWindowModality(Qt.WindowModal)
    progress.show()

    # Run detection on full range
    total_frames = len(self.video_inspector.signals_df)
    events_df = self.run_detection_on_range(0, total_frames)

    progress.close()

    # Update graph overlays
    self.diagnostics_graph.update_events(events_df)

    QMessageBox.information(self, "Complete", f"Found {len(events_df)} events.")

def run_detection_on_range(self, start_frame: int, end_frame: int) -> pd.DataFrame:
    """Run HTR detection on specified frame range using current parameters."""
    # This is where we integrate with core detection classes
    # Import inside method to avoid circular imports
    from core.data_processing import EarsHeadshakeDetector, HeadshakeDetector

    # Get current parameters from config manager
    config = self.config_manager.config

    # Build config dicts for detectors
    ear_cfg = {
        'fps': config.default_fps,
        'peak_threshold': config.ear_detector.peak_threshold,
        'valley_threshold': config.ear_detector.valley_threshold,
        'max_gap': config.ear_detector.max_gap,
        'quick_gap': config.ear_detector.quick_gap,
        'min_crisscrosses': config.ear_detector.min_crisscrosses,
        'between_unit_gap': config.ear_detector.between_unit_gap,
        'merge_gap': config.ear_detector.merge_gap,
        'apply_median_score_filter': config.ear_detector.apply_median_score_filter,
        'median_score_threshold': config.ear_detector.median_score_threshold,
    }

    head_cfg = {
        'interpolation_method': config.head_detector.interpolation_method,
        'min_oscillations': config.head_detector.min_oscillations,
        'amplitude_threshold': config.head_detector.amplitude_threshold,
        'amplitude_median': config.head_detector.amplitude_median,
        'median_score_threshold': config.head_detector.median_score_threshold,
        'peak_prominence': config.head_detector.peak_prominence,
        'peak_distance': config.head_detector.peak_distance,
        'use_smoothing': config.head_detector.use_smoothing,
        'smoothing_window': config.head_detector.smoothing_window,
        'smoothing_polyorder': config.head_detector.smoothing_polyorder,
        'min_cycle_duration': config.head_detector.min_cycle_duration,
        'max_cycle_duration': config.head_detector.max_cycle_duration,
        'max_cycle_gap': config.head_detector.max_cycle_gap,
    }

    # Run detection using CombinedDetector
    # NOTE: We need to adapt this to work with pre-calculated signals
    # For now, we'll use the H5 file path and let detectors recalculate
    # TODO: Optimize to use pre-calculated signals from video_inspector

    h5_path = self.video_inspector.h5_path
    combiner = CombinedDetector(h5_path, ear_cfg, head_cfg, start_frame, end_frame)
    events_df, _, _ = combiner.run(iou_threshold=config.iou_threshold)

    return events_df
```

---

## Signal Calculation Strategy

### One-Time Calculation (Heavy)
When H5 is loaded, calculate these signals **once**:

1. **Ear Distances**
   - For each frame:
     - Find intersection of (back-nose) line and (left_ear-right_ear) line
     - Calculate distance from left_ear to intersection
     - Calculate distance from right_ear to intersection
   - Interpolate NaN values
   - Store in DataFrame

2. **Head Signal**
   - For each frame:
     - Calculate perpendicular distance from head point to (back-nose) line
   - Interpolate NaN values
   - Apply smoothing if configured
   - Store in DataFrame

**Result:** DataFrame with columns `[frame, left_ear_dist, right_ear_dist, head_dist]`

### Re-Analysis (Fast)
When user clicks "Reanalyze":

1. **Extract frame range** from graph view or full video
2. **Slice pre-calculated signals** for that range
3. **Run peak detection** on sliced signals:
   - Find peaks and valleys in ear distance signals
   - Find oscillations in head signal
4. **Group into events** using timing/gap parameters
5. **Filter events** using confidence thresholds
6. **Return events DataFrame** with columns `[start_frame, end_frame, confidence, method]`

**Performance:**
- Full signal calculation: ~1-2 seconds for 10,000 frames
- Re-analysis on 100-frame window: ~50ms
- Re-analysis on full 10,000 frames: ~200-300ms

This is why we separate the two operations!

---

## Node Mapping Handling

### Node Mapping Dialog (`node_mapping_dialog.py`)

**New file to create:**

```python
class NodeMappingDialog(QDialog):
    """Dialog for configuring SLEAP node indices before signal calculation."""

    def __init__(self, node_names: list, default_mapping: dict, parent=None):
        """
        Args:
            node_names: List of node names from H5 file (e.g., ['left-ear', 'right-ear', ...])
            default_mapping: Default node indices from config
        """
        super().__init__(parent)
        self.node_names = node_names
        self.result_mapping = default_mapping.copy()
        self.init_ui()

    def init_ui(self):
        # Layout with:
        # 1. Info label explaining purpose
        # 2. Display of detected nodes (read-only list)
        # 3. Five ComboBoxes for mapping (left_ear, right_ear, back, nose, head)
        # 4. Use Defaults / Cancel / Confirm buttons

    def get_mapping(self) -> dict:
        """Return the configured node mapping."""
        return self.result_mapping
```

**When to Show:**
- Immediately after user selects H5 file
- Before any signal calculation
- Modal dialog (blocks until user confirms or cancels)

**Default Behavior:**
- Try to auto-match node names (e.g., if node_names contains "left-ear", auto-select that index for left_ear)
- Fall back to config defaults if no match found
- Show "Use Defaults" button to quickly accept suggested mapping

**Validation:**
- Ensure all five nodes are mapped
- Warn if same index used for multiple nodes
- Don't allow confirmation if mapping is invalid

---

## Implementation Phases

### Phase 1: Core Video + Graph (PRIORITY)
**Goal:** Get basic video display and signal plotting working

**Tasks:**
1. ✅ Create implementation document (this file)
2. ⬜ Create `NodeMappingDialog` class
3. ⬜ Create `VideoInspectorWidget` class
   - File loading buttons
   - Video display (QLabel + QPixmap)
   - Frame navigation controls
   - H5 signal calculation
4. ⬜ Create `DiagnosticsGraphWidget` class
   - Matplotlib canvas setup
   - Signal plotting (three lines)
   - Frame cursor
5. ⬜ Integrate into `main_window.py`
   - Update layout
   - Wire up signals
6. ⬜ Test: Load H5 + video, navigate frames, see signals

### Phase 2: Parameter Integration
**Goal:** Make reanalysis buttons work

**Tasks:**
1. ⬜ Modify `ParameterPanel` to add analysis buttons
2. ⬜ Implement `run_detection_on_range()` in main_window
3. ⬜ Connect reanalysis signals to graph updates
4. ⬜ Test: Change parameters, reanalyze view, see event overlays

### Phase 3: Polish & Features
**Goal:** Add convenience features

**Tasks:**
1. ⬜ Implement "Jump to Next Event" navigation
2. ⬜ Add bookmark system for marking interesting frames
3. ⬜ Export events to CSV
4. ⬜ Export GIF (video + graph combined)
5. ⬜ Progress indicators for full video analysis
6. ⬜ Keyboard shortcuts (space to play/pause, arrow keys for navigation, etc.)

### Phase 4: Optimization (if needed)
**Goal:** Improve performance for large files

**Tasks:**
1. ⬜ Profile signal calculation performance
2. ⬜ Consider multiprocessing for full video reanalysis
3. ⬜ Implement lazy loading for very large H5 files (>100K frames)
4. ⬜ Cache detection results to avoid redundant computation

---

## Legacy Code Analysis

### Source Files Reviewed
Located in `C:\Users\grays\Dropbox\HDAC\legacy_components\`:

1. **inspect_video.py** (719 lines)
   - Full standalone app with video + matplotlib
   - PoseData class for H5 loading
   - DiagnosticsCanvas for plotting
   - Bookmarks, CSV export
   - **Reusable:** Helper functions, overlay drawing logic

2. **gemini_inspect.py** (652 lines)
   - Basic video + graph side-by-side
   - Dynamic graph window
   - Detection pipeline integration
   - **Reusable:** Layout approach, graph update patterns

3. **gemini_inspect2.py** (718 lines)
   - Adds GIF export to gemini_inspect
   - Jump-to-frame on graph click
   - **Reusable:** GIF export logic

4. **gemini_inspect_gif.py** (830 lines)
   - Combined video+graph GIF export
   - **Reusable:** Combined GIF rendering (lines 646-720)

5. **gemini3_inspect_parameters.py** (836 lines) ⭐ **MOST RELEVANT**
   - Parameter dock with real-time controls
   - Smart performance: one-time signal calculation + fast reanalysis
   - Separation of signal calculation from event detection
   - **Reusable:** Entire architecture pattern, performance strategy
   - **Issue:** Node indices inconsistent (0,1,2,3,4 vs other files using 2,3,4,0,1)

### Key Helper Functions to Reuse

```python
def _line_intersection(p1, p2, q1, q2):
    """Calculate intersection of two lines."""
    # From inspect_video.py lines 28-34

def point_line_distance(point, line_start, line_end):
    """Calculate perpendicular distance from point to line."""
    # From inspect_video.py lines 36-42
```

### Detection Classes Structure

All legacy files use similar detection pipeline:
1. `EarsHeadshakeDetector` - Analyzes ear movement crisscross patterns
2. `HeadshakeDetector` - Analyzes head-to-midline oscillations
3. `CombinedDetector` - Combines both methods with IoU matching

**Key Insight from gemini3_inspect_parameters.py:**
- Signal calculation (expensive) is done once in `load_h5()`
- Detection algorithms (cheap) run on pre-calculated signals
- This is the pattern we'll follow!

---

## Known Issues & Future Enhancements

### Known Limitations
1. **H5 Shape Assumptions:** Code assumes SLEAP H5 format with specific structure
2. **Memory Usage:** Loads entire H5 into memory (not suitable for >1M frame files)
3. **Video Codec Support:** Limited by OpenCV's codec support on Windows
4. **Single Animal:** No multi-animal tracking support yet

### Future Enhancement Ideas
1. **Multi-Animal Support:** Dropdown to select which animal track to visualize
2. **Overlay Tracking Points:** Draw keypoints directly on video frame (like inspect_video.py)
3. **Confidence Heatmap:** Show point_scores as overlay on video
4. **Parameter Presets:** Save/load named parameter sets (e.g., "Aggressive Detection", "Conservative")
5. **Batch Parameter Testing:** Test multiple parameter combinations and compare results
6. **Video Trimming:** Export trimmed video clips around detected events
7. **Event Annotations:** Click on event spans to add notes/labels
8. **Real-time Parameter Hints:** Show parameter impact predictions as user edits values

### Code Quality TODOs
1. Add comprehensive docstrings to all methods
2. Add type hints throughout
3. Write unit tests for signal calculation functions
4. Add integration tests for full workflow
5. Profile performance and optimize bottlenecks
6. Add logging for debugging
7. Handle edge cases (empty H5, corrupted video, etc.)

---

## Quick Reference

### File Locations
- Implementation doc: `gui_v2/TUNE_PARAMETERS_IMPLEMENTATION.md` (this file)
- Main window: `gui_v2/main_window.py`
- Parameter panel: `gui_v2/parameter_panel.py`
- Video inspector: `gui_v2/video_inspector_widget.py` (to be created)
- Graph widget: `gui_v2/diagnostics_graph_widget.py` (to be created)
- Node mapping dialog: `gui_v2/node_mapping_dialog.py` (to be created)
- Legacy reference: `legacy_components/gemini3_inspect_parameters.py`

### Key Dependencies
- PySide6 (GUI framework)
- OpenCV (video loading)
- matplotlib (plotting)
- pandas (signal data)
- numpy (numerical operations)
- h5py (H5 file reading)

### Testing Checklist
- [ ] Load H5 file → Node mapping dialog appears
- [ ] Configure nodes → Signals calculate and plot
- [ ] Load video → Frame displays
- [ ] Navigate frames → Video and cursor update
- [ ] Change parameters → No automatic action
- [ ] Click "Reanalyze View" → Events appear on graph
- [ ] Click "Reanalyze Full" → Progress shown, all events detected
- [ ] Save parameters → File saved to project
- [ ] Export GIF → Combined video+graph saved
- [ ] Close and reopen tab → State preserved

---

**End of Implementation Document**
