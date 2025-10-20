# Tune Parameters Tab - Implementation Complete! 🎉

**Date:** 2025-10-16
**Status:** ✅ Ready for Testing

---

## What Was Implemented

### New Files Created

1. **`detection_utils.py`** - Helper functions for signal calculation
   - `line_intersection()` - Calculate line intersections
   - `point_line_distance()` - Distance from point to line
   - `calculate_ear_distances()` - Ear-based signals
   - `calculate_head_signal()` - Head-based signals
   - `normalize_sleap_tracks()` - Handle various H5 formats
   - `normalize_sleap_scores()` - Normalize confidence scores

2. **`node_mapping_dialog.py`** - Node configuration dialog
   - Shows detected nodes from H5 file
   - Auto-detects common naming patterns
   - Validates against duplicate mappings
   - Warns user if same node used for multiple landmarks

3. **`video_inspector_widget.py`** - Video display and navigation
   - Load H5 and video files independently
   - Frame-by-frame navigation with slider
   - Jump to specific frames
   - One-time signal calculation on H5 load
   - Synchronized video/graph cursor

4. **`diagnostics_graph_widget.py`** - Signal visualization
   - Matplotlib integration
   - Three signal lines (left ear, right ear, head)
   - Event overlay spans with color coding
   - Auto-pan to follow frame cursor
   - Zoom in/out/full functionality

### Modified Files

5. **`parameter_panel.py`** - Added analysis controls
   - "🔍 Reanalyze Current View" button (blue)
   - "📊 Reanalyze Entire Video" button (green)
   - Buttons disabled until H5 loaded
   - New signals: `reanalyze_view_requested`, `reanalyze_full_requested`

6. **`main_window.py`** - Integrated all components
   - New layout: Video (left) | Graph + Parameters (right)
   - Signal wiring for H5→Graph→Cursor synchronization
   - Detection methods: `run_detection_on_range()`
   - Progress dialogs for full video analysis

---

## How to Test

### Step 1: Launch the Application

```bash
cd C:\Users\grays\Dropbox\HDAC\gui_v2
python main_window.py
```

### Step 2: Navigate to "Tune Parameters" Tab

Click on the **"Tune Parameters"** tab (second tab).

You should see:
- **Left side**: "Load H5 and Video files to begin" placeholder
- **Right side**: Empty graph above, parameter panel below

### Step 3: Load H5 File

1. Click **"📁 Load H5 File..."**
2. Select a SLEAP tracking H5 file
3. **Node Mapping Dialog** appears:
   - Review detected nodes
   - Confirm or adjust mapping
   - Click **"Confirm"**
4. System calculates signals (may take 1-2 seconds)
5. Graph shows three colored lines:
   - Blue: Left ear distance
   - Orange: Right ear distance
   - Green: Head-to-midline distance
6. Analysis buttons become enabled (blue and green)

### Step 4: Load Video File

1. Click **"🎬 Load Video..."**
2. Select corresponding video file
3. First frame appears in video display
4. Navigation controls become active

### Step 5: Navigate Frames

- Use **slider** to scrub through frames
- Use **◀◀ 5** / **◀ 1** / **1 ▶** / **5 ▶▶** buttons
- Type frame number in **"Go to:"** box and press Enter
- Watch **red vertical line** on graph follow current frame

### Step 6: Tune Parameters

1. Navigate to a frame range with a known head-twitch (e.g., frames 500-550)
2. Scroll down in parameter panel
3. Adjust parameters (e.g., change "Peak Thresh" from 30 to 25)
4. **Nothing happens yet** - this is correct! (No hot reload)

### Step 7: Reanalyze Current View

1. Click **"🔍 Reanalyze Current View"** (blue button)
2. System runs detection on visible frame range (~0.05 seconds)
3. Colored spans appear on graph:
   - **Green** = High confidence (both detectors)
   - **Orange** = Medium confidence (ears only)
   - **Red** = Low confidence (head only)
4. Check if your known event is detected
5. If not, adjust parameters and reanalyze again

### Step 8: Validate Across Full Video

1. Once parameters look good, click **"📊 Reanalyze Entire Video"** (green button)
2. Progress dialog appears
3. All events detected across full video
4. Scrub through video to spot-check other events

### Step 9: Save Parameters

1. Click **"💾 Save Parameters..."** in parameter panel
2. Save to your project's `parameters/` folder
3. Parameters are now ready to use in batch processing

---

## Expected Behavior

### On H5 Load
- ✅ Node mapping dialog appears
- ✅ Signals calculate and plot immediately
- ✅ Three colored lines visible on graph
- ✅ Analysis buttons become enabled
- ✅ Message box confirms: "Successfully loaded H5 file..."

### On Video Load
- ✅ First frame displays in video area
- ✅ Navigation controls become active
- ✅ Frame counter shows: "Frame: 0 / {total}"
- ✅ Message box confirms: "Successfully loaded video..."

### On Frame Navigation
- ✅ Video updates to show current frame
- ✅ Red vertical line on graph follows cursor
- ✅ Graph auto-pans if cursor moves outside visible range
- ✅ Frame counter updates

### On Reanalyze Current View
- ✅ Takes ~50ms for 100-frame window
- ✅ Event spans appear on graph
- ✅ Status label updates with event count
- ✅ No progress dialog (too fast to need one)

### On Reanalyze Full Video
- ✅ Progress dialog appears
- ✅ Takes ~200-300ms for 10,000 frames
- ✅ All events displayed on graph
- ✅ Message box shows total event count
- ✅ Status label updates

---

## Troubleshooting

### "Config manager not available"
- **Cause**: Core modules not properly imported
- **Fix**: Ensure you're running from project root with `htr_env` activated

### "Could not import detection modules"
- **Cause**: `core.data_processing` not in path
- **Fix**: Check that `core/` directory exists with detection classes

### "Cannot normalize tracks array"
- **Cause**: H5 file has unexpected shape
- **Fix**: Check H5 file format - should be SLEAP export
- **Debug**: Print `tracks_array.shape` to see actual dimensions

### Node mapping dialog shows "Node 0", "Node 1"...
- **Cause**: H5 file missing `node_names` dataset
- **Fix**: Use indices corresponding to your SLEAP skeleton
- **Note**: This is okay - just requires manual mapping

### Video and H5 frame counts don't match
- **Cause**: Video trimmed or different length than tracking
- **Fix**: System uses minimum of both - no error, but may clip data

### Parameters don't seem to affect detection
- **Cause**: Forgot to click "Reanalyze" button
- **Fix**: Parameters only apply when you explicitly reanalyze

### Graph shows no events after reanalysis
- **Possible causes**:
  1. Parameters too strict (no events pass filters)
  2. No actual head-twitches in visible range
  3. Node mapping incorrect
- **Fix**: Try loosening parameters or checking different frame ranges

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Tune Parameters Tab                      │
├────────────────────────────┬────────────────────────────────┤
│                            │                                │
│   VideoInspectorWidget     │   DiagnosticsGraphWidget      │
│   ┌──────────────────┐     │   ┌──────────────────────┐    │
│   │ Load H5/Video    │     │   │  Signal Lines        │    │
│   ├──────────────────┤     │   │  • Left Ear (blue)   │    │
│   │                  │     │   │  • Right Ear (orange)│    │
│   │  Video Display   │     │   │  • Head (green)      │    │
│   │                  │     │   │  Event Overlays      │    │
│   │                  │     │   │  Frame Cursor (red)  │    │
│   ├──────────────────┤     │   └──────────────────────┘    │
│   │  Navigation      │     ├────────────────────────────────┤
│   │  ◀◀ ◀ ▶ ▶▶      │     │   ParameterPanel              │
│   └──────────────────┘     │   ┌──────────────────────┐    │
│                            │   │ Project Management   │    │
│                            │   ├──────────────────────┤    │
│                            │   │ Parameter Mgmt       │    │
│                            │   ├──────────────────────┤    │
│                            │   │ 🔍 Reanalyze View    │    │
│                            │   │ 📊 Reanalyze Full    │    │
│                            │   ├──────────────────────┤    │
│                            │   │ Node Mapping         │    │
│                            │   │ General Settings     │    │
│                            │   ├──────────────────────┤    │
│                            │   │ Ear Detector Params  │    │
│                            │   │ Head Detector Params │    │
│                            │   └──────────────────────┘    │
└────────────────────────────┴────────────────────────────────┘
```

### Signal Flow

```
User Loads H5
    ↓
NodeMappingDialog (configure nodes)
    ↓
VideoInspector calculates signals (one-time heavy)
    ↓
DiagnosticsGraph plots signal lines
    ↓
User adjusts parameters (no immediate action)
    ↓
User clicks "Reanalyze" button
    ↓
MainWindow.run_detection_on_range()
    ↓
CombinedDetector runs on specified range
    ↓
DiagnosticsGraph updates event overlays
```

---

## Performance Benchmarks

Based on gemini3_inspect_parameters.py testing:

| Operation | Frames | Time | Notes |
|-----------|--------|------|-------|
| Load H5 + Calculate Signals | 10,000 | ~1-2 sec | One-time only |
| Reanalyze Current View | 100 | ~50ms | Very fast |
| Reanalyze Full Video | 10,000 | ~200-300ms | Acceptable |
| Frame Navigation | 1 | ~16ms | Smooth |

---

## Next Steps (Future Enhancements)

Phase 3 tasks not yet implemented (nice-to-haves):

- [ ] **GIF Export**: Export combined video+graph as GIF
- [ ] **Bookmarks**: Mark interesting frames for review
- [ ] **Jump to Next Event**: Navigate between detected events
- [ ] **Export Events CSV**: Save detected events to CSV
- [ ] **Keyboard Shortcuts**: Arrow keys, space bar, etc.
- [ ] **Zoom Controls**: Explicit zoom buttons on graph
- [ ] **Overlay Keypoints**: Draw SLEAP points on video frame

---

## Files Modified/Created Summary

```
gui_v2/
├── TUNE_PARAMETERS_IMPLEMENTATION.md    [CREATED] Design doc
├── IMPLEMENTATION_COMPLETE.md           [CREATED] This file
├── detection_utils.py                   [CREATED] Helper functions
├── node_mapping_dialog.py               [CREATED] Node config dialog
├── video_inspector_widget.py            [CREATED] Video display widget
├── diagnostics_graph_widget.py          [CREATED] Graph widget
├── parameter_panel.py                   [MODIFIED] Added analysis buttons
└── main_window.py                       [MODIFIED] Integration + detection
```

**Total new code**: ~1,200 lines across 4 new files + modifications to 2 existing files

---

## Questions or Issues?

If you encounter any problems during testing:

1. Check console output for error messages
2. Verify H5 file is valid SLEAP export format
3. Ensure core detection modules are importable
4. Try with a simple test case (short video, few frames)
5. Review `TUNE_PARAMETERS_IMPLEMENTATION.md` for architecture details

**All implementation is complete and ready for testing!**

---

**Enjoy tuning your HTR detection parameters! 🧠🐭📊**
