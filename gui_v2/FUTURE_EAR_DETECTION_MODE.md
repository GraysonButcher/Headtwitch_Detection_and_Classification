# Future Enhancement: Single-Ear Oscillation Detection Mode

**Created:** 2025-01-28
**Status:** Not Implemented - Reference for Future Work

---

## Problem Statement

The current ear detector requires a "crisscross" pattern where one ear moves toward the midline while the other moves away. This works well for rat data but fails for some mouse datasets where: Both ears oscillate, but their absolute height thresholds never "crisscross" (i.e., one ear is always further from the midline than the other). Thus absolute height thresholds don't capture the relative oscillations. 

## Proposed Solution: Option D - Single-Ear Oscillation Mode

Add an alternative ear detection mode that:
1. Uses **prominence-based** peak/valley detection (like the head detector) instead of absolute height thresholds
2. Detects oscillations in **each ear independently** rather than requiring crisscross patterns
3. Counts oscillations in either ear as potential headtwitch indicators
4. (Optional) And potentially still uses some of the old "crisscross" logic: just the part about how the peak of one ear occurs around the same time as the valley of the other ear (over and over). It's only that the requirement that these peaks and valleys "cross" is eliminated and instead the peaks and valley identifications are replaced by the relative prominence (movement) instead of absolute movement with respect to midline and each other. 

## Implementation Approach

### Step 1: Add Prominence-Based Ear Detection

Modify `EarsDetector._detect_crisscross_units()` to support prominence mode:

```python
# Current (absolute height):
left_peaks, _ = find_peaks(left_dist, height=self.config.peak_threshold, distance=self.config.max_gap)
left_valleys, _ = find_peaks(-left_dist, height=-self.config.valley_threshold, distance=self.config.max_gap)

# New (prominence-based):
left_peaks, _ = find_peaks(left_dist, prominence=self.config.ear_prominence, distance=self.config.max_gap)
left_valleys, _ = find_peaks(-left_dist, prominence=self.config.ear_prominence, distance=self.config.max_gap)
```

### Step 2: Add Single-Ear Oscillation Grouping

Create new method `_detect_single_ear_oscillations()` that:
1. Finds peaks and valleys in each ear signal independently
2. Groups consecutive peak-valley pairs (like head detector does)
3. Counts oscillations per ear
4. Marks regions with sufficient oscillations as potential headtwitches

### Step 3: Add Configuration Options

Add to `EarDetectorConfig`:
```python
use_prominence_mode: bool = False  # Use prominence instead of absolute thresholds
ear_prominence: int = 5            # Prominence value for peak/valley detection
require_crisscross: bool = True    # If False, use single-ear oscillation mode
min_single_ear_oscillations: int = 3  # Min oscillations in one ear to count
```

### Step 4: Update Parameter Panel UI

Add controls in `parameter_panel.py`:
- Checkbox: "Use Relative Detection (Prominence)"
- Checkbox: "Allow Single-Ear Detection"
- Spinbox: "Ear Prominence" (shown when prominence mode enabled)
- Spinbox: "Min Single-Ear Oscillations"

## Code Files to Review

| File | Relevant Sections | Purpose |
|------|-------------------|---------|
| `core/detectors.py` | `EarsDetector` class (lines 13-189) | Current ear detection logic |
| `core/detectors.py` | `_detect_crisscross_units()` (lines 52-103) | Peak/valley detection and crisscross pairing |
| `core/detectors.py` | `_group_into_headshakes()` (lines 105-138) | Grouping crisscrosses into events |
| `core/detectors.py` | `HeadDetector._detect_oscillations()` (lines 248-260) | Example of prominence-based detection |
| `core/detectors.py` | `HeadDetector._build_cycles_from_events()` (lines 262-284) | Example of single-signal cycle building |
| `core/config.py` | `EarDetectorConfig` class | Configuration dataclass to extend |
| `gui_v2/parameter_panel.py` | `create_detector_row_groups()` (lines 199-256) | UI for ear detector parameters |

## Key Logic to Preserve

When implementing, ensure backwards compatibility:
1. Default behavior should match current detector (crisscross mode with absolute thresholds)
2. New modes should be opt-in via configuration
3. Existing saved parameter files should continue to work

## Testing Considerations

1. Test with rat dataset (should work same as before in default mode)
2. Test with mouse dataset (should detect headtwitches in new mode)
3. Test parameter save/load with new config options
4. Verify visualization updates work with new detection mode

## Visual Feedback Updates

The `diagnostics_graph_widget.py` peak/valley visualization would need updates:
1. When prominence mode is enabled, use prominence parameter instead of height
2. When single-ear mode is enabled, show oscillation groupings per ear (not crisscross pairs)
3. Consider different visual indicator for single-ear vs crisscross detections

---

## References

- Original discussion: Conversation on 2025-01-28 regarding mouse vs rat ear movement patterns
- Related: The right ear in mouse data shows minimal movement during headtwitches, while left ear oscillates significantly
