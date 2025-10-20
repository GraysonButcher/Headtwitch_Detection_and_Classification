"""
Quick Test Script for GUI v3

Tests the new 5-tab workflow structure.
"""

import sys
import os

# Add gui_v2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gui_v2'))

from gui_v2.main_window import HTRAnalysisAppV3
from PySide6.QtWidgets import QApplication

def test_gui_v3():
    """Test the new GUI v3."""
    print("=" * 60)
    print("HTR Analysis Tool v3 - Test Launch")
    print("=" * 60)
    print()
    print("Testing new 5-tab structure:")
    print("  1. Welcome - Project overview and workflow navigation")
    print("  2. Tune Parameters - Parameter optimization with video feedback")
    print("  3. Prepare Data - Feature extraction + ground truth labeling")
    print("  4. Train Model - ML training with evaluation and iteration")
    print("  5. Deploy - Smart batch processing (fresh and incremental)")
    print()
    print("New features:")
    print("  ✓ Built-in CSV editor for ground truth labeling")
    print("  ✓ Smart incremental batch processing")
    print("  ✓ Workflow state tracking")
    print("  ✓ Model evaluation and iteration support")
    print()
    print("Launching GUI...")
    print("=" * 60)

    app = QApplication(sys.argv)
    app.setApplicationName("HTR Analysis Tool v3 - Test")

    window = HTRAnalysisAppV3()
    window.show()

    print()
    print("✓ GUI launched successfully!")
    print("  - Window size: 1400x750")
    print("  - Tabs: 5")
    print()
    print("Test each tab:")
    print("  [ ] Welcome tab loads and shows 4 workflow cards")
    print("  [ ] Tune Parameters tab shows parameter panel")
    print("  [ ] Prepare Data tab shows Extract Features + Label sections")
    print("  [ ] Train Model tab shows Configure & Evaluate sections")
    print("  [ ] Deploy tab shows processing mode selection")
    print()
    print("Close the window when done testing.")

    sys.exit(app.exec())

if __name__ == "__main__":
    test_gui_v3()
