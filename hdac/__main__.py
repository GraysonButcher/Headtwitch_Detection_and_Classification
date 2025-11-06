"""
Main entry point for the H-DaC application.

This module is executed when running 'hdac' from the command line
after installing the package.
"""

import sys
import os

# Add the parent directory to path to import gui_v2 and core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui_v2.main_window import HTRAnalysisAppV3
from PySide6.QtWidgets import QApplication


def main():
    """Launch the H-DaC GUI application."""
    print("=" * 60)
    print("H-DaC: Head-Twitch Detection & Classification Tool v3")
    print("=" * 60)
    print()
    print("Launching application...")

    app = QApplication(sys.argv)
    app.setApplicationName("H-DaC v3")
    app.setOrganizationName("H-DaC")

    window = HTRAnalysisAppV3()
    window.show()

    print("Application launched successfully!")
    print()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
