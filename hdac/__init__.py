"""
H-DaC: Head-Twitch Response Detection & Classification Tool

A comprehensive desktop application for detecting and analyzing Head-Twitch
Responses (HTRs) in rodent behavioral videos using SLEAP pose-tracking data
and machine learning.
"""

__version__ = "3.0.0"
__author__ = "Grayson Butcher"

# Make version accessible
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hdac")
except PackageNotFoundError:
    # Package is not installed
    pass
