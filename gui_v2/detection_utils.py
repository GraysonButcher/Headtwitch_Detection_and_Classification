"""
Detection Utilities - Helper functions for HTR signal calculation.

Extracted from legacy implementations for reuse across video inspector components.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def line_intersection(p1: np.ndarray, p2: np.ndarray,
                      q1: np.ndarray, q2: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate the intersection point of two lines.

    Args:
        p1: First point of first line (x, y)
        p2: Second point of first line (x, y)
        q1: First point of second line (x, y)
        q2: Second point of second line (x, y)

    Returns:
        Intersection point as (x, y) array, or None if lines are parallel
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator

    intersection = np.array([
        x1 + t * (x2 - x1),
        y1 + t * (y2 - y1)
    ])

    return intersection


def point_line_distance(point: np.ndarray, line_start: np.ndarray,
                       line_end: np.ndarray) -> float:
    """
    Calculate perpendicular distance from a point to a line.

    Args:
        point: Point coordinates (x, y)
        line_start: Start point of line (x, y)
        line_end: End point of line (x, y)

    Returns:
        Perpendicular distance from point to line
    """
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)

    # If line start and end are the same, return distance to that point
    if np.allclose(a, b):
        return float(np.linalg.norm(p - a))

    # Calculate projection parameter
    t = np.dot(p - a, b - a) / np.dot(b - a, b - a)

    # Calculate projection point
    projection = a + t * (b - a)

    # Return distance from point to projection
    return float(np.linalg.norm(p - projection))


def calculate_ear_distances(left_ear_locs: np.ndarray,
                            right_ear_locs: np.ndarray,
                            back_locs: np.ndarray,
                            nose_locs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate distances from ears to the intersection of back-nose and ear-ear lines.

    This is the core signal for ear-based headshake detection.

    Args:
        left_ear_locs: Array of left ear positions, shape (n_frames, 2)
        right_ear_locs: Array of right ear positions, shape (n_frames, 2)
        back_locs: Array of back positions, shape (n_frames, 2)
        nose_locs: Array of nose positions, shape (n_frames, 2)

    Returns:
        Tuple of (left_distances, right_distances), both shape (n_frames,)
        NaN values are inserted where intersection cannot be calculated
    """
    n_frames = len(back_locs)
    left_distances = []
    right_distances = []

    for i in range(n_frames):
        # Find intersection of back-nose line and left_ear-right_ear line
        intersection = line_intersection(
            back_locs[i], nose_locs[i],
            left_ear_locs[i], right_ear_locs[i]
        )

        if intersection is not None:
            # Calculate distances from each ear to intersection
            left_dist = np.linalg.norm(left_ear_locs[i] - intersection)
            right_dist = np.linalg.norm(right_ear_locs[i] - intersection)
            left_distances.append(left_dist)
            right_distances.append(right_dist)
        else:
            # Lines are parallel or identical - insert NaN
            left_distances.append(np.nan)
            right_distances.append(np.nan)

    # Convert to arrays and interpolate NaN values
    left_distances = pd.Series(left_distances).interpolate().fillna(0.0).to_numpy()
    right_distances = pd.Series(right_distances).interpolate().fillna(0.0).to_numpy()

    return left_distances, right_distances


def calculate_head_signal(head_locs: np.ndarray,
                          back_locs: np.ndarray,
                          nose_locs: np.ndarray) -> np.ndarray:
    """
    Calculate head-to-midline distance signal.

    This is the core signal for head-based headshake detection.
    The midline is defined as the line from back to nose.

    Args:
        head_locs: Array of head positions, shape (n_frames, 2)
        back_locs: Array of back positions, shape (n_frames, 2)
        nose_locs: Array of nose positions, shape (n_frames, 2)

    Returns:
        Array of distances from head to midline, shape (n_frames,)
        NaN values are interpolated
    """
    n_frames = len(head_locs)
    distances = []

    for i in range(n_frames):
        # Calculate perpendicular distance from head to back-nose line
        dist = point_line_distance(head_locs[i], back_locs[i], nose_locs[i])
        distances.append(dist)

    # Convert to array and interpolate NaN values
    distances = pd.Series(distances).interpolate().fillna(0.0).to_numpy()

    return distances


def normalize_sleap_tracks(tracks_array: np.ndarray) -> np.ndarray:
    """
    Normalize SLEAP tracks array to consistent shape (n_frames, n_nodes, 2).

    SLEAP outputs can have various shapes depending on export settings:
    - (n_frames, n_nodes, 2, n_instances)
    - (n_nodes, 2, n_frames, n_instances)
    - (n_instances, 2, n_nodes, n_frames) <- Common format
    - (n_frames, n_nodes, 2)

    This function handles common variants and always returns (n_frames, n_nodes, 2).

    Args:
        tracks_array: Raw tracks array from SLEAP H5 file

    Returns:
        Normalized array with shape (n_frames, n_nodes, 2)

    Raises:
        ValueError: If array shape cannot be normalized
    """
    arr = np.array(tracks_array)
    original_shape = arr.shape

    # Handle 4D arrays (with instances)
    if arr.ndim == 4:
        shape = arr.shape

        # Pattern: (n_instances, 2, n_nodes, n_frames) - SLEAP v1.3+ format
        # Example: (1, 2, 6, 285356)
        if shape[0] == 1 and shape[1] == 2:
            arr = arr[0, :, :, :]  # Remove instance dim -> (2, n_nodes, n_frames)
            arr = np.transpose(arr, (2, 1, 0))  # -> (n_frames, n_nodes, 2)
            return arr

        # Pattern: (n_frames, n_nodes, 2, n_instances)
        if shape[2] == 2:
            arr = arr[..., 0]  # Take first instance
            return arr

        # Pattern: (n_nodes, 2, n_frames, n_instances)
        elif shape[1] == 2:
            arr = arr[..., 0]  # Take first instance
            arr = np.transpose(arr, (2, 0, 1))  # -> (n_frames, n_nodes, 2)
            return arr

        # Pattern: (n_instances, n_frames, n_nodes, 2)
        elif shape[3] == 2:
            arr = arr[0, :, :, :]  # Take first instance -> (n_frames, n_nodes, 2)
            return arr

    # Handle 3D arrays (squeezed or no instances)
    if arr.ndim == 3:
        shape = arr.shape
        # Already (n_frames, n_nodes, 2)
        if shape[2] == 2:
            return arr
        # Pattern: (2, n_nodes, n_frames)
        elif shape[0] == 2:
            arr = np.transpose(arr, (2, 1, 0))  # -> (n_frames, n_nodes, 2)
            return arr
        # Pattern: (n_nodes, 2, n_frames)
        elif shape[1] == 2:
            arr = np.transpose(arr, (2, 0, 1))  # -> (n_frames, n_nodes, 2)
            return arr

    # If we get here, we couldn't normalize
    raise ValueError(
        f"Cannot normalize tracks array with shape {original_shape}. "
        f"Expected shape reducible to (n_frames, n_nodes, 2). "
        f"Detected {arr.ndim}D array with dimensions {arr.shape}."
    )


def normalize_sleap_scores(scores_array: np.ndarray,
                           n_frames: int,
                           n_nodes: int) -> np.ndarray:
    """
    Normalize SLEAP point scores array to consistent shape (n_frames, n_nodes).

    Args:
        scores_array: Raw point scores array from SLEAP H5 file
        n_frames: Expected number of frames
        n_nodes: Expected number of nodes

    Returns:
        Normalized array with shape (n_frames, n_nodes)

    Raises:
        ValueError: If array cannot be normalized to expected shape
    """
    arr = np.array(scores_array)
    original_shape = arr.shape

    # Handle 3D arrays (with instances)
    if arr.ndim == 3:
        shape = arr.shape

        # Pattern: (n_instances, n_nodes, n_frames) - SLEAP v1.3+ format
        if shape[0] == 1:
            arr = arr[0, :, :]  # Remove instance dim -> (n_nodes, n_frames)
            if arr.shape == (n_nodes, n_frames):
                return arr.T  # Transpose to (n_frames, n_nodes)
            elif arr.shape == (n_frames, n_nodes):
                return arr

        # Take first instance from other patterns
        arr = arr[:, :, 0]

    # Handle 2D arrays
    if arr.ndim == 2:
        # Already correct shape
        if arr.shape == (n_frames, n_nodes):
            return arr
        # Transposed
        elif arr.shape == (n_nodes, n_frames):
            return arr.T

    # Try to reshape if total size matches
    if arr.size == n_frames * n_nodes:
        # Try both orientations
        reshaped = arr.reshape(n_frames, n_nodes)
        return reshaped

    # Give up gracefully
    raise ValueError(
        f"Cannot normalize scores array with shape {original_shape} "
        f"to expected shape ({n_frames}, {n_nodes}). "
        f"Array has {arr.size} elements, expected {n_frames * n_nodes}."
    )
