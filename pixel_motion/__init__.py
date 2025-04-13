"""
Pixel2UAP: UAP Detection and Tracking Module

This module implements specialized techniques for detecting Unidentified Aerial Phenomena
(UAP) through advanced motion analysis and 3D spatial tracking. It is designed specifically
for identifying and analyzing unusual aerial objects that exhibit distinctive movement patterns.

The implementation analyzes pixel changes between sequential video frames, projects potential
UAP signatures into 3D space using camera orientation data, and tracks movement trajectories
over time to identify objects with flight characteristics consistent with UAP observations.
"""

from .core import PixelMotionVoxelProjection
from .data_structures import VoxelGrid, CameraInfo, MotionMap
from .visualization import UAPVisualizer
from .calibration import CameraCalibrator
from .utils import MotionFilter, ObjectTracker, ExportManager

__all__ = [
    'PixelMotionVoxelProjection',
    'VoxelGrid',
    'CameraInfo',
    'MotionMap',
    'UAPVisualizer',
    'CameraCalibrator',
    'MotionFilter',
    'ObjectTracker',
    'ExportManager',
]