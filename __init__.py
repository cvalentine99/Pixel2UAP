"""
Voxel Projector v2
------------------
An advanced tool for projecting 2D images and medical data into 3D voxel representations.
"""

__version__ = '0.1.0'

import os
import sys
import platform

# Platform-specific initializations
OS_NAME = platform.system()  # 'Windows', 'Linux', 'Darwin'

# Platform-specific path handling
if OS_NAME == 'Windows':
    CONFIG_PATH = os.path.join(os.getenv('APPDATA'), 'VoxelProjector')
elif OS_NAME == 'Darwin':  # macOS
    CONFIG_PATH = os.path.expanduser('~/Library/Application Support/VoxelProjector')
else:  # Linux and others
    CONFIG_PATH = os.path.expanduser('~/.config/voxel-projector')

# Create config directory if it doesn't exist
os.makedirs(CONFIG_PATH, exist_ok=True)

# Feature detection
try:
    import pyvista
    import pyvistaqt
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    import nibabel
    NIFTI_AVAILABLE = True
except ImportError:
    NIFTI_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False