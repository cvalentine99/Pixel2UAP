# Pixel Motion Voxel Projection

A technique for detecting objects through motion analysis and voxel projection from multiple camera perspectives.

## Overview

The Pixel Motion Voxel Projection technique is designed for detecting moving objects against complex backgrounds, particularly useful for applications tracking aircraft, drones, birds, and other objects where traditional detection methods may struggle.

Key features:
- Frame difference analysis to detect motion
- Projection of motion pixels into 3D voxel space
- Combination of projections from multiple perspectives
- Finding intersections to identify objects in 3D space
- Optional GPU acceleration with Vulkan/CuPy

## Module Structure

The implementation is organized into several modules:

- `__init__.py`: Package structure and exports
- `data_structures.py`: Core data classes (VoxelGrid, CameraInfo, MotionMap)
- `core.py`: Main implementation of PixelMotionVoxelProjection technique
- `visualization.py`: 3D visualization tools for voxel data
- `calibration.py`: Camera calibration utilities
- `utils.py`: Filtering, tracking, and export utilities
- `interface.py`: Interfaces for different input sources

## Dependencies

- Required:
  - NumPy: For numerical operations
  - OpenCV: For image processing and camera access
  
- Optional (for enhanced functionality):
  - Open3D: For 3D visualization (with Vulkan acceleration)
  - CuPy: For GPU acceleration
  - Matplotlib: For plotting and fallback visualization
  - SciPy: For connected component analysis

## Usage Examples

### Basic Setup

```python
from voxel_projector_v2.pixel_motion import (
    PixelMotionVoxelProjection, CameraInfo, VoxelVisualizer
)
import numpy as np

# Create a voxel projector with 100x100x100 grid
projector = PixelMotionVoxelProjection(
    grid_resolution=(100, 100, 100),
    grid_bounds=((-5, 5), (-5, 5), (0, 10)),  # X, Y, Z bounds in meters
    use_gpu=True  # Use GPU acceleration if available
)

# Add two cameras
camera1 = CameraInfo(
    position=np.array([0, 0, 0]),
    orientation=np.eye(3),  # Identity rotation matrix (camera looking forward)
    intrinsic_matrix=np.array([
        [800, 0, 320],
        [0, 800, 240], 
        [0, 0, 1]
    ]),
    distortion_coeffs=np.zeros(5),
    resolution=(640, 480),
    fov=(60, 45),  # Horizontal/vertical field of view in degrees
    name="Camera1"
)

camera2 = CameraInfo(
    position=np.array([2, 1, 0]),
    orientation=np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ]),  # Camera rotated 90 degrees
    intrinsic_matrix=np.array([
        [800, 0, 320],
        [0, 800, 240], 
        [0, 0, 1]
    ]),
    distortion_coeffs=np.zeros(5),
    resolution=(640, 480),
    fov=(60, 45),
    name="Camera2"
)

projector.add_camera(camera1)
projector.add_camera(camera2)

# Initialize visualizer
visualizer = VoxelVisualizer(use_vulkan=True)
```

### Processing Video Frames

```python
import cv2

# Open video files or cameras
cap1 = cv2.VideoCapture("camera1.mp4")
cap2 = cv2.VideoCapture("camera2.mp4")

while True:
    # Read frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break
    
    # Process frames from both cameras
    projector.process_video_frame(frame1, "Camera1")
    projector.process_video_frame(frame2, "Camera2")
    
    # Detect objects from voxel intersections
    objects = projector.get_detected_objects(threshold=5.0)
    
    # Visualize results
    visualizer.visualize_voxel_grid(projector.voxel_grid)
    visualizer.add_detected_objects(objects)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
visualizer.close()
```

### Camera Calibration

```python
from voxel_projector_v2.pixel_motion import CameraCalibrator

# Create a calibrator
calibrator = CameraCalibrator(
    chessboard_size=(9, 6),  # Number of inner corners
    square_size=0.025  # 2.5cm squares
)

# Calibrate camera from chessboard images
images = [cv2.imread(f"calib_{i}.jpg") for i in range(10)]
camera_info = calibrator.calibrate_from_chessboard(
    images, 
    "Camera1",
    use_as_reference=True  # Use this camera as the origin
)

# Save calibration
calibrator.save_calibration("camera_calibration.npy")
```

### Using Interfaces for Input Sources

```python
from voxel_projector_v2.pixel_motion.interface import (
    VideoFileInterface, LiveCameraInterface, MultiCameraInterface
)

# Create projector
projector = PixelMotionVoxelProjection(
    grid_resolution=(100, 100, 100),
    use_gpu=True
)

# Setup interfaces
video_interface = VideoFileInterface(
    camera_info=camera1,
    video_path="video.mp4",
    processor=projector,
    frame_interval=2  # Process every second frame
)

live_interface = LiveCameraInterface(
    camera_info=camera2,
    camera_id=0,
    processor=projector,
    resolution=(1280, 720)
)

# Create multi-camera manager
multi_cam = MultiCameraInterface(projector)
multi_cam.add_interface("video", video_interface)
multi_cam.add_interface("live", live_interface)

# Start all cameras
multi_cam.start_all()

# Process for a while
import time
time.sleep(30)

# Stop all cameras
multi_cam.stop_all()
```

### Advanced Object Tracking

```python
from voxel_projector_v2.pixel_motion.utils import ObjectTracker, ExportManager

# Create and setup tracker
tracker = ObjectTracker(
    max_disappeared=10,  # Maximum frames an object can disappear
    max_distance=0.5     # Maximum distance for matching objects between frames
)

# Create export manager
export_manager = ExportManager(
    output_directory="output",
    enable_compression=True
)

# Process frames
for i in range(100):
    # Detect objects
    objects = projector.get_detected_objects(threshold=5.0)
    
    # Update tracker with detected objects
    tracked_objects = tracker.update(objects)
    
    # Classify trajectories
    classifications = tracker.classify_trajectories()
    
    # Export results periodically
    if i % 10 == 0:
        export_manager.export_detection_results(tracked_objects)
        
        # Export trajectories
        trajectories = tracker.get_trajectories()
        export_manager.export_trajectory_data(
            trajectories,
            classifications
        )
        
        # Create visualization
        export_manager.create_trajectory_visualization(
            trajectories,
            classifications,
            filename=f"trajectory_viz_{i}.png"
        )
```

## Integration with Voxel Projector

This module is designed to integrate with the Voxel Projector application, providing advanced motion-based object detection capabilities. 

## Performance Optimization

For best performance:

1. Enable GPU acceleration when available:
   ```python
   projector = PixelMotionVoxelProjection(use_gpu=True)
   visualizer = VoxelVisualizer(use_vulkan=True)
   ```

2. Adjust grid resolution based on your needs:
   - Higher resolution (e.g., 200x200x200) provides more detailed results but requires more processing power
   - Lower resolution (e.g., 50x50x50) is faster but less precise

3. Use the frame interval parameter to process fewer frames:
   ```python
   video_interface = VideoFileInterface(
       camera_info=camera_info,
       video_path="video.mp4",
       frame_interval=5  # Process every 5th frame
   )
   ```

4. Apply motion filtering to reduce noise:
   ```python
   from voxel_projector_v2.pixel_motion.utils import MotionFilter
   
   motion_filter = MotionFilter(history_length=10)
   filtered_motion = motion_filter.apply_temporal_filter(motion_map)
   filtered_motion = motion_filter.filter_by_size(filtered_motion, min_size=20)
   ```

## Future Enhancements

Planned future enhancements include:

1. Improved GPU acceleration using CUDA kernels
2. Support for real-time streaming over network
3. Integration with deep learning models for object classification
4. Enhanced temporal filtering algorithms
5. WebGL visualization for remote monitoring