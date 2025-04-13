"""
Core implementation of Pixel Motion Voxel Projection technique.

This module implements the main functionality for detecting objects through
motion analysis and projecting the results into 3D voxel space.
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional, Union, Any
import os

# Setup Vulkan for GPU acceleration if possible
os.environ["OPEN3D_ENABLE_VULKAN"] = "1"

from .data_structures import VoxelGrid, CameraInfo, MotionMap

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


class PixelMotionVoxelProjection:
    """
    Main class for the Pixel Motion Voxel Projection technique.
    
    This class handles the core workflow:
    1. Process frame differences to detect motion
    2. Project pixels with motion into 3D voxel space 
    3. Combine multiple projections from different perspectives
    4. Find intersections to identify objects in 3D space
    
    Attributes:
        voxel_grid (VoxelGrid): 3D grid for accumulating projections
        cameras (Dict[str, CameraInfo]): Information for each camera
        motion_history (List[MotionMap]): Recent motion detection results
        use_gpu (bool): Whether to use GPU acceleration
    """
    
    def __init__(self, 
                grid_resolution: Tuple[int, int, int] = (100, 100, 100),
                grid_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
                use_gpu: bool = True):
        """
        Initialize the pixel motion voxel projection system.
        
        Args:
            grid_resolution (Tuple[int, int, int]): Resolution of the voxel grid
            grid_bounds (Tuple[Tuple[float, float], ...]): Physical bounds for the voxel grid
            use_gpu (bool): Whether to use GPU acceleration when available
        """
        self.voxel_grid = VoxelGrid(grid_resolution, grid_bounds)
        self.cameras = {}
        self.motion_history = []
        self.history_limit = 100  # Maximum number of motion maps to keep
        
        # Set up GPU acceleration if requested and available
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        if use_gpu and not CUPY_AVAILABLE:
            logger.warning("GPU acceleration requested but CuPy not available. Using CPU instead.")
        
        # Initialize visualization components
        self.visualizer = None
        
    def add_camera(self, camera: CameraInfo) -> None:
        """
        Add a camera to the system.
        
        Args:
            camera (CameraInfo): Camera information with calibration data
        """
        self.cameras[camera.name] = camera
        logger.info(f"Added camera '{camera.name}' to the system")
        
    def remove_camera(self, camera_name: str) -> bool:
        """
        Remove a camera from the system.
        
        Args:
            camera_name (str): Name of the camera to remove
            
        Returns:
            bool: True if camera was removed, False if it didn't exist
        """
        if camera_name in self.cameras:
            del self.cameras[camera_name]
            logger.info(f"Removed camera '{camera_name}' from the system")
            return True
        return False
    
    def process_frame_difference(self,
                                frame1: np.ndarray,
                                frame2: np.ndarray,
                                camera_name: str,
                                timestamp: float = None,
                                threshold: float = 15.0,
                                preprocess: bool = True) -> MotionMap:
        """
        Calculate motion between two sequential frames.
        
        Args:
            frame1 (np.ndarray): First frame (earlier in time)
            frame2 (np.ndarray): Second frame (later in time)
            camera_name (str): Name of the camera that captured the frames
            timestamp (float): Timestamp for the frames (default: current time)
            threshold (float): Motion detection threshold
            preprocess (bool): Whether to apply preprocessing to frames
            
        Returns:
            MotionMap: Processed motion data
        """
        if camera_name not in self.cameras:
            raise ValueError(f"Camera '{camera_name}' not registered in the system")
        
        if timestamp is None:
            timestamp = time.time()
            
        # Check if frames have the same shape
        if frame1.shape != frame2.shape:
            raise ValueError(f"Frames must have the same shape. Got {frame1.shape} and {frame2.shape}")
            
        # Convert to grayscale if needed
        if len(frame1.shape) == 3 and frame1.shape[2] == 3:
            if OPENCV_AVAILABLE:
                frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                # Basic grayscale conversion without OpenCV
                frame1_gray = np.dot(frame1[..., :3], [0.299, 0.587, 0.114])
                frame2_gray = np.dot(frame2[..., :3], [0.299, 0.587, 0.114])
        else:
            frame1_gray = frame1
            frame2_gray = frame2
            
        # Preprocess if requested
        if preprocess and OPENCV_AVAILABLE:
            # Apply Gaussian blur to reduce noise
            frame1_gray = cv2.GaussianBlur(frame1_gray, (5, 5), 0)
            frame2_gray = cv2.GaussianBlur(frame2_gray, (5, 5), 0)
        
        # Calculate absolute difference between frames
        if self.use_gpu:
            # Use GPU for frame difference calculation
            frame1_gpu = cp.asarray(frame1_gray)
            frame2_gpu = cp.asarray(frame2_gray)
            diff_gpu = cp.abs(frame2_gpu - frame1_gpu)
            frame_diff = cp.asnumpy(diff_gpu)
        else:
            # Use CPU for frame difference calculation
            frame_diff = np.abs(frame2_gray - frame1_gray)
            
        # Create motion map
        motion_map = MotionMap(
            data=frame_diff,
            threshold=threshold,
            timestamp=timestamp,
            camera_info=self.cameras[camera_name]
        )
        
        # Apply morphological operations to clean up noise if OpenCV is available
        if preprocess and OPENCV_AVAILABLE:
            # First erode to remove small noise
            motion_map = motion_map.erode(kernel_size=3)
            # Then dilate to enhance motion regions
            motion_map = motion_map.dilate(kernel_size=3)
            
        # Store in history
        self.motion_history.append(motion_map)
        
        # Trim history if needed
        if len(self.motion_history) > self.history_limit:
            self.motion_history = self.motion_history[-self.history_limit:]
            
        return motion_map
    
    def project_pixels_to_voxels(self, 
                                motion_map: MotionMap,
                                max_ray_distance: float = None,
                                voxel_value: float = 1.0) -> VoxelGrid:
        """
        Project motion pixels into 3D voxel space.
        
        Projects rays from the camera through each motion pixel and marks
        voxels along each ray's path.
        
        Args:
            motion_map (MotionMap): Motion map with detected pixels
            max_ray_distance (float): Maximum distance to project rays
            voxel_value (float): Value to add to each intersected voxel
            
        Returns:
            VoxelGrid: Voxel grid with projections
        """
        if motion_map.camera_info is None:
            raise ValueError("Motion map must have associated camera information")
            
        # Create new voxel grid for this projection
        projection_grid = VoxelGrid(
            self.voxel_grid.resolution,
            self.voxel_grid.bounds
        )
        
        # Get motion pixels
        motion_pixels = motion_map.get_motion_pixels()
        
        # Check if we have any motion pixels
        if len(motion_pixels) == 0:
            return projection_grid
            
        # Get camera info
        camera = motion_map.camera_info
        camera_origin = camera.position
        
        # Process each motion pixel
        if self.use_gpu and CUPY_AVAILABLE:
            # TODO: Implement batched GPU version for ray tracing
            # For now, fall back to CPU implementation
            self._project_pixels_cpu(projection_grid, motion_pixels, camera, max_ray_distance, voxel_value)
        else:
            # CPU implementation
            self._project_pixels_cpu(projection_grid, motion_pixels, camera, max_ray_distance, voxel_value)
            
        return projection_grid
    
    def _project_pixels_cpu(self, 
                           grid: VoxelGrid, 
                           pixels: np.ndarray, 
                           camera: CameraInfo,
                           max_distance: float,
                           value: float) -> None:
        """
        Project pixels to voxels using CPU implementation.
        
        Args:
            grid (VoxelGrid): Voxel grid to update
            pixels (np.ndarray): Array of pixel coordinates (x, y)
            camera (CameraInfo): Camera information
            max_distance (float): Maximum ray distance
            value (float): Value to add to each voxel
        """
        for pixel in pixels:
            # Get ray from pixel
            origin, direction = camera.pixel_to_ray(pixel)
            
            # Trace ray through voxel grid
            grid.ray_intersect(origin, direction, value, max_distance)
    
    def _project_pixels_gpu(self, 
                           grid: VoxelGrid, 
                           pixels: np.ndarray, 
                           camera: CameraInfo,
                           max_distance: float,
                           value: float) -> None:
        """
        Project pixels to voxels using GPU implementation.
        
        Args:
            grid (VoxelGrid): Voxel grid to update
            pixels (np.ndarray): Array of pixel coordinates (x, y)
            camera (CameraInfo): Camera information
            max_distance (float): Maximum ray distance
            value (float): Value to add to each voxel
        """
        # Note: This is a placeholder for future GPU implementation
        # Currently relies on CPU implementation
        logger.warning("GPU implementation not yet available, using CPU instead")
        self._project_pixels_cpu(grid, pixels, camera, max_distance, value)
    
    def combine_voxel_projections(self, 
                                 voxel_grids: List[VoxelGrid],
                                 operation: str = "add") -> VoxelGrid:
        """
        Combine multiple voxel grids from different projections.
        
        Args:
            voxel_grids (List[VoxelGrid]): List of voxel grids to combine
            operation (str): Operation to combine grids ("add", "multiply", "min", "max")
            
        Returns:
            VoxelGrid: Combined voxel grid
        """
        if not voxel_grids:
            raise ValueError("No voxel grids provided to combine")
            
        if len(voxel_grids) == 1:
            return voxel_grids[0]
            
        # Start with the first grid
        result = voxel_grids[0]
        
        # Combine with each remaining grid
        for grid in voxel_grids[1:]:
            result = result.combine(grid, operation)
            
        return result
    
    def find_intersections(self, 
                          threshold: float,
                          min_cluster_size: int = 10) -> List[np.ndarray]:
        """
        Identify statistically significant voxel intersections.
        
        Args:
            threshold (float): Threshold value for considering a voxel
            min_cluster_size (int): Minimum size of a valid cluster
            
        Returns:
            List[np.ndarray]: List of 3D coordinates for detected objects
        """
        # Get object centers from the voxel grid
        return self.voxel_grid.get_object_centers(threshold, min_cluster_size)
    
    def process_video_frame(self,
                           frame: np.ndarray,
                           camera_name: str,
                           timestamp: float = None) -> None:
        """
        Process a single video frame, updating the system state.
        
        Args:
            frame (np.ndarray): New video frame
            camera_name (str): Name of the camera that captured the frame
            timestamp (float): Timestamp for the frame
        """
        # Skip if this is the first frame from this camera
        previous_frames = [m for m in self.motion_history 
                           if m.camera_info.name == camera_name]
        
        if not previous_frames:
            # Store frame for future processing
            if timestamp is None:
                timestamp = time.time()
                
            # Create a dummy motion map just to store the frame
            dummy_motion = np.zeros_like(frame if len(frame.shape) == 2 else frame[:, :, 0])
            self.motion_history.append(MotionMap(
                data=dummy_motion,
                threshold=0.0,
                timestamp=timestamp,
                camera_info=self.cameras[camera_name]
            ))
            
            # Store the original frame for future reference
            setattr(self.motion_history[-1], 'original_frame', frame)
            
            return
            
        # Get most recent frame from this camera
        latest_motion = max(previous_frames, key=lambda m: m.timestamp)
        
        # Get the previous original frame
        if hasattr(latest_motion, 'original_frame'):
            previous_frame = latest_motion.original_frame
            
            # Process frame difference
            motion_map = self.process_frame_difference(
                previous_frame, frame, camera_name, timestamp
            )
            
            # Save original frame for next time
            setattr(motion_map, 'original_frame', frame)
            
            # Project motion to voxels
            projection = self.project_pixels_to_voxels(motion_map)
            
            # Update the global voxel grid - using addition for accumulation
            self.voxel_grid = self.voxel_grid.combine(projection, "add")
        else:
            # If no original frame stored, just store current frame
            if timestamp is None:
                timestamp = time.time()
                
            dummy_motion = np.zeros_like(frame if len(frame.shape) == 2 else frame[:, :, 0])
            new_map = MotionMap(
                data=dummy_motion,
                threshold=0.0,
                timestamp=timestamp,
                camera_info=self.cameras[camera_name]
            )
            setattr(new_map, 'original_frame', frame)
            self.motion_history.append(new_map)
    
    def process_video_stream(self,
                            stream,
                            camera_name: str,
                            max_frames: int = None,
                            frame_interval: int = 1) -> None:
        """
        Process frames from a video stream.
        
        Args:
            stream: Video stream object (like cv2.VideoCapture)
            camera_name (str): Name of the camera
            max_frames (int): Maximum number of frames to process
            frame_interval (int): Process every Nth frame
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required for video stream processing")
            
        if not hasattr(stream, 'read'):
            raise ValueError("Stream object must have a 'read' method")
            
        frame_count = 0
        processed_count = 0
        
        while True:
            # Read frame
            ret, frame = stream.read()
            
            if not ret:
                # End of stream
                break
                
            frame_count += 1
            
            # Process only every Nth frame
            if frame_count % frame_interval != 0:
                continue
                
            # Process frame
            self.process_video_frame(frame, camera_name)
            
            processed_count += 1
            
            # Check if we've reached the max frame count
            if max_frames is not None and processed_count >= max_frames:
                break
                
        logger.info(f"Processed {processed_count} frames from video stream")
    
    def reset_voxel_grid(self) -> None:
        """Reset the voxel grid to all zeros."""
        resolution = self.voxel_grid.resolution
        bounds = self.voxel_grid.bounds
        self.voxel_grid = VoxelGrid(resolution, bounds)
        
    def get_detected_objects(self, 
                           threshold: float = 5.0,
                           min_cluster_size: int = 10) -> List[np.ndarray]:
        """
        Get the 3D positions of detected objects.
        
        Args:
            threshold (float): Detection threshold
            min_cluster_size (int): Minimum cluster size
            
        Returns:
            List[np.ndarray]: 3D positions of detected objects
        """
        return self.find_intersections(threshold, min_cluster_size)
    
    def visualize(self, show_volume: bool = True, show_points: bool = True) -> Any:
        """
        Visualize the current state of the voxel grid.
        
        Args:
            show_volume (bool): Whether to show the volume rendering
            show_points (bool): Whether to show detected object points
            
        Returns:
            Any: Visualization object or None if visualization is not available
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D required for visualization")
            return None
            
        # Initialize visualizer if needed
        if self.visualizer is None:
            self.visualizer = o3d.visualization.Visualizer()
            self.visualizer.create_window()
            
            # Add coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            self.visualizer.add_geometry(frame)
            
            # Add initial point cloud
            self.point_cloud = self.voxel_grid.to_point_cloud()
            self.visualizer.add_geometry(self.point_cloud)
            
            # Set up view
            view_control = self.visualizer.get_view_control()
            view_control.set_front([0, 0, -1])  # Look toward negative z-axis
            view_control.set_up([0, -1, 0])     # Up is negative y-axis
            view_control.set_zoom(0.8)
            
        else:
            # Update point cloud
            new_cloud = self.voxel_grid.to_point_cloud()
            self.point_cloud.points = new_cloud.points
            self.point_cloud.colors = new_cloud.colors
            self.visualizer.update_geometry(self.point_cloud)
        
        # Update visualization
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
        return self.visualizer
    
    def close_visualization(self) -> None:
        """Close the visualization window."""
        if hasattr(self, 'visualizer') and self.visualizer is not None:
            self.visualizer.destroy_window()
            self.visualizer = None