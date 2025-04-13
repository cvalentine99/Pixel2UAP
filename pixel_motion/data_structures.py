"""
Data structures for Pixel Motion Voxel Projection

This module defines the foundational data structures used for
motion detection and 3D voxel projection.
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import os

# Setup Vulkan for GPU acceleration
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL for headless rendering
# Use Vulkan by default if available
os.environ["OPEN3D_ENABLE_VULKAN"] = "1"

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    
logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """
    Stores camera calibration, position, and orientation data.
    
    Attributes:
        position (np.ndarray): 3D position of the camera (x, y, z)
        orientation (np.ndarray): Camera orientation as a rotation matrix (3x3)
        intrinsic_matrix (np.ndarray): Camera intrinsic parameters (3x3)
        distortion_coeffs (np.ndarray): Lens distortion coefficients
        resolution (Tuple[int, int]): Camera resolution (width, height)
        fov (Tuple[float, float]): Field of view in degrees (horizontal, vertical)
        name (str): Optional identifier for the camera
    """
    position: np.ndarray
    orientation: np.ndarray
    intrinsic_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    resolution: Tuple[int, int]
    fov: Tuple[float, float]
    name: str = "Camera"
    
    def __post_init__(self):
        """Validate camera parameters after initialization"""
        # Ensure position is a 3D vector
        if self.position.shape != (3,):
            raise ValueError("Camera position must be a 3D vector")
        
        # Ensure orientation is a 3x3 rotation matrix
        if self.orientation.shape != (3, 3):
            raise ValueError("Camera orientation must be a 3x3 rotation matrix")
        
        # Ensure intrinsic matrix is 3x3
        if self.intrinsic_matrix.shape != (3, 3):
            raise ValueError("Camera intrinsic matrix must be 3x3")
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Calculate the camera projection matrix combining intrinsics and extrinsics.
        
        Returns:
            np.ndarray: 3x4 projection matrix
        """
        # Create rotation and translation combined matrix (extrinsic matrix)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = self.orientation
        extrinsic[:3, 3] = self.position
        
        # Create projection matrix (intrinsic * extrinsic)
        projection = np.zeros((3, 4))
        projection[:3, :3] = self.intrinsic_matrix
        
        return projection @ extrinsic
    
    def world_to_pixel(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Project a 3D world point to 2D pixel coordinates.
        
        Args:
            point_3d (np.ndarray): 3D point in world coordinates
            
        Returns:
            np.ndarray: 2D pixel coordinates (u, v)
        """
        # Ensure point is in homogeneous coordinates
        if point_3d.shape[-1] != 4:
            point_homogeneous = np.ones(4)
            point_homogeneous[:3] = point_3d
        else:
            point_homogeneous = point_3d
            
        # Get projection matrix
        projection = self.get_projection_matrix()
        
        # Project point
        pixel_homogeneous = projection @ point_homogeneous
        
        # Convert to 2D coordinates
        pixel = pixel_homogeneous[:2] / pixel_homogeneous[2]
        
        return pixel
    
    def pixel_to_ray(self, pixel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a 2D pixel coordinate to a 3D ray from the camera.
        
        Args:
            pixel (np.ndarray): 2D pixel coordinates (u, v)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Ray origin (camera position) and ray direction
        """
        # Get normalized device coordinates
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        cx = self.intrinsic_matrix[0, 2]
        cy = self.intrinsic_matrix[1, 2]
        
        x = (pixel[0] - cx) / fx
        y = (pixel[1] - cy) / fy
        
        # Create ray direction in camera space
        ray_camera = np.array([x, y, 1.0])
        
        # Transform to world space
        ray_world = self.orientation.T @ ray_camera
        ray_world = ray_world / np.linalg.norm(ray_world)  # Normalize
        
        return self.position, ray_world


class MotionMap:
    """
    Stores and processes the difference maps between sequential frames.
    
    Attributes:
        data (np.ndarray): Motion map data as a 2D array
        threshold (float): Threshold value for distinguishing motion from noise
        timestamp (float): Timestamp when the motion was detected
        camera_info (CameraInfo): Information about the camera that captured the motion
    """
    def __init__(self, 
                 data: np.ndarray, 
                 threshold: float = 15.0,
                 timestamp: float = None,
                 camera_info: Optional[CameraInfo] = None):
        """
        Initialize with motion data, threshold, and optional camera info.
        
        Args:
            data (np.ndarray): Raw motion data (usually frame difference)
            threshold (float): Value for distinguishing motion from noise
            timestamp (float): When the motion was detected (epoch seconds)
            camera_info (CameraInfo): Information about the capturing camera
        """
        self.data = data
        self.threshold = threshold
        self.timestamp = timestamp
        self.camera_info = camera_info
        self.motion_pixels = None  # Cached list of motion pixel coordinates
        
    def get_motion_pixels(self) -> np.ndarray:
        """
        Extract pixel coordinates where motion exceeds the threshold.
        
        Returns:
            np.ndarray: Array of pixel coordinates with shape (N, 2)
        """
        if self.motion_pixels is not None:
            return self.motion_pixels
            
        # Find pixels above threshold
        y_indices, x_indices = np.where(self.data > self.threshold)
        
        # Stack coordinates into pairs
        self.motion_pixels = np.column_stack((x_indices, y_indices))
        
        return self.motion_pixels
    
    def apply_mask(self, mask: np.ndarray) -> "MotionMap":
        """
        Apply a binary mask to the motion map.
        
        Args:
            mask (np.ndarray): Binary mask with same shape as data
            
        Returns:
            MotionMap: New motion map with mask applied
        """
        if mask.shape != self.data.shape:
            raise ValueError(f"Mask shape {mask.shape} must match data shape {self.data.shape}")
        
        masked_data = self.data.copy()
        masked_data[~mask] = 0
        
        return MotionMap(
            masked_data, 
            self.threshold,
            self.timestamp,
            self.camera_info
        )
    
    def dilate(self, kernel_size: int = 3) -> "MotionMap":
        """
        Apply dilation to expand motion areas.
        
        Args:
            kernel_size (int): Size of the dilation kernel
            
        Returns:
            MotionMap: New motion map with dilation applied
        """
        try:
            import cv2
            
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_data = cv2.dilate(self.data, kernel, iterations=1)
            
            return MotionMap(
                dilated_data,
                self.threshold,
                self.timestamp,
                self.camera_info
            )
        except ImportError:
            logger.warning("OpenCV not available for dilation operation")
            return self
    
    def erode(self, kernel_size: int = 3) -> "MotionMap":
        """
        Apply erosion to reduce noise.
        
        Args:
            kernel_size (int): Size of the erosion kernel
            
        Returns:
            MotionMap: New motion map with erosion applied
        """
        try:
            import cv2
            
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded_data = cv2.erode(self.data, kernel, iterations=1)
            
            return MotionMap(
                eroded_data,
                self.threshold,
                self.timestamp,
                self.camera_info
            )
        except ImportError:
            logger.warning("OpenCV not available for erosion operation")
            return self


class VoxelGrid:
    """
    3D grid of voxels for representing motion projections and intersections.
    
    Attributes:
        grid (np.ndarray): 3D array of voxel values
        resolution (Tuple[int, int, int]): Number of voxels in each dimension
        bounds (Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]): 
            Physical bounds of the grid in each dimension (min, max)
        voxel_size (Tuple[float, float, float]): Size of each voxel in physical units
    """
    def __init__(self, 
                 resolution: Tuple[int, int, int],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
                 dtype=np.float32):
        """
        Initialize an empty voxel grid with the specified resolution and bounds.
        
        Args:
            resolution (Tuple[int, int, int]): Number of voxels in (x, y, z)
            bounds (Tuple[Tuple[float, float], ...]): Physical bounds as (min, max) for each dimension
            dtype: Data type for the grid (default: np.float32)
        """
        self.resolution = resolution
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        self.bounds = bounds
        
        # Calculate voxel size
        self.voxel_size = (
            (bounds[0][1] - bounds[0][0]) / resolution[0],
            (bounds[1][1] - bounds[1][0]) / resolution[1],
            (bounds[2][1] - bounds[2][0]) / resolution[2]
        )
        
        # Initialize the grid with zeros
        self.grid = np.zeros(resolution, dtype=dtype)
        
    def world_to_voxel(self, point: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert a 3D world point to voxel grid indices.
        
        Args:
            point (np.ndarray): 3D point in world coordinates (x, y, z)
            
        Returns:
            Tuple[int, int, int]: Voxel indices (i, j, k)
        """
        # Calculate normalized position within bounds (0 to 1)
        normalized = np.array([
            (point[0] - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]),
            (point[1] - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]),
            (point[2] - self.bounds[2][0]) / (self.bounds[2][1] - self.bounds[2][0])
        ])
        
        # Convert to voxel indices
        indices = np.floor(normalized * np.array(self.resolution)).astype(int)
        
        # Clamp to valid indices
        indices = np.clip(indices, 0, np.array(self.resolution) - 1)
        
        return tuple(indices)
    
    def voxel_to_world(self, indices: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel indices to a 3D world point (center of the voxel).
        
        Args:
            indices (Tuple[int, int, int]): Voxel indices (i, j, k)
            
        Returns:
            np.ndarray: 3D point in world coordinates
        """
        # Convert indices to normalized position (center of voxel)
        normalized = (np.array(indices) + 0.5) / np.array(self.resolution)
        
        # Convert to world coordinates
        world = np.array([
            normalized[0] * (self.bounds[0][1] - self.bounds[0][0]) + self.bounds[0][0],
            normalized[1] * (self.bounds[1][1] - self.bounds[1][0]) + self.bounds[1][0],
            normalized[2] * (self.bounds[2][1] - self.bounds[2][0]) + self.bounds[2][0]
        ])
        
        return world
    
    def ray_intersect(self, origin: np.ndarray, direction: np.ndarray, 
                      value: float = 1.0, max_distance: float = None) -> None:
        """
        Perform ray-voxel intersection and increment voxels along the ray.
        
        Uses a 3D DDA (Digital Differential Analyzer) algorithm.
        
        Args:
            origin (np.ndarray): Ray origin in world coordinates
            direction (np.ndarray): Normalized ray direction
            value (float): Value to add to intersected voxels
            max_distance (float): Maximum distance along ray to check
        """
        # Convert origin to grid coordinates
        grid_origin = np.array([
            (origin[0] - self.bounds[0][0]) / (self.bounds[0][1] - self.bounds[0][0]) * self.resolution[0],
            (origin[1] - self.bounds[1][0]) / (self.bounds[1][1] - self.bounds[1][0]) * self.resolution[1],
            (origin[2] - self.bounds[2][0]) / (self.bounds[2][1] - self.bounds[2][0]) * self.resolution[2]
        ])
        
        # Current voxel indices
        voxel = np.floor(grid_origin).astype(int)
        
        # Step direction for traversal
        step = np.sign(direction)
        
        # Calculate delta distance for each axis
        delta = np.abs(np.divide(
            np.array(self.resolution),
            np.array([
                (self.bounds[0][1] - self.bounds[0][0]),
                (self.bounds[1][1] - self.bounds[1][0]),
                (self.bounds[2][1] - self.bounds[2][0])
            ]) * direction,
            out=np.full(3, np.inf),
            where=direction != 0
        ))
        
        # Calculate initial intersection distances
        next_boundary = np.zeros(3)
        for i in range(3):
            if step[i] > 0:
                next_boundary[i] = (voxel[i] + 1)
            else:
                next_boundary[i] = voxel[i]
                
        # Calculate first intersection
        t_max = np.divide(
            (next_boundary - grid_origin) * step,
            direction,
            out=np.full(3, np.inf),
            where=direction != 0
        )
        
        # Convert max_distance to grid units if specified
        grid_max_distance = None
        if max_distance is not None:
            grid_max_distance = max_distance * min([
                self.resolution[0] / (self.bounds[0][1] - self.bounds[0][0]),
                self.resolution[1] / (self.bounds[1][1] - self.bounds[1][0]),
                self.resolution[2] / (self.bounds[2][1] - self.bounds[2][0])
            ])
        
        # Traversal loop
        total_distance = 0
        while True:
            # Check if current voxel is within bounds
            if (0 <= voxel[0] < self.resolution[0] and
                0 <= voxel[1] < self.resolution[1] and
                0 <= voxel[2] < self.resolution[2]):
                
                # Increment voxel value
                self.grid[voxel[0], voxel[1], voxel[2]] += value
            
            # Find axis with smallest t_max
            axis = np.argmin(t_max)
            
            # Check if we've hit max distance
            if grid_max_distance is not None and t_max[axis] > grid_max_distance:
                break
                
            # Update t_max for selected axis
            t_max[axis] += delta[axis]
            
            # Move to next voxel
            voxel[axis] += step[axis]
            
            # Update total distance
            total_distance = t_max[axis]
            
            # Exit if we're outside the grid in any dimension
            if (voxel[0] < 0 or voxel[0] >= self.resolution[0] or
                voxel[1] < 0 or voxel[1] >= self.resolution[1] or
                voxel[2] < 0 or voxel[2] >= self.resolution[2]):
                break
    
    def combine(self, other: "VoxelGrid", operation: str = "add") -> "VoxelGrid":
        """
        Combine this voxel grid with another one.
        
        Args:
            other (VoxelGrid): Another voxel grid with same resolution and bounds
            operation (str): The operation to perform ("add", "multiply", "min", "max")
            
        Returns:
            VoxelGrid: A new voxel grid resulting from the combination
        """
        if self.resolution != other.resolution:
            raise ValueError("Voxel grids must have the same resolution")
            
        result = VoxelGrid(self.resolution, self.bounds, self.grid.dtype)
        
        if operation == "add":
            result.grid = self.grid + other.grid
        elif operation == "multiply":
            result.grid = self.grid * other.grid
        elif operation == "min":
            result.grid = np.minimum(self.grid, other.grid)
        elif operation == "max":
            result.grid = np.maximum(self.grid, other.grid)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return result
    
    def threshold(self, threshold_value: float) -> "VoxelGrid":
        """
        Create a new voxel grid with values thresholded.
        
        Args:
            threshold_value (float): The threshold cutoff value
            
        Returns:
            VoxelGrid: A new binary voxel grid with values above threshold
        """
        result = VoxelGrid(self.resolution, self.bounds, dtype=np.bool_)
        result.grid = self.grid > threshold_value
        return result
    
    def to_point_cloud(self) -> Optional[object]:
        """
        Convert the voxel grid to a point cloud.
        
        Returns:
            object: An Open3D point cloud object or None if Open3D is not available
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available for point cloud conversion")
            return None
            
        # Find non-zero voxels
        indices = np.where(self.grid > 0)
        
        if len(indices[0]) == 0:
            # No points to convert
            return o3d.geometry.PointCloud()
            
        # Collect points and colors
        points = []
        colors = []
        
        # Find max value for normalization
        max_value = np.max(self.grid)
        
        for i, j, k in zip(indices[0], indices[1], indices[2]):
            # Convert voxel indices to world coordinates
            point = self.voxel_to_world((i, j, k))
            points.append(point)
            
            # Create a color based on voxel value (red intensity)
            normalized_value = self.grid[i, j, k] / max_value
            colors.append([normalized_value, 0, 1-normalized_value])  # Red to blue gradient
            
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd
    
    def get_object_centers(self, threshold: float, min_cluster_size: int = 10) -> List[np.ndarray]:
        """
        Identify distinct objects as clusters and return their centers.
        
        Args:
            threshold (float): Value threshold for considering voxels
            min_cluster_size (int): Minimum number of voxels to form a cluster
            
        Returns:
            List[np.ndarray]: List of object center points
        """
        # Create binary grid of voxels above threshold
        binary_grid = self.grid > threshold
        
        # Find connected components (objects)
        try:
            from scipy import ndimage
            
            # Label connected regions
            labeled_array, num_features = ndimage.label(binary_grid)
            
            # Calculate properties for each region
            objects = []
            for label in range(1, num_features + 1):
                # Get indices of voxels in this object
                object_indices = np.where(labeled_array == label)
                
                # Skip small clusters
                if len(object_indices[0]) < min_cluster_size:
                    continue
                    
                # Calculate center as weighted average of voxel positions
                total_weight = 0
                weighted_position = np.zeros(3)
                
                for i, j, k in zip(object_indices[0], object_indices[1], object_indices[2]):
                    weight = self.grid[i, j, k]
                    position = self.voxel_to_world((i, j, k))
                    
                    weighted_position += weight * position
                    total_weight += weight
                
                if total_weight > 0:
                    center = weighted_position / total_weight
                    objects.append(center)
            
            return objects
        except ImportError:
            logger.warning("SciPy not available for connected component analysis")
            return []