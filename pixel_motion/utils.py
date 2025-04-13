"""
Utility functions for UAP Detection and Analysis (Pixel2UAP).

This module provides specialized utilities for:
1. Statistical filtering of UAP motion signatures
2. Tracking detected UAP phenomena over time
3. Flight trajectory analysis and classification
4. Exporting UAP detection data and analysis results
"""

import numpy as np
import logging
import time
import os
from typing import List, Tuple, Dict, Optional, Union, Any
from datetime import datetime
import json

from .data_structures import VoxelGrid, CameraInfo, MotionMap

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MotionFilter:
    """
    Advanced statistical filtering for UAP motion signatures.
    
    Provides specialized methods to:
    1. Apply temporal filtering to isolate genuine UAP signatures from noise
    2. Filter motion based on size, shape, and intensity characteristics of UAPs
    3. Perform advanced background subtraction to highlight anomalous aerial phenomena
    4. Enhance subtle UAP motion signatures for improved detection
    
    Attributes:
        background_model: Adaptive background model for anomaly detection
        history_length (int): Number of frames to maintain for temporal pattern analysis
        motion_history (List): Recent UAP motion signature maps
    """
    
    def __init__(self, history_length: int = 10):
        """
        Initialize the motion filter.
        
        Args:
            history_length (int): Number of frames to keep for temporal filtering
        """
        self.history_length = history_length
        self.motion_history = []
        self.background_model = None
        self.background_subtractor = None
        
        # Initialize background subtractor if OpenCV is available
        if OPENCV_AVAILABLE:
            try:
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=False
                )
            except Exception as e:
                logger.warning(f"Failed to initialize background subtractor: {e}")
    
    def apply_temporal_filter(self, motion_map: MotionMap) -> MotionMap:
        """
        Apply temporal filtering to isolate genuine UAP signatures from noise.
        
        This method combines multiple frames of motion data with weighted averaging
        to enhance consistent UAP motion patterns while suppressing random noise.
        
        Args:
            motion_map (MotionMap): Current UAP motion map
            
        Returns:
            MotionMap: Filtered UAP motion map with enhanced signatures
        """
        # Add to history
        self.motion_history.append(motion_map)
        
        # Trim history if needed
        if len(self.motion_history) > self.history_length:
            self.motion_history = self.motion_history[-self.history_length:]
            
        # If we don't have enough history yet, return the original
        if len(self.motion_history) < 3:
            return motion_map
        
        # Get data type and shape from original motion map
        original_dtype = motion_map.data.dtype
        
        # Create a new motion map with temporal filtering using float32 for calculations
        filtered_data = np.zeros_like(motion_map.data, dtype=np.float32)
        
        # Weight recent frames more heavily
        weights = np.linspace(0.5, 1.0, len(self.motion_history))
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Combine weighted motion data
        for i, hist_map in enumerate(self.motion_history):
            # Convert data to float32 for calculations to avoid overflow
            hist_data = hist_map.data.astype(np.float32)
            filtered_data += hist_data * weights[i]
        
        # Convert back to original data type with proper clipping to avoid overflow
        if np.issubdtype(original_dtype, np.integer):
            # For integer types, clip to valid range
            info = np.iinfo(original_dtype)
            filtered_data = np.clip(filtered_data, info.min, info.max)
            filtered_data = filtered_data.astype(original_dtype)
        else:
            # For float types, just convert
            filtered_data = filtered_data.astype(original_dtype)
            
        # Create new motion map with filtered data
        return MotionMap(
            data=filtered_data,
            threshold=motion_map.threshold,
            timestamp=motion_map.timestamp,
            camera_info=motion_map.camera_info
        )
    
    def filter_by_size(self, motion_map: MotionMap, 
                      min_size: int = 10, 
                      max_size: int = 1000) -> MotionMap:
        """
        Filter UAP motion signatures based on size characteristics.
        
        This method identifies potential UAP signatures by analyzing the size of motion
        regions, filtering out both small noise and large background motion to isolate
        objects of interest that match typical UAP size profiles.
        
        Args:
            motion_map (MotionMap): UAP motion map to filter
            min_size (int): Minimum size (in pixels) of UAP motion signatures
            max_size (int): Maximum size of UAP motion signatures
            
        Returns:
            MotionMap: Filtered motion map with isolated UAP signatures
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV required for UAP size filtering")
            return motion_map
            
        # Store original data type
        original_dtype = motion_map.data.dtype
            
        # Create binary mask of motion above threshold - always use uint8 for mask
        motion_mask = (motion_map.data > motion_map.threshold).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask)
        
        # Create mask for components within size range (ignore label 0 which is background)
        valid_sizes = np.logical_and(
            stats[1:, cv2.CC_STAT_AREA] >= min_size,
            stats[1:, cv2.CC_STAT_AREA] <= max_size
        )
        
        # Create new mask with only valid components
        filtered_mask = np.zeros_like(motion_mask)
        for i, valid in enumerate(valid_sizes, 1):  # Start from 1 to skip background
            if valid:
                filtered_mask[labels == i] = 255
                
        # Apply mask to motion map with proper type handling
        # Convert to float32 for the operation
        filtered_data = motion_map.data.astype(np.float32).copy()
        # Zero out non-UAP regions
        filtered_data[filtered_mask == 0] = 0
        
        # Convert back to original type with proper clipping
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            filtered_data = np.clip(filtered_data, info.min, info.max)
            filtered_data = filtered_data.astype(original_dtype)
        else:
            filtered_data = filtered_data.astype(original_dtype)
        
        # Create new motion map with filtered data
        return MotionMap(
            data=filtered_data,
            threshold=motion_map.threshold,
            timestamp=motion_map.timestamp,
            camera_info=motion_map.camera_info
        )
    
    def apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to isolate moving objects.
        
        Args:
            frame (np.ndarray): Current video frame
            
        Returns:
            np.ndarray: Foreground mask
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV required for background subtraction")
            return np.zeros_like(frame)
            
        if self.background_subtractor is None:
            logger.warning("Background subtractor not initialized")
            return np.zeros_like(frame)
            
        # Apply background subtraction
        fgmask = self.background_subtractor.apply(frame)
        
        # Optional morphological operations to clean up mask
        if len(fgmask.shape) == 2:  # If mask is grayscale
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            
        return fgmask
    
    def filter_by_velocity(self, motion_maps: List[MotionMap], 
                         min_velocity: float = 1.0,
                         max_velocity: float = 100.0) -> MotionMap:
        """
        Filter UAP motion signatures based on characteristic velocity profiles.
        
        This specialized method analyzes the estimated velocity of detected motion
        to isolate potential UAP signatures based on their movement speed. This helps
        distinguish genuine UAP phenomena from mundane objects based on their
        atypical velocity characteristics.
        
        Args:
            motion_maps (List[MotionMap]): Sequence of UAP motion maps for velocity calculation
            min_velocity (float): Minimum UAP velocity in pixels per second
            max_velocity (float): Maximum UAP velocity in pixels per second
            
        Returns:
            MotionMap: Filtered motion map with velocity-filtered UAP signatures
        """
        if len(motion_maps) < 2:
            logger.warning("At least 2 motion maps required for UAP velocity filtering")
            return motion_maps[-1] if motion_maps else None
            
        # Get most recent maps and their timestamps
        current_map = motion_maps[-1]
        previous_map = motion_maps[-2]
        
        # Store original data type
        original_dtype = current_map.data.dtype
        
        time_diff = current_map.timestamp - previous_map.timestamp
        if time_diff <= 0:
            logger.warning("Invalid time difference between UAP motion maps")
            return current_map
            
        # Get motion pixels for both maps
        current_pixels = current_map.get_motion_pixels()
        previous_pixels = previous_map.get_motion_pixels()
        
        if len(current_pixels) == 0 or len(previous_pixels) == 0:
            return current_map
            
        # Calculate velocities using nearest neighbor matching
        from scipy.spatial import cKDTree
        
        # Build KD-tree for previous pixels
        tree = cKDTree(previous_pixels)
        
        # Find nearest neighbors for current pixels
        distances, indices = tree.query(current_pixels, k=1)
        
        # Calculate velocities in pixels per second
        velocities = distances / time_diff
        
        # Create mask for pixels with valid velocities
        valid_velocities = np.logical_and(
            velocities >= min_velocity,
            velocities <= max_velocity
        )
        
        # Create filtered motion data with proper type handling
        filtered_data = np.zeros_like(current_map.data, dtype=np.float32)
        
        for i, valid in enumerate(valid_velocities):
            if valid:
                x, y = current_pixels[i]
                if 0 <= x < filtered_data.shape[1] and 0 <= y < filtered_data.shape[0]:
                    filtered_data[y, x] = float(current_map.data[y, x])
        
        # Convert back to original type with proper clipping
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            filtered_data = np.clip(filtered_data, info.min, info.max)
            filtered_data = filtered_data.astype(original_dtype)
        else:
            filtered_data = filtered_data.astype(original_dtype)
        
        # Create new motion map with filtered data
        return MotionMap(
            data=filtered_data,
            threshold=current_map.threshold,
            timestamp=current_map.timestamp,
            camera_info=current_map.camera_info
        )
    
    def update_background_model(self, frame: np.ndarray, 
                               learning_rate: float = 0.01) -> None:
        """
        Update the background model with a new frame.
        
        Args:
            frame (np.ndarray): New frame for background model update
            learning_rate (float): Rate at which to update the model (0-1)
        """
        # Initialize background model if needed
        if self.background_model is None:
            self.background_model = frame.copy().astype(np.float32)
        else:
            # Update background model
            cv2.accumulateWeighted(
                frame,
                self.background_model,
                learning_rate
            )


class ObjectTracker:
    """
    Track detected UAP phenomena across frames.
    
    Provides specialized methods to:
    1. Associate detected UAPs between frames using advanced matching algorithms
    2. Estimate UAP flight trajectories and velocities with high precision
    3. Classify UAP flight patterns and behavior characteristics
    4. Maintain temporal consistency for UAP identification
    
    Attributes:
        tracked_objects (List): Currently tracked UAP phenomena
        history (Dict): History of previously tracked UAPs
        next_id (int): Next unique ID to assign to a newly detected UAP
        max_disappeared (int): Maximum number of frames a UAP can disappear before being considered a new object
    """
    
    class TrackedObject:
        """Class representing a tracked UAP phenomenon with complete flight trajectory data."""
        
        def __init__(self, object_id: int, position: np.ndarray, timestamp: float):
            """
            Initialize a tracked UAP.
            
            Args:
                object_id (int): Unique identifier for the UAP
                position (np.ndarray): Initial 3D position (x, y, z)
                timestamp (float): Timestamp of first detection
            """
            self.id = object_id
            self.positions = [position]
            self.timestamps = [timestamp]
            self.disappeared = 0
            self.velocity = np.zeros(3)
            self.classification = None
            
        def update(self, position: np.ndarray, timestamp: float) -> None:
            """
            Update object with new position.
            
            Args:
                position (np.ndarray): New 3D position
                timestamp (float): Timestamp of update
            """
            # Calculate velocity if we have previous positions
            if len(self.timestamps) > 0:
                time_diff = timestamp - self.timestamps[-1]
                if time_diff > 0:
                    pos_diff = position - self.positions[-1]
                    self.velocity = pos_diff / time_diff
            
            # Add new data
            self.positions.append(position)
            self.timestamps.append(timestamp)
            self.disappeared = 0
            
        def predict_position(self, timestamp: float) -> np.ndarray:
            """
            Predict object position at a future timestamp.
            
            Args:
                timestamp (float): Future timestamp
                
            Returns:
                np.ndarray: Predicted 3D position
            """
            if len(self.timestamps) < 2:
                return self.positions[-1]
                
            # Time since last update
            time_diff = timestamp - self.timestamps[-1]
            
            # Predict using current velocity
            return self.positions[-1] + self.velocity * time_diff
            
        def get_trajectory(self) -> np.ndarray:
            """
            Get the object's trajectory as a sequence of positions.
            
            Returns:
                np.ndarray: Array of 3D positions
            """
            return np.array(self.positions)
    
    def __init__(self, max_disappeared: int = 10, max_distance: float = 0.5):
        """
        Initialize the object tracker.
        
        Args:
            max_disappeared (int): Maximum frames an object can disappear before being removed
            max_distance (float): Maximum distance for considering object matches
        """
        self.tracked_objects = []
        self.history = {}  # Objects that are no longer tracked
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def update(self, 
              detected_objects: List[np.ndarray], 
              timestamp: float = None) -> Dict[int, np.ndarray]:
        """
        Update tracker with newly detected objects.
        
        Args:
            detected_objects (List[np.ndarray]): List of detected object positions
            timestamp (float): Current timestamp (default: current time)
            
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping object IDs to positions
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Handle case with no detections
        if not detected_objects:
            # Increment disappeared counter for all objects
            for obj in self.tracked_objects:
                obj.disappeared += 1
                
            # Remove objects that have disappeared for too long
            self.tracked_objects = [obj for obj in self.tracked_objects 
                                   if obj.disappeared <= self.max_disappeared]
                
            # Return currently tracked objects
            return {obj.id: obj.positions[-1] for obj in self.tracked_objects}
        
        # Handle case with no existing objects
        if not self.tracked_objects:
            # Create new tracked objects for all detections
            for detection in detected_objects:
                self.tracked_objects.append(
                    self.TrackedObject(self.next_id, detection, timestamp)
                )
                self.next_id += 1
                
            # Return newly tracked objects
            return {obj.id: obj.positions[-1] for obj in self.tracked_objects}
        
        # Handle case with existing objects and new detections
        
        # Predict current positions of tracked objects
        predicted_positions = np.array([obj.predict_position(timestamp) 
                                      for obj in self.tracked_objects])
        
        # Convert detections to array
        detections_array = np.array(detected_objects)
        
        # Calculate distance matrix between predictions and detections
        distances = self._calculate_distance_matrix(predicted_positions, detections_array)
        
        # Find optimal assignment using Hungarian algorithm
        used_trackers, used_detections = self._assign_detections_to_trackers(distances)
        
        # Update matched trackers
        for tracker_idx, detection_idx in zip(used_trackers, used_detections):
            self.tracked_objects[tracker_idx].update(
                detections_array[detection_idx], timestamp
            )
            
        # Handle unmatched trackers (disappeared)
        for i in range(len(self.tracked_objects)):
            if i not in used_trackers:
                self.tracked_objects[i].disappeared += 1
                
        # Handle unmatched detections (new objects)
        for i in range(len(detections_array)):
            if i not in used_detections:
                self.tracked_objects.append(
                    self.TrackedObject(self.next_id, detections_array[i], timestamp)
                )
                self.next_id += 1
                
        # Remove objects that have disappeared for too long
        expired_objects = [obj for obj in self.tracked_objects 
                         if obj.disappeared > self.max_disappeared]
        
        # Move expired objects to history
        for obj in expired_objects:
            self.history[obj.id] = obj
            
        # Update tracked objects list
        self.tracked_objects = [obj for obj in self.tracked_objects 
                              if obj.disappeared <= self.max_disappeared]
                
        # Return currently tracked objects
        return {obj.id: obj.positions[-1] for obj in self.tracked_objects}
    
    def _calculate_distance_matrix(self, 
                                 predicted_positions: np.ndarray,
                                 detected_positions: np.ndarray) -> np.ndarray:
        """
        Calculate distance matrix between predicted positions and detections.
        
        Args:
            predicted_positions (np.ndarray): Predicted positions of tracked objects
            detected_positions (np.ndarray): Positions of newly detected objects
            
        Returns:
            np.ndarray: Distance matrix
        """
        num_trackers = len(predicted_positions)
        num_detections = len(detected_positions)
        
        # Initialize distance matrix
        distance_matrix = np.zeros((num_trackers, num_detections))
        
        # Calculate Euclidean distance for each pair
        for i in range(num_trackers):
            for j in range(num_detections):
                distance_matrix[i, j] = np.linalg.norm(
                    predicted_positions[i] - detected_positions[j]
                )
                
        return distance_matrix
    
    def _assign_detections_to_trackers(self, 
                                     distance_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Assign detections to trackers using Hungarian algorithm.
        
        Args:
            distance_matrix (np.ndarray): Matrix of distances between predictions and detections
            
        Returns:
            Tuple[List[int], List[int]]: Indices of matched trackers and detections
        """
        from scipy.optimize import linear_sum_assignment
        
        # Apply maximum distance threshold
        thresholded_matrix = distance_matrix.copy()
        thresholded_matrix[thresholded_matrix > self.max_distance] = 1000000  # Large value
        
        # Use Hungarian algorithm to find optimal assignment
        tracker_indices, detection_indices = linear_sum_assignment(thresholded_matrix)
        
        # Filter out assignments with distance exceeding threshold
        valid_assignments = thresholded_matrix[tracker_indices, detection_indices] <= self.max_distance
        
        used_trackers = tracker_indices[valid_assignments].tolist()
        used_detections = detection_indices[valid_assignments].tolist()
        
        return used_trackers, used_detections
    
    def get_trajectories(self) -> Dict[int, np.ndarray]:
        """
        Get trajectories of all tracked objects.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping object IDs to trajectories
        """
        trajectories = {}
        
        # Get trajectories for active objects
        for obj in self.tracked_objects:
            trajectories[obj.id] = obj.get_trajectory()
            
        # Get trajectories for objects in history
        for obj_id, obj in self.history.items():
            trajectories[obj_id] = obj.get_trajectory()
            
        return trajectories
    
    def get_velocities(self) -> Dict[int, np.ndarray]:
        """
        Get velocities of all currently tracked objects.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary mapping object IDs to velocities
        """
        return {obj.id: obj.velocity for obj in self.tracked_objects}
    
    def classify_trajectories(self) -> Dict[int, str]:
        """
        Classify UAP flight trajectories based on characteristic movement patterns.
        
        This analysis examines UAP trajectory data to identify flight characteristics
        that may indicate technological or non-conventional propulsion systems.
        
        Returns:
            Dict[int, str]: Dictionary mapping UAP IDs to flight pattern classifications
        """
        classifications = {}
        
        for obj in self.tracked_objects:
            # Skip UAPs with too few position samples
            if len(obj.positions) < 5:
                classifications[obj.id] = "insufficient_data"
                continue
                
            # Calculate trajectory properties
            trajectory = np.array(obj.positions)
            
            # Calculate total distance traveled
            total_distance = 0
            for i in range(1, len(trajectory)):
                total_distance += np.linalg.norm(trajectory[i] - trajectory[i-1])
                
            # Calculate straight-line distance
            straight_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            
            # Linearity ratio (straight distance / total distance)
            if total_distance > 0:
                linearity = straight_distance / total_distance
            else:
                linearity = 0
                
            # Classify based on linearity and velocity
            speed = np.linalg.norm(obj.velocity)
            
            if linearity > 0.9:
                if speed > 5.0:
                    classification = "transmedium_hypersonic"  # Very fast and straight
                else:
                    classification = "ballistic_trajectory"    # Straight but normal speed
            elif linearity > 0.7:
                classification = "controlled_maneuver"         # Somewhat direct path
            elif linearity > 0.3:
                classification = "intelligent_pattern"         # Complex but purposeful path
            else:
                classification = "non_inertial_anomaly"        # Erratic movement defying physics
                
            # Store classification
            classifications[obj.id] = classification
            obj.classification = classification
            
        return classifications


class ExportManager:
    """
    Manage export of UAP detection data and analysis results.
    
    Provides specialized methods to:
    1. Export UAP detection data in various formats for further analysis
    2. Save UAP trajectory data with precise timestamps for temporal analysis
    3. Create scientific visualizations of UAP flight patterns and behavior
    4. Generate comprehensive reports for UAP research
    
    Attributes:
        output_directory (str): Directory for UAP analysis files
        enable_compression (bool): Whether to compress large UAP datasets
    """
    
    def __init__(self, output_directory: str = "output", enable_compression: bool = True):
        """
        Initialize the export manager.
        
        Args:
            output_directory (str): Directory for output files
            enable_compression (bool): Whether to compress large output files
        """
        self.output_directory = output_directory
        self.enable_compression = enable_compression
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
    
    def export_voxel_grid(self, voxel_grid: VoxelGrid, filename: str = None) -> str:
        """
        Export voxel grid data to a file.
        
        Args:
            voxel_grid (VoxelGrid): Voxel grid to export
            filename (str): Output filename (default: auto-generated)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voxel_grid_{timestamp}.npz"
            
        # Ensure filename has proper extension
        if not filename.endswith(".npz") and not filename.endswith(".npy"):
            filename += ".npz"
            
        # Create full output path
        output_path = os.path.join(self.output_directory, filename)
        
        # Save voxel grid data
        if self.enable_compression or filename.endswith(".npz"):
            # Save as compressed .npz file
            np.savez_compressed(
                output_path,
                grid=voxel_grid.grid,
                resolution=voxel_grid.resolution,
                bounds=voxel_grid.bounds
            )
        else:
            # Save as uncompressed .npy file
            np.save(
                output_path,
                {
                    'grid': voxel_grid.grid,
                    'resolution': voxel_grid.resolution,
                    'bounds': voxel_grid.bounds
                }
            )
            
        logger.info(f"Voxel grid exported to {output_path}")
        return output_path
    
    def export_point_cloud(self, voxel_grid: VoxelGrid, filename: str = None) -> str:
        """
        Export voxel grid as a point cloud file.
        
        Args:
            voxel_grid (VoxelGrid): Voxel grid to export
            filename (str): Output filename (default: auto-generated)
            
        Returns:
            str: Path to exported file
        """
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D required for point cloud export")
            return None
            
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"point_cloud_{timestamp}.ply"
            
        # Ensure filename has proper extension
        if not filename.endswith((".ply", ".pcd")):
            filename += ".ply"
            
        # Create full output path
        output_path = os.path.join(self.output_directory, filename)
        
        # Convert voxel grid to point cloud
        point_cloud = voxel_grid.to_point_cloud()
        
        if point_cloud is None:
            logger.error("Failed to convert voxel grid to point cloud")
            return None
            
        # Write point cloud to file
        try:
            o3d.io.write_point_cloud(output_path, point_cloud)
            logger.info(f"Point cloud exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error exporting point cloud: {e}")
            return None
    
    def export_detection_results(self, 
                               detected_objects: Dict[int, np.ndarray],
                               timestamps: Dict[int, float] = None,
                               filename: str = None) -> str:
        """
        Export object detection results.
        
        Args:
            detected_objects (Dict[int, np.ndarray]): Dictionary of detected objects by ID
            timestamps (Dict[int, float]): Dictionary of detection timestamps
            filename (str): Output filename (default: auto-generated)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.json"
            
        # Ensure filename has proper extension
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Create full output path
        output_path = os.path.join(self.output_directory, filename)
        
        # Prepare detection data
        detection_data = []
        
        for obj_id, position in detected_objects.items():
            detection = {
                'id': int(obj_id),
                'position': position.tolist()
            }
            
            # Add timestamp if available
            if timestamps and obj_id in timestamps:
                detection['timestamp'] = timestamps[obj_id]
                
            detection_data.append(detection)
            
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump({
                'detections': detection_data,
                'timestamp': time.time(),
                'count': len(detection_data)
            }, f, indent=2)
            
        logger.info(f"Detection results exported to {output_path}")
        return output_path
    
    def export_trajectory_data(self, trajectories: Dict[int, np.ndarray],
                             classifications: Dict[int, str] = None,
                             filename: str = None) -> str:
        """
        Export object trajectory data.
        
        Args:
            trajectories (Dict[int, np.ndarray]): Dictionary mapping object IDs to trajectories
            classifications (Dict[int, str]): Dictionary mapping object IDs to classifications
            filename (str): Output filename (default: auto-generated)
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectories_{timestamp}.json"
            
        # Ensure filename has proper extension
        if not filename.endswith(".json"):
            filename += ".json"
            
        # Create full output path
        output_path = os.path.join(self.output_directory, filename)
        
        # Prepare trajectory data
        trajectory_data = []
        
        for obj_id, trajectory in trajectories.items():
            data = {
                'id': int(obj_id),
                'trajectory': trajectory.tolist()
            }
            
            # Add classification if available
            if classifications and obj_id in classifications:
                data['classification'] = classifications[obj_id]
                
            trajectory_data.append(data)
            
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump({
                'trajectories': trajectory_data,
                'timestamp': time.time(),
                'count': len(trajectory_data)
            }, f, indent=2)
            
        logger.info(f"Trajectory data exported to {output_path}")
        return output_path
    
    def create_trajectory_visualization(self, 
                                      trajectories: Dict[int, np.ndarray],
                                      classifications: Dict[int, str] = None,
                                      filename: str = None) -> str:
        """
        Create a scientific visualization of UAP flight trajectories.
        
        This method generates a high-quality 3D visualization of UAP flight paths,
        color-coded by flight pattern classification, with markers for key events
        such as sudden accelerations, hovering, or direction changes.
        
        Args:
            trajectories (Dict[int, np.ndarray]): Dictionary mapping UAP IDs to flight trajectories
            classifications (Dict[int, str]): Dictionary mapping UAP IDs to flight pattern classifications
            filename (str): Output filename (default: auto-generated with timestamp)
            
        Returns:
            str: Path to exported UAP trajectory visualization
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for trajectory visualization")
            return None
            
        if not trajectories:
            logger.warning("No trajectories provided for visualization")
            return None
            
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_viz_{timestamp}.png"
            
        # Ensure filename has proper extension
        if not filename.endswith((".png", ".jpg", ".pdf")):
            filename += ".png"
            
        # Create full output path
        output_path = os.path.join(self.output_directory, filename)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for different UAP flight pattern classifications
        class_colors = {
            'transmedium_hypersonic': 'red',
            'ballistic_trajectory': 'blue',
            'controlled_maneuver': 'green',
            'intelligent_pattern': 'orange',
            'non_inertial_anomaly': 'purple',
            'insufficient_data': 'gray'
        }
        
        # Plot each trajectory
        for obj_id, trajectory in trajectories.items():
            # Get classification and color
            if classifications and obj_id in classifications:
                obj_class = classifications[obj_id]
                color = class_colors.get(obj_class, 'gray')
            else:
                obj_class = 'unknown'
                color = 'gray'
                
            # Plot 3D trajectory
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=2,
                label=f"ID {obj_id} ({obj_class})" if obj_id in classifications else f"ID {obj_id}"
            )
            
            # Mark start and end points
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color=color,
                marker='o',
                s=100
            )
            
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                trajectory[-1, 2],
                color=color,
                marker='s',
                s=100
            )
            
        # Add labels and legend with UAP-specific terminology
        ax.set_xlabel('X - Position (meters)')
        ax.set_ylabel('Y - Position (meters)')
        ax.set_zlabel('Z - Altitude (meters)')
        ax.set_title('UAP Flight Trajectories Analysis', fontsize=14, fontweight='bold')
        
        # Add legend for unique classes
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
                
        ax.legend(unique_handles, unique_labels)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        
        logger.info(f"Trajectory visualization saved to {output_path}")
        return output_path