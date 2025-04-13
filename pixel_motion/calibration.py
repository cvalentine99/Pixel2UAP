"""
Camera calibration utilities for Pixel Motion Voxel Projection.

This module provides tools for calibrating cameras, determining their
positions and orientations, and synchronizing multiple cameras temporally.
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Optional, Union, Any
import os
from datetime import datetime
import threading
import queue

from .data_structures import CameraInfo

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


class CameraCalibrator:
    """
    Tools for calibrating cameras for use with the Pixel Motion Voxel Projection system.
    
    Provides methods to:
    1. Determine camera intrinsic parameters (focal length, optical center)
    2. Find camera position and orientation in 3D space
    3. Synchronize multiple cameras temporally
    4. Correct for orientation errors
    
    Attributes:
        chessboard_size (Tuple[int, int]): Number of inner corners in the calibration chessboard
        square_size (float): Physical size of the chessboard squares in meters
        calibrated_cameras (Dict[str, CameraInfo]): Dictionary of calibrated cameras
    """
    
    def __init__(self, 
                chessboard_size: Tuple[int, int] = (9, 6),
                square_size: float = 0.025):  # Default: 2.5cm squares
        """
        Initialize the camera calibrator.
        
        Args:
            chessboard_size (Tuple[int, int]): Number of inner corners in the calibration chessboard
            square_size (float): Physical size of the chessboard squares in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.calibrated_cameras = {}
        self.reference_points = None
        self.sync_timestamps = {}
        
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Camera calibration capabilities will be limited.")
    
    def calibrate_camera_intrinsics(self,
                                   images: List[np.ndarray],
                                   camera_name: str) -> Optional[CameraInfo]:
        """
        Calibrate camera intrinsic parameters using a chessboard pattern.
        
        Args:
            images (List[np.ndarray]): List of images containing chessboard pattern
            camera_name (str): Name identifier for the camera
            
        Returns:
            Optional[CameraInfo]: Calibrated camera info if successful, None otherwise
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for camera calibration")
            return None
            
        # Prepare object points (corners in 3D space)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scale to real-world coordinates
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Get image dimensions from first image
        if not images:
            logger.error("No images provided for calibration")
            return None
            
        img_shape = images[0].shape[:2]
        
        # Process each calibration image
        successful_detections = 0
        
        for img in images:
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.chessboard_size, 
                None,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # If corners found, refine them and add to our lists
            if ret:
                objpoints.append(objp)
                
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                successful_detections += 1
                
                # Optional visualization of detected corners
                # img_with_corners = img.copy()
                # cv2.drawChessboardCorners(img_with_corners, self.chessboard_size, corners2, ret)
                # cv2.imshow(f'Corners - Image {len(imgpoints)}', img_with_corners)
                # cv2.waitKey(500)
        
        logger.info(f"Successfully detected chessboard in {successful_detections}/{len(images)} images")
        
        if successful_detections < 3:
            logger.error("Insufficient successful chessboard detections for calibration")
            return None
            
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        if not ret:
            logger.error("Camera calibration failed")
            return None
            
        # Get optimal camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, img_shape, 1, img_shape
        )
        
        # Calculate re-projection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
            
        mean_error /= len(objpoints)
        logger.info(f"Re-projection error: {mean_error}")
        
        # Calculate FOV from the camera matrix
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        fov_x = 2 * np.arctan(img_shape[1] / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(img_shape[0] / (2 * fy)) * 180 / np.pi
        
        # Create CameraInfo object (with placeholder position and orientation)
        camera_info = CameraInfo(
            position=np.zeros(3),
            orientation=np.eye(3),  # Identity matrix as placeholder
            intrinsic_matrix=mtx,
            distortion_coeffs=dist,
            resolution=(img_shape[1], img_shape[0]),  # width, height
            fov=(fov_x, fov_y),
            name=camera_name
        )
        
        # Store calibration
        self.calibrated_cameras[camera_name] = camera_info
        
        return camera_info
    
    def calibrate_camera_extrinsics(self,
                                   reference_points: np.ndarray,
                                   image_points: np.ndarray,
                                   camera_name: str) -> bool:
        """
        Determine camera position and orientation using known 3D reference points.
        
        Args:
            reference_points (np.ndarray): 3D coordinates of reference points
            image_points (np.ndarray): 2D coordinates of reference points in the image
            camera_name (str): Name of the camera to calibrate
            
        Returns:
            bool: True if calibration was successful
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for extrinsic calibration")
            return False
            
        if camera_name not in self.calibrated_cameras:
            logger.error(f"Camera '{camera_name}' not found in calibrated cameras")
            return False
            
        camera = self.calibrated_cameras[camera_name]
        
        # Use solvePnP to get rotation and translation
        ret, rvec, tvec = cv2.solvePnP(
            reference_points, 
            image_points, 
            camera.intrinsic_matrix, 
            camera.distortion_coeffs
        )
        
        if not ret:
            logger.error("Failed to determine camera pose")
            return False
            
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Update camera info
        camera.position = tvec.reshape(3)
        camera.orientation = rotation_matrix
        
        # Save reference points for later use
        self.reference_points = reference_points
        
        logger.info(f"Camera '{camera_name}' extrinsic calibration completed")
        return True
    
    def calibrate_from_chessboard(self,
                                 images: List[np.ndarray],
                                 camera_name: str,
                                 use_as_reference: bool = False) -> Optional[CameraInfo]:
        """
        Perform both intrinsic and extrinsic calibration using a chessboard.
        
        Args:
            images (List[np.ndarray]): Images containing chessboard for calibration
            camera_name (str): Name of the camera
            use_as_reference (bool): Whether to use this camera as the origin
            
        Returns:
            Optional[CameraInfo]: Calibrated camera info if successful
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for chessboard calibration")
            return None
            
        # First calibrate intrinsics
        camera_info = self.calibrate_camera_intrinsics(images, camera_name)
        if camera_info is None:
            return None
            
        # Use the last image for extrinsic calibration
        last_image = images[-1]
        if len(last_image.shape) == 3:
            gray = cv2.cvtColor(last_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = last_image
            
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size, 
            None,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not ret:
            logger.error("Could not find chessboard corners for extrinsic calibration")
            return camera_info  # Return with only intrinsic calibration
            
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Create object points (chessboard corners in 3D space)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scale to real-world coordinates
        
        # If this is the reference camera, set its position at origin and orientation as identity
        if use_as_reference:
            camera_info.position = np.zeros(3)
            camera_info.orientation = np.eye(3)
            self.reference_points = objp  # Use chessboard corners as reference points
        else:
            # Otherwise, calculate extrinsics relative to the reference points
            if self.reference_points is None:
                logger.warning("No reference points available. Using chessboard corners as reference in world coordinates.")
                self.reference_points = objp
                
            # Calculate camera pose relative to reference points
            ret, rvec, tvec = cv2.solvePnP(
                self.reference_points, 
                corners2, 
                camera_info.intrinsic_matrix, 
                camera_info.distortion_coeffs
            )
            
            if not ret:
                logger.error("Failed to determine camera pose")
                return camera_info  # Return with only intrinsic calibration
                
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Update camera info
            camera_info.position = tvec.reshape(3)
            camera_info.orientation = rotation_matrix
        
        logger.info(f"Camera '{camera_name}' fully calibrated (intrinsics and extrinsics)")
        return camera_info
    
    def save_calibration(self, filename: str) -> bool:
        """
        Save calibration data to a file.
        
        Args:
            filename (str): Path to save calibration data
            
        Returns:
            bool: True if successful
        """
        try:
            calibration_data = {}
            
            for name, camera in self.calibrated_cameras.items():
                calibration_data[name] = {
                    'position': camera.position.tolist(),
                    'orientation': camera.orientation.tolist(),
                    'intrinsic_matrix': camera.intrinsic_matrix.tolist(),
                    'distortion_coeffs': camera.distortion_coeffs.tolist(),
                    'resolution': camera.resolution,
                    'fov': camera.fov,
                    'name': camera.name
                }
                
            if self.reference_points is not None:
                calibration_data['reference_points'] = self.reference_points.tolist()
                
            # Save as numpy file
            np.save(filename, calibration_data)
            logger.info(f"Calibration data saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """
        Load calibration data from a file.
        
        Args:
            filename (str): Path to saved calibration data
            
        Returns:
            bool: True if successful
        """
        try:
            # Load numpy file
            calibration_data = np.load(filename, allow_pickle=True).item()
            
            # Clear existing calibration
            self.calibrated_cameras = {}
            
            # Load reference points if available
            if 'reference_points' in calibration_data:
                self.reference_points = np.array(calibration_data['reference_points'])
                
            # Load camera calibrations
            for name, data in calibration_data.items():
                if name == 'reference_points':
                    continue
                    
                camera_info = CameraInfo(
                    position=np.array(data['position']),
                    orientation=np.array(data['orientation']),
                    intrinsic_matrix=np.array(data['intrinsic_matrix']),
                    distortion_coeffs=np.array(data['distortion_coeffs']),
                    resolution=tuple(data['resolution']),
                    fov=tuple(data['fov']),
                    name=data['name']
                )
                
                self.calibrated_cameras[name] = camera_info
                
            logger.info(f"Loaded calibration data for {len(self.calibrated_cameras)} cameras from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")
            return False
    
    def undistort_image(self, image: np.ndarray, camera_name: str) -> Optional[np.ndarray]:
        """
        Remove lens distortion from an image.
        
        Args:
            image (np.ndarray): Distorted input image
            camera_name (str): Name of the camera (must be calibrated)
            
        Returns:
            Optional[np.ndarray]: Undistorted image, or None if failed
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for image undistortion")
            return None
            
        if camera_name not in self.calibrated_cameras:
            logger.error(f"Camera '{camera_name}' not found in calibrated cameras")
            return None
            
        camera = self.calibrated_cameras[camera_name]
        
        # Get optimal camera matrix
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera.intrinsic_matrix, 
            camera.distortion_coeffs, 
            (w, h), 
            1, 
            (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(
            image, 
            camera.intrinsic_matrix, 
            camera.distortion_coeffs, 
            None, 
            newcameramtx
        )
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def synchronize_cameras(self, 
                           streams: Dict[str, Any], 
                           sync_frames: int = 10,
                           timeout: float = 30.0) -> bool:
        """
        Synchronize multiple cameras temporally using visual cues.
        
        Uses a sudden change (like a flash or object movement) to align camera timestreams.
        
        Args:
            streams (Dict[str, Any]): Dictionary of camera stream objects
            sync_frames (int): Number of frames to analyze for synchronization
            timeout (float): Maximum time to wait for synchronization in seconds
            
        Returns:
            bool: True if successful
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for camera synchronization")
            return False
            
        if len(streams) < 2:
            logger.warning("Need at least 2 cameras for synchronization")
            return False
            
        logger.info(f"Attempting to synchronize {len(streams)} cameras")
        
        # Setup data structures for synchronization
        sync_data = {}
        frame_diffs = {}
        
        for name in streams:
            sync_data[name] = {
                'frames': [],
                'timestamps': [],
                'diffs': []
            }
            
        # Prepare background frame cache
        background_frames = {}
        
        # Function to process frames from a single camera
        def process_camera_stream(stream, name, shared_data, event_flag):
            try:
                prev_frame = None
                start_time = time.time()
                
                while time.time() - start_time < timeout and not event_flag.is_set():
                    ret, frame = stream.read()
                    if not ret:
                        logger.warning(f"Failed to read frame from camera '{name}'")
                        break
                        
                    timestamp = time.time()
                    
                    # Convert to grayscale for analysis
                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame.copy()
                        
                    # First frame becomes background
                    if prev_frame is None:
                        prev_frame = gray
                        background_frames[name] = gray.copy()
                        continue
                        
                    # Calculate frame difference
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    diff_sum = np.sum(frame_diff)
                    
                    # Store frame data
                    with shared_data_lock:
                        shared_data[name]['frames'].append(frame)
                        shared_data[name]['timestamps'].append(timestamp)
                        shared_data[name]['diffs'].append(diff_sum)
                        
                        if len(shared_data[name]['diffs']) > sync_frames:
                            # Keep only the most recent frames
                            shared_data[name]['frames'] = shared_data[name]['frames'][-sync_frames:]
                            shared_data[name]['timestamps'] = shared_data[name]['timestamps'][-sync_frames:]
                            shared_data[name]['diffs'] = shared_data[name]['diffs'][-sync_frames:]
                    
                    # Update previous frame
                    prev_frame = gray
                    
                    # Check if we have enough data from all cameras
                    if all(len(shared_data[cam]['diffs']) >= sync_frames for cam in shared_data):
                        event_flag.set()
            
            except Exception as e:
                logger.error(f"Error in camera stream '{name}': {e}")
        
        # Create shared event flag for threads
        sync_complete = threading.Event()
        shared_data_lock = threading.Lock()
        
        # Start processing threads for each camera
        threads = []
        for name, stream in streams.items():
            thread = threading.Thread(
                target=process_camera_stream,
                args=(stream, name, sync_data, sync_complete)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
        # Wait for synchronization or timeout
        sync_complete.wait(timeout)
        
        # Stop all threads
        for thread in threads:
            thread.join(1.0)  # Wait up to 1 second for each thread
            
        # Analyze the collected data for synchronization
        if not sync_complete.is_set():
            logger.error("Synchronization timed out")
            return False
            
        # Calculate frame differences for each camera
        for name in sync_data:
            diffs = np.array(sync_data[name]['diffs'])
            timestamps = np.array(sync_data[name]['timestamps'])
            
            # Find the frame with maximum motion
            max_diff_idx = np.argmax(diffs)
            sync_time = timestamps[max_diff_idx]
            
            self.sync_timestamps[name] = sync_time
            
        # Calculate time offsets relative to the first camera
        reference_camera = list(sync_data.keys())[0]
        reference_time = self.sync_timestamps[reference_camera]
        
        for name in self.sync_timestamps:
            offset = self.sync_timestamps[name] - reference_time
            logger.info(f"Camera '{name}' time offset: {offset:.6f} seconds")
            
        return True
    
    def correct_orientation(self, camera_name: str, 
                           reference_points: np.ndarray,
                           measured_points: np.ndarray) -> bool:
        """
        Correct camera orientation errors using reference points.
        
        Args:
            camera_name (str): Name of the camera to correct
            reference_points (np.ndarray): True 3D coordinates of reference points
            measured_points (np.ndarray): Measured 3D coordinates of the same points
            
        Returns:
            bool: True if correction was successful
        """
        if camera_name not in self.calibrated_cameras:
            logger.error(f"Camera '{camera_name}' not found in calibrated cameras")
            return False
            
        if len(reference_points) != len(measured_points) or len(reference_points) < 3:
            logger.error("Need at least 3 matching reference and measured points")
            return False
            
        camera = self.calibrated_cameras[camera_name]
        
        try:
            # Calculate optimal rotation and translation using Kabsch algorithm
            # Center the points
            ref_centroid = np.mean(reference_points, axis=0)
            measured_centroid = np.mean(measured_points, axis=0)
            
            ref_centered = reference_points - ref_centroid
            measured_centered = measured_points - measured_centroid
            
            # Compute covariance matrix
            H = measured_centered.T @ ref_centered
            
            # Singular value decomposition
            U, _, Vt = np.linalg.svd(H)
            
            # Compute rotation matrix
            R = U @ Vt
            
            # Handle reflection case
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
                
            # Compute translation
            t = ref_centroid - R @ measured_centroid
            
            # Update camera parameters
            camera.orientation = R @ camera.orientation
            camera.position = R @ camera.position + t
            
            # Calculate and report error
            corrected_points = np.array([R @ p + t for p in measured_points])
            error = np.mean(np.linalg.norm(corrected_points - reference_points, axis=1))
            
            logger.info(f"Camera '{camera_name}' orientation corrected. Mean error: {error:.6f} meters")
            return True
            
        except Exception as e:
            logger.error(f"Error correcting camera orientation: {e}")
            return False
    
    def detect_aruco_markers(self, image: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
        """
        Detect ArUco markers in an image.
        
        ArUco markers are useful for easy point detection and camera calibration.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[List[int], List[np.ndarray]]: Marker IDs and corner coordinates
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for ArUco marker detection")
            return [], []
            
        # Check if the aruco module is available
        if not hasattr(cv2, 'aruco'):
            logger.error("OpenCV ArUco module not available")
            return [], []
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Define ArUco dictionary
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()
        
        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # Return marker IDs and corners
        if ids is None:
            return [], []
            
        return ids.flatten().tolist(), corners
    
    def calibrate_with_aruco(self, 
                            images: List[np.ndarray],
                            camera_name: str,
                            marker_size: float = 0.05,
                            marker_positions: Dict[int, np.ndarray] = None) -> Optional[CameraInfo]:
        """
        Calibrate a camera using ArUco markers.
        
        Args:
            images (List[np.ndarray]): Images containing ArUco markers
            camera_name (str): Name of the camera to calibrate
            marker_size (float): Physical size of ArUco markers in meters
            marker_positions (Dict[int, np.ndarray]): Known 3D positions of markers by ID
            
        Returns:
            Optional[CameraInfo]: Calibrated camera info if successful
        """
        if not OPENCV_AVAILABLE or not hasattr(cv2, 'aruco'):
            logger.error("OpenCV with ArUco module required for this calibration method")
            return None
            
        if not images:
            logger.error("No images provided for calibration")
            return None
            
        # Get image dimensions
        img_shape = images[0].shape[:2]
        
        # Arrays to store object points and image points
        objpoints = []
        imgpoints = []
        
        # Process each calibration image
        for img in images:
            # Detect ArUco markers
            ids, corners = self.detect_aruco_markers(img)
            
            if not ids:
                continue
                
            # If we have known marker positions
            if marker_positions is not None:
                for i, marker_id in enumerate(ids):
                    if marker_id in marker_positions:
                        # Add the marker's 3D position
                        marker_obj_pts = self._create_aruco_object_points(marker_positions[marker_id], marker_size)
                        objpoints.append(marker_obj_pts)
                        
                        # Add the marker's image corners
                        imgpoints.append(corners[i].reshape(4, 2))
            else:
                # Use markers in a planar configuration
                for i in range(len(ids)):
                    # Create object points for this marker (assuming Z=0 plane)
                    marker_obj_pts = np.zeros((4, 3), dtype=np.float32)
                    marker_obj_pts[:, :2] = np.array([
                        [0, 0],
                        [marker_size, 0],
                        [marker_size, marker_size],
                        [0, marker_size]
                    ])
                    
                    objpoints.append(marker_obj_pts)
                    imgpoints.append(corners[i].reshape(4, 2))
        
        if not objpoints:
            logger.error("No markers detected in the provided images")
            return None
            
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        if not ret:
            logger.error("Camera calibration failed")
            return None
            
        # Calculate re-projection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
            
        mean_error /= len(objpoints)
        logger.info(f"Re-projection error: {mean_error}")
        
        # Calculate FOV
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        fov_x = 2 * np.arctan(img_shape[1] / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(img_shape[0] / (2 * fy)) * 180 / np.pi
        
        # Create CameraInfo object (with placeholder position and orientation)
        camera_info = CameraInfo(
            position=np.zeros(3),
            orientation=np.eye(3),
            intrinsic_matrix=mtx,
            distortion_coeffs=dist,
            resolution=(img_shape[1], img_shape[0]),
            fov=(fov_x, fov_y),
            name=camera_name
        )
        
        # Determine extrinsics if marker positions are known and consistent
        if marker_positions is not None and len(ids) >= 3:
            # Use the last image for extrinsic calibration
            last_image = images[-1]
            last_ids, last_corners = self.detect_aruco_markers(last_image)
            
            if last_ids:
                # Collect 3D-2D correspondences
                obj_pts = []
                img_pts = []
                
                for i, marker_id in enumerate(last_ids):
                    if marker_id in marker_positions:
                        marker_obj_pts = marker_positions[marker_id]
                        # Use center of marker for simplicity
                        obj_pts.append(marker_obj_pts)
                        
                        # Calculate center of marker in image
                        corner = last_corners[i].reshape(4, 2)
                        center = np.mean(corner, axis=0)
                        img_pts.append(center)
                
                if len(obj_pts) >= 3:
                    # Calculate camera pose
                    obj_pts = np.array(obj_pts, dtype=np.float32)
                    img_pts = np.array(img_pts, dtype=np.float32)
                    
                    ret, rvec, tvec = cv2.solvePnP(
                        obj_pts, 
                        img_pts, 
                        mtx, 
                        dist
                    )
                    
                    if ret:
                        # Convert rotation vector to matrix
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        
                        # Update camera info
                        camera_info.position = tvec.reshape(3)
                        camera_info.orientation = rotation_matrix
        
        # Store calibration
        self.calibrated_cameras[camera_name] = camera_info
        
        return camera_info
    
    def _create_aruco_object_points(self, center: np.ndarray, size: float) -> np.ndarray:
        """
        Create 3D object points for an ArUco marker.
        
        Args:
            center (np.ndarray): Center position of the marker
            size (float): Physical size of the marker
            
        Returns:
            np.ndarray: 3D coordinates of marker corners
        """
        half_size = size / 2
        
        # Corners relative to center
        corners = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0]
        ], dtype=np.float32)
        
        # Translate to center position
        return corners + center