"""
Interfaces for different input sources for Pixel Motion Voxel Projection.

This module provides interfaces for:
1. Processing video files
2. Handling live camera feeds
3. Working with pre-recorded data
4. Integration with other input sources
"""

import numpy as np
import logging
import time
import os
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import threading
import queue
from datetime import datetime

from .data_structures import CameraInfo, MotionMap
from .core import PixelMotionVoxelProjection

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


class PixelMotionInputInterface(ABC):
    """
    Abstract base class for pixel motion input interfaces.
    
    This class defines the common interface for different input sources
    to interact with the Pixel Motion Voxel Projection system.
    
    Attributes:
        camera_info (CameraInfo): Information about the camera
        processor (PixelMotionVoxelProjection): Voxel projection processor
        is_running (bool): Whether the interface is currently running
    """
    
    def __init__(self, 
                camera_info: CameraInfo,
                processor: Optional[PixelMotionVoxelProjection] = None):
        """
        Initialize the input interface.
        
        Args:
            camera_info (CameraInfo): Information about the camera
            processor (PixelMotionVoxelProjection): Reference to voxel projection processor
        """
        self.camera_info = camera_info
        self.processor = processor
        self.is_running = False
        self._stop_event = threading.Event()
        
        # Add camera to processor if provided
        if processor is not None:
            processor.add_camera(camera_info)
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start the input source.
        
        Returns:
            bool: True if successfully started
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """
        Stop the input source.
        
        Returns:
            bool: True if successfully stopped
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the input source.
        
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        pass
    
    def set_processor(self, processor: PixelMotionVoxelProjection) -> None:
        """
        Set the voxel projection processor.
        
        Args:
            processor (PixelMotionVoxelProjection): Voxel projection processor
        """
        self.processor = processor
        processor.add_camera(self.camera_info)
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> None:
        """
        Process a frame using the voxel projection processor.
        
        Args:
            frame (np.ndarray): Frame to process
            timestamp (float): Frame timestamp
        """
        if self.processor is None:
            logger.warning("No processor set for frame processing")
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        self.processor.process_video_frame(
            frame, 
            self.camera_info.name,
            timestamp
        )


class VideoFileInterface(PixelMotionInputInterface):
    """
    Interface for processing video files.
    
    Attributes:
        video_path (str): Path to the video file
        frame_interval (int): Process every Nth frame
        loop (bool): Whether to loop the video when it reaches the end
        capture: OpenCV VideoCapture object
    """
    
    def __init__(self, 
                camera_info: CameraInfo,
                video_path: str,
                processor: Optional[PixelMotionVoxelProjection] = None,
                frame_interval: int = 1,
                loop: bool = False):
        """
        Initialize the video file interface.
        
        Args:
            camera_info (CameraInfo): Information about the camera
            video_path (str): Path to the video file
            processor (PixelMotionVoxelProjection): Reference to voxel projection processor
            frame_interval (int): Process every Nth frame
            loop (bool): Whether to loop the video when it reaches the end
        """
        super().__init__(camera_info, processor)
        
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required for video file interface")
            
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.loop = loop
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer for 30 frames
        self.thread = None
        
        # Video properties
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        self.current_frame_idx = 0
    
    def start(self) -> bool:
        """
        Start video file processing.
        
        Returns:
            bool: True if successfully started
        """
        if self.is_running:
            logger.warning("Video file interface already running")
            return False
            
        # Open video file
        self.capture = cv2.VideoCapture(self.video_path)
        
        if not self.capture.isOpened():
            logger.error(f"Failed to open video file: {self.video_path}")
            return False
            
        # Get video properties
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Opened video file: {self.video_path}")
        logger.info(f"  - Resolution: {int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        logger.info(f"  - FPS: {self.fps}")
        logger.info(f"  - Duration: {self.duration:.2f} seconds")
        logger.info(f"  - Total frames: {self.frame_count}")
        
        # Start processing thread
        self._stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._process_video)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop(self) -> bool:
        """
        Stop video file processing.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self.is_running:
            return False
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # Release resources
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.is_running = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from the video.
        
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def _process_video(self) -> None:
        """Background thread for reading and processing video frames."""
        self.current_frame_idx = 0
        
        while not self._stop_event.is_set():
            # Read frame
            ret, frame = self.capture.read()
            
            # Handle end of video
            if not ret:
                if self.loop:
                    # Reset to beginning of video
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    continue
                else:
                    # End of video
                    logger.info("Reached end of video file")
                    break
                    
            self.current_frame_idx += 1
            
            # Skip frames according to interval
            if (self.current_frame_idx - 1) % self.frame_interval != 0:
                continue
                
            # Calculate timestamp based on frame position
            timestamp = self.current_frame_idx / self.fps if self.fps > 0 else time.time()
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait((frame, timestamp))
            except queue.Full:
                # Skip frame if queue is full
                continue
                
            # Process frame if processor is available
            if self.processor is not None:
                self.process_frame(frame, timestamp)
                
            # Control processing speed to match fps if specified
            if self.fps > 0:
                # Sleep to maintain frame rate (adjust for processing time)
                sleep_time = 1.0 / (self.fps * self.frame_interval) * 0.9  # 90% of theoretical time
                time.sleep(max(0, sleep_time))
        
        self.is_running = False
        
    def get_progress(self) -> float:
        """
        Get current playback progress.
        
        Returns:
            float: Progress as a percentage (0-100)
        """
        if self.frame_count > 0:
            return 100.0 * self.current_frame_idx / self.frame_count
        return 0.0
    
    def seek(self, position_percent: float) -> bool:
        """
        Seek to a specific position in the video.
        
        Args:
            position_percent (float): Position as percentage (0-100)
            
        Returns:
            bool: True if seek was successful
        """
        if not self.is_running or self.capture is None:
            return False
            
        if not 0 <= position_percent <= 100:
            return False
            
        # Calculate frame number
        frame_number = int((position_percent / 100.0) * self.frame_count)
        
        # Seek to frame
        ret = self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        if ret:
            self.current_frame_idx = frame_number
            
        return ret


class LiveCameraInterface(PixelMotionInputInterface):
    """
    Interface for processing live camera feeds.
    
    Attributes:
        camera_id (int): Camera device ID
        frame_interval (int): Process every Nth frame
        resolution (Tuple[int, int]): Desired camera resolution
        capture: OpenCV VideoCapture object
    """
    
    def __init__(self, 
                camera_info: CameraInfo,
                camera_id: int = 0,
                processor: Optional[PixelMotionVoxelProjection] = None,
                frame_interval: int = 1,
                resolution: Tuple[int, int] = None):
        """
        Initialize the live camera interface.
        
        Args:
            camera_info (CameraInfo): Information about the camera
            camera_id (int): Camera device ID
            processor (PixelMotionVoxelProjection): Reference to voxel projection processor
            frame_interval (int): Process every Nth frame
            resolution (Tuple[int, int]): Desired camera resolution
        """
        super().__init__(camera_info, processor)
        
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required for live camera interface")
            
        self.camera_id = camera_id
        self.frame_interval = frame_interval
        self.resolution = resolution
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=10)  # Smaller buffer for live feed
        self.thread = None
        self.frame_count = 0
    
    def start(self) -> bool:
        """
        Start live camera processing.
        
        Returns:
            bool: True if successfully started
        """
        if self.is_running:
            logger.warning("Live camera interface already running")
            return False
            
        # Open camera
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            logger.error(f"Failed to open camera: {self.camera_id}")
            return False
            
        # Set resolution if specified
        if self.resolution is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
        # Get actual camera properties
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Opened camera: {self.camera_id}")
        logger.info(f"  - Resolution: {actual_width}x{actual_height}")
        logger.info(f"  - FPS: {actual_fps}")
        
        # Start processing thread
        self._stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._process_camera)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop(self) -> bool:
        """
        Stop live camera processing.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self.is_running:
            return False
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # Release resources
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.is_running = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera.
        
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def _process_camera(self) -> None:
        """Background thread for reading and processing camera frames."""
        self.frame_count = 0
        
        while not self._stop_event.is_set():
            # Read frame
            ret, frame = self.capture.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)  # Avoid tight loop if camera is having issues
                continue
                
            self.frame_count += 1
            
            # Skip frames according to interval
            if (self.frame_count - 1) % self.frame_interval != 0:
                continue
                
            # Use current time as timestamp
            timestamp = time.time()
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait((frame, timestamp))
            except queue.Full:
                # Skip frame if queue is full (common in live feeds)
                continue
                
            # Process frame if processor is available
            if self.processor is not None:
                self.process_frame(frame, timestamp)
        
        self.is_running = False


class MultiCameraInterface:
    """
    Interface for managing multiple camera inputs.
    
    Coordinates processing across multiple cameras and ensures
    synchronization for the Pixel Motion Voxel Projection system.
    
    Attributes:
        processor (PixelMotionVoxelProjection): Voxel projection processor
        interfaces (Dict[str, PixelMotionInputInterface]): Input interfaces by name
        is_running (bool): Whether the multi-camera system is running
    """
    
    def __init__(self, processor: PixelMotionVoxelProjection):
        """
        Initialize the multi-camera interface.
        
        Args:
            processor (PixelMotionVoxelProjection): Voxel projection processor
        """
        self.processor = processor
        self.interfaces = {}
        self.is_running = False
        self._stop_event = threading.Event()
        self.processing_thread = None
    
    def add_interface(self, name: str, interface: PixelMotionInputInterface) -> bool:
        """
        Add a camera interface to the system.
        
        Args:
            name (str): Name for this interface
            interface (PixelMotionInputInterface): Camera interface
            
        Returns:
            bool: True if interface was added
        """
        if name in self.interfaces:
            logger.warning(f"Interface '{name}' already exists, will be replaced")
            
        # Set processor for the interface
        interface.set_processor(self.processor)
        
        # Add to interfaces
        self.interfaces[name] = interface
        
        return True
    
    def remove_interface(self, name: str) -> bool:
        """
        Remove a camera interface from the system.
        
        Args:
            name (str): Name of the interface to remove
            
        Returns:
            bool: True if interface was removed
        """
        if name not in self.interfaces:
            return False
            
        # Stop interface if it's running
        interface = self.interfaces[name]
        if interface.is_running:
            interface.stop()
            
        # Remove from interfaces
        del self.interfaces[name]
        
        # Remove camera from processor
        self.processor.remove_camera(interface.camera_info.name)
        
        return True
    
    def start_all(self) -> bool:
        """
        Start all camera interfaces.
        
        Returns:
            bool: True if all interfaces were started
        """
        if self.is_running:
            logger.warning("Multi-camera interface already running")
            return False
            
        if not self.interfaces:
            logger.warning("No camera interfaces to start")
            return False
            
        # Start each interface
        success = True
        for name, interface in self.interfaces.items():
            if not interface.start():
                logger.error(f"Failed to start interface: {name}")
                success = False
                
        # Start processing thread
        self._stop_event.clear()
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_all_cameras)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return success
    
    def stop_all(self) -> bool:
        """
        Stop all camera interfaces.
        
        Returns:
            bool: True if all interfaces were stopped
        """
        if not self.is_running:
            return False
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        # Stop each interface
        for name, interface in self.interfaces.items():
            if interface.is_running:
                interface.stop()
                
        self.is_running = False
        return True
    
    def _process_all_cameras(self) -> None:
        """Background thread for coordinating processing across cameras."""
        while not self._stop_event.is_set():
            # Sleep briefly to avoid tight loop
            time.sleep(0.01)
            
            # Check if any interfaces have stopped
            if not all(interface.is_running for interface in self.interfaces.values()):
                logger.warning("One or more camera interfaces have stopped")
                
            # This thread doesn't need to do much since each interface
            # processes frames in its own thread and sends them to the processor
    
    def get_frame_from_all(self) -> Dict[str, np.ndarray]:
        """
        Get the latest frame from all cameras.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping interface names to frames
        """
        frames = {}
        for name, interface in self.interfaces.items():
            if interface.is_running:
                frame = interface.get_frame()
                if frame is not None:
                    frames[name] = frame
                    
        return frames


class RecordedDataInterface(PixelMotionInputInterface):
    """
    Interface for processing pre-recorded data sets.
    
    Attributes:
        data_directory (str): Directory containing recorded data
        file_pattern (str): Pattern for matching data files
        playback_speed (float): Speed multiplier for playback
    """
    
    def __init__(self, 
                camera_info: CameraInfo,
                data_directory: str,
                processor: Optional[PixelMotionVoxelProjection] = None,
                file_pattern: str = "*.npz",
                playback_speed: float = 1.0):
        """
        Initialize the recorded data interface.
        
        Args:
            camera_info (CameraInfo): Information about the camera
            data_directory (str): Directory containing recorded data
            processor (PixelMotionVoxelProjection): Reference to voxel projection processor
            file_pattern (str): Pattern for matching data files
            playback_speed (float): Speed multiplier for playback
        """
        super().__init__(camera_info, processor)
        
        self.data_directory = data_directory
        self.file_pattern = file_pattern
        self.playback_speed = playback_speed
        self.data_files = []
        self.current_idx = 0
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=30)
    
    def start(self) -> bool:
        """
        Start processing recorded data.
        
        Returns:
            bool: True if successfully started
        """
        if self.is_running:
            logger.warning("Recorded data interface already running")
            return False
            
        # Find data files
        import glob
        self.data_files = sorted(glob.glob(os.path.join(self.data_directory, self.file_pattern)))
        
        if not self.data_files:
            logger.error(f"No data files found in {self.data_directory} matching {self.file_pattern}")
            return False
            
        logger.info(f"Found {len(self.data_files)} data files")
        
        # Start processing thread
        self._stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._process_data)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop(self) -> bool:
        """
        Stop processing recorded data.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self.is_running:
            return False
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.is_running = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from the recorded data.
        
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def _process_data(self) -> None:
        """Background thread for reading and processing recorded data."""
        self.current_idx = 0
        last_timestamp = None
        
        while not self._stop_event.is_set() and self.current_idx < len(self.data_files):
            # Load data file
            data_file = self.data_files[self.current_idx]
            
            try:
                data = np.load(data_file, allow_pickle=True)
                
                if 'frame' in data and 'timestamp' in data:
                    frame = data['frame']
                    timestamp = float(data['timestamp'])
                    
                    # Adjust playback speed
                    if last_timestamp is not None and self.playback_speed != 1.0:
                        real_time_diff = time.time() - last_timestamp
                        file_time_diff = (timestamp - float(data['timestamp'])) / self.playback_speed
                        
                        # Sleep to maintain correct playback speed
                        sleep_time = max(0, file_time_diff - real_time_diff)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            
                    last_timestamp = time.time()
                    
                    # Add to queue
                    try:
                        self.frame_queue.put_nowait((frame, timestamp))
                    except queue.Full:
                        # Skip frame if queue is full
                        pass
                        
                    # Process frame if processor is available
                    if self.processor is not None:
                        self.process_frame(frame, timestamp)
                else:
                    logger.warning(f"Data file {data_file} does not contain required fields")
                    
            except Exception as e:
                logger.error(f"Error processing data file {data_file}: {e}")
                
            self.current_idx += 1
            
            # Sleep briefly to avoid tight loop
            time.sleep(0.01)
            
        logger.info("Finished processing recorded data")
        self.is_running = False
    
    def get_progress(self) -> float:
        """
        Get current playback progress.
        
        Returns:
            float: Progress as a percentage (0-100)
        """
        if len(self.data_files) > 0:
            return 100.0 * self.current_idx / len(self.data_files)
        return 0.0
    
    def seek(self, position_percent: float) -> bool:
        """
        Seek to a specific position in the data.
        
        Args:
            position_percent (float): Position as percentage (0-100)
            
        Returns:
            bool: True if seek was successful
        """
        if not self.is_running or not self.data_files:
            return False
            
        if not 0 <= position_percent <= 100:
            return False
            
        # Calculate new index
        new_idx = int((position_percent / 100.0) * len(self.data_files))
        new_idx = max(0, min(new_idx, len(self.data_files) - 1))
        
        # Update index
        self.current_idx = new_idx
        
        return True


class CustomSourceInterface(PixelMotionInputInterface):
    """
    Interface for custom data sources.
    
    Allows integration with custom data sources by providing callback functions.
    
    Attributes:
        frame_callback (Callable): Callback function to get the next frame
        start_callback (Callable): Callback function to start the source
        stop_callback (Callable): Callback function to stop the source
    """
    
    def __init__(self, 
                camera_info: CameraInfo,
                frame_callback: Callable[[], Tuple[np.ndarray, float]],
                processor: Optional[PixelMotionVoxelProjection] = None,
                start_callback: Callable[[], bool] = None,
                stop_callback: Callable[[], bool] = None,
                interval: float = 0.033):  # Default: ~30 FPS
        """
        Initialize the custom source interface.
        
        Args:
            camera_info (CameraInfo): Information about the camera
            frame_callback (Callable): Function to get the next frame
            processor (PixelMotionVoxelProjection): Reference to voxel projection processor
            start_callback (Callable): Function to start the source
            stop_callback (Callable): Function to stop the source
            interval (float): Time interval between frames in seconds
        """
        super().__init__(camera_info, processor)
        
        self.frame_callback = frame_callback
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.interval = interval
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=30)
    
    def start(self) -> bool:
        """
        Start the custom source.
        
        Returns:
            bool: True if successfully started
        """
        if self.is_running:
            logger.warning("Custom source interface already running")
            return False
            
        # Call start callback if provided
        if self.start_callback is not None:
            if not self.start_callback():
                logger.error("Failed to start custom source")
                return False
                
        # Start processing thread
        self._stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._process_source)
        self.thread.daemon = True
        self.thread.start()
        
        return True
    
    def stop(self) -> bool:
        """
        Stop the custom source.
        
        Returns:
            bool: True if successfully stopped
        """
        if not self.is_running:
            return False
            
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        # Call stop callback if provided
        if self.stop_callback is not None:
            self.stop_callback()
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        self.is_running = False
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the custom source.
        
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def _process_source(self) -> None:
        """Background thread for processing custom source data."""
        next_frame_time = time.time()
        
        while not self._stop_event.is_set():
            current_time = time.time()
            
            # Check if it's time for the next frame
            if current_time >= next_frame_time:
                try:
                    # Get frame from callback
                    frame, timestamp = self.frame_callback()
                    
                    if frame is not None:
                        # Add to queue
                        try:
                            self.frame_queue.put_nowait((frame, timestamp))
                        except queue.Full:
                            # Skip frame if queue is full
                            pass
                            
                        # Process frame if processor is available
                        if self.processor is not None:
                            self.process_frame(frame, timestamp)
                            
                except Exception as e:
                    logger.error(f"Error getting frame from custom source: {e}")
                    
                # Calculate next frame time
                next_frame_time = current_time + self.interval
            else:
                # Sleep until next frame time
                sleep_time = next_frame_time - current_time
                time.sleep(max(0.001, sleep_time))  # At least 1ms to avoid tight loop
        
        self.is_running = False