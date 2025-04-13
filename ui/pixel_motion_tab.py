"""
Pixel2UAP Projector - UAP Detection and Tracking Module.

This module implements the Pixel Motion Voxel Projection technique specifically designed
for detecting and tracking Unidentified Aerial Phenomena (UAP) by analyzing motion 
between video frames and projecting it into 3D voxel space for trajectory analysis.

The key components include:
- Video file loading and analysis
- UAP motion detection through frame differencing
- 3D trajectory reconstruction and tracking
- Interactive visualization of UAP data
- Analysis reporting and data export
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

# PyQt imports for UI
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QMessageBox, QSlider, QProgressBar,
    QTabWidget, QSplitter, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QLinearGradient, 
    QRadialGradient, QFont, QBrush, QPainterPath
)

# UAP detection core imports
from ..pixel_motion import (
    PixelMotionVoxelProjection, VoxelGrid, CameraInfo, MotionMap, UAPVisualizer
)
from ..pixel_motion.utils import ObjectTracker, MotionFilter, ExportManager
from ..pixel_motion.interface import VideoFileInterface

# Set up module logger
logger = logging.getLogger(__name__)


@dataclass
class UAPProcessingSettings:
    """Data class for UAP detection processing settings."""
    # Motion detection settings
    threshold: float = 15.0              # Threshold for motion detection
    detection_threshold: float = 1.0     # Threshold for object detection in voxel space
    min_cluster_size: int = 5           # Minimum size of voxel clusters for object detection
    min_motion_size: int = 20           # Minimum size of motion regions to consider (pixels)
    skip_frames: int = 2                # Process every Nth frame
    
    # Voxel grid configuration
    resolution: Tuple[int, int, int] = (100, 100, 100)   # Voxel grid resolution
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-5, 5), (-5, 5), (0, 10)       # Spatial bounds of voxel grid in meters
    )
    
    # Tracking settings
    history_length: int = 5             # Number of frames to use for temporal filtering
    max_disappeared: int = 10           # Max frames object can disappear before losing ID
    max_distance: float = 1.0           # Max distance for tracking object between frames
    max_ray_distance: float = 10.0      # Max distance to project rays from camera
    
    # Performance settings
    use_gpu: bool = True                # Whether to use GPU acceleration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary format."""
        return {
            'threshold': self.threshold,
            'detection_threshold': self.detection_threshold,
            'min_cluster_size': self.min_cluster_size,
            'min_motion_size': self.min_motion_size,
            'skip_frames': self.skip_frames,
            'resolution': self.resolution,
            'bounds': self.bounds,
            'history_length': self.history_length,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance,
            'max_ray_distance': self.max_ray_distance,
            'use_gpu': self.use_gpu
        }


class VideoProcessorThread(QThread):
    """Thread for processing video files to detect and track UAP."""
    
    # Signals for updating the UI
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(object, object, object, list, dict)  # frame, motion_map, voxel_grid, objects, tracked_objects
    processing_finished = pyqtSignal(dict)  # results dictionary
    processing_error = pyqtSignal(str)  # error message
    
    def __init__(self, video_path: str, settings: Union[Dict[str, Any], UAPProcessingSettings]):
        """
        Initialize the UAP video processor thread.
        
        Args:
            video_path: Path to the video file to process
            settings: Dictionary or UAPProcessingSettings object with processing parameters
        """
        super().__init__()
        self.video_path = video_path
        
        # Convert settings to dictionary if it's a UAPProcessingSettings object
        if isinstance(settings, UAPProcessingSettings):
            self.settings = settings.to_dict()
        else:
            self.settings = settings
            
        self.abort_flag = False    # Flag to stop processing
    
    def run(self):
        """Run the video processing."""
        try:
            import cv2
            
            # Open video file
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.processing_error.emit(f"Could not open video file: {self.video_path}")
                return
                
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create camera info
            camera_info = CameraInfo(
                position=np.array([0, 0, 0]),  # Camera at origin
                orientation=np.eye(3),          # Looking along Z axis
                intrinsic_matrix=np.array([
                    [width, 0, width/2],   # fx, 0, cx
                    [0, width, height/2],  # 0, fy, cy
                    [0, 0, 1]              # 0, 0, 1
                ]),
                distortion_coeffs=np.zeros(5),  # No distortion
                resolution=(width, height),
                fov=(60, 40),                   # Approximate FOV 
                name="VideoCamera"
            )
            
            # Create voxel projector
            voxel_projector = PixelMotionVoxelProjection(
                grid_resolution=self.settings.get('resolution', (100, 100, 100)),
                grid_bounds=self.settings.get('bounds', ((-5, 5), (-5, 5), (0, 10))),
                use_gpu=self.settings.get('use_gpu', True)
            )
            
            # Add camera to projector
            voxel_projector.add_camera(camera_info)
            
            # Create motion filter
            motion_filter = MotionFilter(history_length=self.settings.get('history_length', 5))
            
            # Create object tracker
            tracker = ObjectTracker(
                max_disappeared=self.settings.get('max_disappeared', 10),
                max_distance=self.settings.get('max_distance', 1.0)
            )
            
            # Processing loop variables
            prev_frame = None
            frame_num = 0
            processed_count = 0
            skip_frames = self.settings.get('skip_frames', 2)
            
            # Process frames
            while not self.abort_flag:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                
                frame_num += 1
                
                # Skip frames if requested
                if frame_num % skip_frames != 0:
                    continue
                
                # Skip first frame (no previous frame to compare)
                if prev_frame is None:
                    prev_frame = frame.copy()
                    continue
                
                processed_count += 1
                
                # Get current timestamp (in video time)
                timestamp = frame_num / fps if fps > 0 else time.time()
                
                # Process frame difference
                motion_map = voxel_projector.process_frame_difference(
                    prev_frame,
                    frame,
                    camera_info.name,
                    timestamp,
                    threshold=self.settings.get('threshold', 15.0),
                    preprocess=True
                )
                
                # Apply temporal filtering
                filtered_map = motion_filter.apply_temporal_filter(motion_map)
                
                # Filter by size to remove small noise
                try:
                    filtered_map = motion_filter.filter_by_size(
                        filtered_map, 
                        min_size=self.settings.get('min_motion_size', 20),
                        max_size=width * height // 4  # Max 1/4 of frame
                    )
                except Exception as e:
                    logger.warning(f"Size filtering failed: {e}")
                
                # Reset voxel grid
                voxel_projector.reset_voxel_grid()
                
                # Project motion to voxels
                projection = voxel_projector.project_pixels_to_voxels(
                    filtered_map,
                    max_ray_distance=self.settings.get('max_ray_distance', 10.0)
                )
                
                # Update voxel grid
                voxel_projector.voxel_grid = projection
                
                # Find objects in voxel space
                objects = voxel_projector.get_detected_objects(
                    threshold=self.settings.get('detection_threshold', 1.0),
                    min_cluster_size=self.settings.get('min_cluster_size', 5)
                )
                
                # Update object tracker
                tracked_objects = tracker.update(objects, timestamp)
                
                # Emit progress and frame info
                progress = int(frame_num / frame_count * 100)
                self.progress_updated.emit(progress)
                
                # Emit processed data
                self.frame_processed.emit(
                    frame.copy(), 
                    filtered_map, 
                    voxel_projector.voxel_grid,
                    objects,
                    tracked_objects
                )
                
                # Update previous frame
                prev_frame = frame.copy()
            
            # Finish processing
            cap.release()
            
            # Get trajectories and classifications
            trajectories = tracker.get_trajectories()
            classifications = tracker.classify_trajectories()
            
            # Emit final results
            results = {
                'total_frames': frame_count,
                'processed_frames': processed_count,
                'trajectories': trajectories,
                'classifications': classifications,
                'objects_detected': len(trajectories)
            }
            
            self.processing_finished.emit(results)
            
        except Exception as e:
            logger.error(f"Error in video processing thread: {e}")
            self.processing_error.emit(str(e))
    
    def abort(self):
        """Abort the processing."""
        self.abort_flag = True


class MotionMapWidget(QWidget):
    """
    Widget for visualizing UAP motion maps with motion detection highlights.
    
    This widget renders a color-coded representation of detected motion in the
    video frames, with specific highlighting for potential UAP signatures.
    """
    
    def __init__(self, parent=None):
        """Initialize the motion map visualization widget."""
        super().__init__(parent)
        self.motion_map = None
        self.setMinimumSize(320, 240)
        
        # Color scheme for UAP visualization
        self.colors = {
            'background': QColor(0, 0, 0),            # Black background
            'text': QColor(200, 200, 200),            # Light gray text
            'highlight': QColor(255, 255, 255),       # White highlights
            'error': QColor(255, 80, 80),             # Red error messages
            'warning': QColor(255, 200, 80),          # Orange warnings
            'uap_highlight': QColor(0, 255, 0, 150)   # Green UAP highlights
        }
    
    def set_motion_map(self, motion_map: MotionMap):
        """
        Set the motion map data to display and update the visualization.
        
        Args:
            motion_map: The MotionMap object containing UAP motion detection data
        """
        self.motion_map = motion_map
        self.update()  # Trigger a repaint with the new data
        
    def paintEvent(self, event):
        """Paint the motion map."""
        # Fill background with black
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        if self.motion_map is None:
            # Draw "No data" text
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No motion data")
            return
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        
        try:
            # Check if motion_map data is valid
            if self.motion_map.data is None or self.motion_map.data.size == 0:
                painter.setPen(QPen(QColor(200, 200, 200)))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Empty motion data")
                return
                
            # Log the motion map properties
            logger.debug(f"Motion map shape: {self.motion_map.data.shape}, " 
                         f"min: {self.motion_map.data.min()}, max: {self.motion_map.data.max()}")
            
            import cv2
            
            # Ensure data is in the correct format for normalization
            motion_data = self.motion_map.data.copy()
            
            # Normalize data to 0-255 range
            if motion_data.max() > motion_data.min():  # Avoid division by zero
                normalized = cv2.normalize(
                    motion_data, 
                    None, 
                    0, 
                    255, 
                    cv2.NORM_MINMAX
                ).astype(np.uint8)
            else:
                normalized = np.zeros_like(motion_data, dtype=np.uint8)
            
            # Apply color map - JET is better for UAP detection visualization
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Resize to widget dimensions
            if width > 0 and height > 0:  # Make sure dimensions are valid
                resized = cv2.resize(colored, (width, height))
                
                # Convert to QImage
                h, w, c = resized.shape
                bytes_per_line = w * c
                qimg = QImage(resized.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                
                # Draw image
                painter.drawImage(0, 0, qimg)
                
                # Get motion pixels and highlight them
                pixels = self.motion_map.get_motion_pixels()
                if len(pixels) > 0:
                    # Scale pixels to widget size
                    scale_x = width / self.motion_map.data.shape[1]
                    scale_y = height / self.motion_map.data.shape[0]
                    
                    # Set pen for motion pixels
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    
                    # Draw circles at motion pixels (maximum 200 for performance)
                    display_pixels = pixels[:200] if len(pixels) > 200 else pixels
                    for pixel in display_pixels:
                        x = int(pixel[0] * scale_x)
                        y = int(pixel[1] * scale_y)
                        painter.drawEllipse(x - 2, y - 2, 4, 4)
                    
                    # Draw count in corner
                    painter.setPen(QPen(QColor(255, 255, 255)))
                    painter.drawText(10, 20, f"UAP Motion Pixels: {len(pixels)}")
            else:
                raise ValueError("Invalid widget dimensions")
                
        except Exception as e:
            logger.error(f"Error drawing motion map: {e}")
            
            # Fallback: draw error text
            painter.setPen(QPen(QColor(255, 80, 80)))
            painter.drawText(10, 30, f"Error: {str(e)}")
            painter.drawText(10, 50, "Try restarting the application or selecting a different video.")


class UAPTrackingRadar(QWidget):
    """
    Interactive 3D radar visualization for UAP tracking and trajectory analysis.
    
    This widget provides a dynamic 3D visualization of the voxel space, 
    detected UAP objects, and their trajectories. It includes interactive
    elements and animation for enhanced analysis capabilities.
    """
    
    def __init__(self, parent=None):
        """Initialize the UAP tracking radar visualization widget."""
        super().__init__(parent)
        
        # Setup logging
        self.logger = logging.getLogger(__name__ + ".UAPTrackingRadar")
        
        # Core data structures
        self.voxel_grid = None         # Current voxel grid data
        self.objects = []              # Detected UAP objects
        self.tracked_objects = {}      # UAP objects with tracking IDs
        
        # UI configuration
        self.setMinimumSize(320, 240)
        
        # Animation and interaction settings
        self.rotate_angle = 0          # Current rotation angle
        self.animation_speed = 1       # Rotation speed factor
        self.paused = False            # Animation pause flag
        
        # Set up rotation animation timer - using a safer interval
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.safely_rotate_view)
        self.animation_timer.start(200)  # 200ms = 5 updates per second to reduce load
        
        # Add a safety flag to prevent multiple updates during rendering
        self.is_updating = False
        
        # Mouse interaction setup
        self.setMouseTracking(True)
        self.mouse_position = None
        self.is_dragging = False
        self.drag_start_angle = 0
        
        # Visual configuration
        self.theme = {
            # Core colors
            'background_top': QColor(0, 0, 40),      # Deep space blue
            'background_bottom': QColor(0, 0, 0),    # Black
            'grid': QColor(30, 30, 60),              # Subtle grid lines
            
            # UAP tracking visualization
            'radar_glow': QColor(0, 150, 255, 40),   # Radar sweep effect
            'uap_glow': QColor(0, 255, 100, 150),    # UAP detection glow
            'uap_core': QColor(0, 255, 100),         # UAP object body
            'uap_outline': QColor(255, 255, 255),    # UAP object outline
            
            # Tracked UAP visualization
            'tracked_glow': QColor(255, 200, 0, 80), # Tracked object glow
            'tracked_core': QColor(255, 50, 50),     # Tracked object body
            'tracked_outline': QColor(255, 200, 0),  # Tracked object outline
            
            # Observer/camera position
            'observer': QColor(100, 100, 0),         # Observer fill color
            'observer_outline': QColor(255, 255, 0), # Observer outline
            
            # Text and UI elements
            'title': QColor(100, 200, 255),          # Title text
            'title_glow': QColor(0, 100, 200, 50),   # Title glow effect
            'label': QColor(255, 255, 0),            # Labels
            'info': QColor(200, 200, 200)            # Information text
        }
        
        # Log successful initialization
        self.logger.debug("UAPTrackingRadar initialized successfully")
    
    def safely_rotate_view(self):
        """Safely rotate the view with error handling."""
        try:
            if not self.is_updating and self.voxel_grid is not None:
                self.rotate_angle = (self.rotate_angle + 1) % 360
                # Use update() with care to prevent cascading repaints
                self.update()
        except Exception as e:
            self.logger.error(f"Error in rotation timer: {e}")
            # Consider stopping the timer if there are persistent errors
            # self.animation_timer.stop()
    
    def set_data(self, voxel_grid: VoxelGrid, objects: List[np.ndarray], 
                tracked_objects: Dict[int, np.ndarray]):
        """Set the data to display with validation."""
        try:
            # Validate inputs to prevent crashes
            if voxel_grid is None:
                self.logger.warning("Received None voxel_grid in set_data")
                
            if objects is None:
                self.logger.warning("Received None objects in set_data")
                objects = []
                
            if tracked_objects is None:
                self.logger.warning("Received None tracked_objects in set_data")
                tracked_objects = {}
            
            # Update internal data
            self.voxel_grid = voxel_grid
            self.objects = objects if isinstance(objects, list) else []
            self.tracked_objects = tracked_objects if isinstance(tracked_objects, dict) else {}
            
            # Request update (avoid direct update() call to reduce cascading repaints)
            QTimer.singleShot(10, self.update)
        except Exception as e:
            self.logger.error(f"Error in set_data: {e}")
    
    def mouseMoveEvent(self, event):
        """Track mouse movements for interactive elements."""
        try:
            self.mouse_position = event.position()
            # Avoid immediate update to prevent paint event cascade
            # self.update()
        except Exception as e:
            self.logger.error(f"Error in mouseMoveEvent: {e}")
        
    def paintEvent(self, event):
        """Paint the voxel grid visualization."""
        try:
            # Set update flag to prevent multiple concurrent paint events
            self.is_updating = True
            
            # Get painter
            painter = QPainter(self)
            
            # Get widget dimensions
            width = self.width()
            height = self.height()
            
            # Fill background with gradient for space effect
            gradient = QLinearGradient(0, 0, 0, height)
            gradient.setColorAt(0, QColor(0, 0, 40))  # Dark blue at top
            gradient.setColorAt(1, QColor(0, 0, 0))   # Black at bottom
            painter.fillRect(0, 0, width, height, gradient)
            
            # Draw grid lines with perspective effect
            painter.setPen(QPen(QColor(30, 30, 60)))
            
            # Horizontal grid lines with perspective effect
            vanishing_point_x = width // 2
            vanishing_point_y = height * 0.4  # Above center for better perspective
            
            # Draw radial grid lines - limit number to reduce CPU
            num_lines = 8  # Reduced from 12 for better performance
            try:
                for i in range(num_lines):
                    angle = 2 * np.pi * i / num_lines + np.radians(self.rotate_angle)
                    end_x = vanishing_point_x + 2 * width * np.cos(angle)
                    end_y = vanishing_point_y + 2 * height * np.sin(angle)
                    painter.drawLine(int(vanishing_point_x), int(vanishing_point_y), 
                                    int(end_x), int(end_y))
            except Exception as grid_err:
                self.logger.error(f"Error drawing grid: {grid_err}")
            
            # Draw concentric circles - fewer for performance
            try:
                for radius in range(50, 300, 50):  # Reduced upper bound for performance
                    scaled_radius = radius
                    painter.drawEllipse(int(vanishing_point_x - scaled_radius), 
                                      int(vanishing_point_y - scaled_radius * 0.4),
                                      int(scaled_radius * 2), 
                                      int(scaled_radius * 0.8))
            except Exception as circle_err:
                self.logger.error(f"Error drawing circles: {circle_err}")
            
            # Add title text 
            try:
                title_text = "UAP 3D Tracking Space"
                title_font = painter.font()
                title_font.setPointSize(14)
                title_font.setBold(True)
                painter.setFont(title_font)
                
                # Simplified glow effect - just one outline for performance
                glow_pen = QPen(QColor(0, 100, 200, 80))
                painter.setPen(glow_pen)
                painter.drawText(16, 31, title_text)
                
                # Draw main text
                painter.setPen(QPen(QColor(100, 200, 255)))
                painter.drawText(15, 30, title_text)
            except Exception as text_err:
                self.logger.error(f"Error drawing title: {text_err}")
            
            # If no voxel grid, show message and return early
            if self.voxel_grid is None:
                painter.setPen(QPen(QColor(200, 200, 200)))
                painter.drawText(width//2 - 100, height//2, "No UAP data to display")
                self.is_updating = False  # Reset update flag
                return
            
            # If voxel grid doesn't have bounds, add a warning and return early
            if not hasattr(self.voxel_grid, 'bounds') or self.voxel_grid.bounds is None:
                painter.setPen(QPen(QColor(255, 150, 150)))
                painter.drawText(width//2 - 100, height//2, "Invalid voxel data")
                self.is_updating = False  # Reset update flag
                return
            
            # Adjust center for visualization
            center_x = width // 2
            center_y = height // 2
            
            # Draw detected objects with animation - with safety checks
            if self.objects and isinstance(self.objects, list) and len(self.objects) > 0:
                try:
                    # Glow effect for UAP detection
                    radial_gradient = QRadialGradient(center_x, center_y, 100)
                    radial_gradient.setColorAt(0, QColor(0, 150, 255, 40))
                    radial_gradient.setColorAt(1, QColor(0, 0, 100, 0))
                    painter.setBrush(radial_gradient)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(center_x - 100, center_y - 40, 200, 80)
                    
                    # Draw UAP objects
                    painter.setPen(QPen(QColor(0, 255, 100), 2))
                    
                    # Only draw up to 50 objects max for performance
                    max_objects = min(len(self.objects), 50)
                    
                    for i in range(max_objects):
                        obj = self.objects[i]
                        try:
                            # Basic validation to prevent crashes
                            if not isinstance(obj, np.ndarray) or len(obj) < 3:
                                continue
                                
                            # Convert 3D coordinates to 2D screen coordinates
                            bounds = self.voxel_grid.bounds
                            
                            # Check for valid bounds
                            if not all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
                                continue
                                
                            # Check for division by zero
                            if bounds[0][1] == bounds[0][0] or bounds[2][1] == bounds[2][0]:
                                continue
                                
                            x_scale = (width * 0.8) / (bounds[0][1] - bounds[0][0])
                            z_scale = (height * 0.8) / (bounds[2][1] - bounds[2][0])
                            
                            # Apply rotation based on current angle
                            rotation_rad = np.radians(self.rotate_angle)
                            rot_x = obj[0] * np.cos(rotation_rad) - obj[2] * np.sin(rotation_rad)
                            rot_z = obj[0] * np.sin(rotation_rad) + obj[2] * np.cos(rotation_rad)
                            
                            x = int(center_x + rot_x * x_scale)
                            z = int(center_y - rot_z * z_scale)  # Z is depth, negate for screen coords
                            
                            # Safety check for valid coordinates
                            if not (0 <= x < width and 0 <= z < height):
                                continue
                            
                            # Simpler effect for performance
                            size = 12  # Fixed size
                            
                            # Draw glow effect
                            glow_gradient = QRadialGradient(x, z, size * 1.5)
                            glow_gradient.setColorAt(0, QColor(0, 255, 100, 150))
                            glow_gradient.setColorAt(1, QColor(0, 255, 100, 0))
                            painter.setBrush(glow_gradient)
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.drawEllipse(x - size*1.5, z - size*1.5, size*3, size*3)
                            
                            # Draw object
                            painter.setBrush(QColor(0, 255, 100))
                            painter.setPen(QPen(QColor(255, 255, 255), 1))
                            painter.drawEllipse(x - size//2, z - size//2, size, size)
                        except Exception as obj_err:
                            # Just skip this object, don't crash the whole render
                            continue
                except Exception as objs_err:
                    self.logger.error(f"Error drawing objects: {objs_err}")
            
            # Draw tracked UAP objects with IDs
            if self.tracked_objects and isinstance(self.tracked_objects, dict) and len(self.tracked_objects) > 0:
                try:
                    # Only draw up to 20 tracked objects for performance
                    count = 0
                    for obj_id, position in self.tracked_objects.items():
                        if count >= 20:
                            break
                            
                        try:
                            # Basic validation
                            if not isinstance(position, np.ndarray) or len(position) < 3:
                                continue
                                
                            # Convert 3D coordinates to 2D screen coordinates
                            bounds = self.voxel_grid.bounds
                            
                            # Check for valid bounds
                            if not all(isinstance(b, tuple) and len(b) == 2 for b in bounds):
                                continue
                                
                            # Check for division by zero
                            if bounds[0][1] == bounds[0][0] or bounds[2][1] == bounds[2][0]:
                                continue
                                
                            x_scale = (width * 0.8) / (bounds[0][1] - bounds[0][0])
                            z_scale = (height * 0.8) / (bounds[2][1] - bounds[2][0])
                            
                            # Apply rotation based on current angle
                            rotation_rad = np.radians(self.rotate_angle)
                            rot_x = position[0] * np.cos(rotation_rad) - position[2] * np.sin(rotation_rad)
                            rot_z = position[0] * np.sin(rotation_rad) + position[2] * np.cos(rotation_rad)
                            
                            x = int(center_x + rot_x * x_scale)
                            z = int(center_y - rot_z * z_scale)  # Z is depth
                            
                            # Safety check for valid coordinates
                            if not (0 <= x < width and 0 <= z < height):
                                continue
                            
                            # Draw targeting rectangle with simplified effect
                            painter.setPen(QPen(QColor(255, 200, 0, 80), 2))
                            size = 20
                            painter.drawRect(x - size, z - size, size*2, size*2)
                            
                            # Draw object
                            painter.setBrush(QColor(255, 50, 50))
                            painter.setPen(QPen(QColor(255, 200, 0), 2))
                            painter.drawRect(x - size, z - size, size*2, size*2)
                            
                            # Draw UAP ID with enhanced styling
                            id_font = painter.font()
                            id_font.setBold(True)
                            painter.setFont(id_font)
                            
                            # Draw text outline for better visibility
                            painter.setPen(QPen(QColor(0, 0, 0), 2))
                            id_text = f"UAP-{obj_id}"
                            painter.drawText(x + 15, z, id_text)
                            
                            painter.setPen(QPen(QColor(255, 200, 0)))
                            painter.drawText(x + 15, z, id_text)
                            
                            count += 1
                        except Exception as track_err:
                            # Just skip this tracked object, don't crash
                            continue
                except Exception as tracked_err:
                    self.logger.error(f"Error drawing tracked objects: {tracked_err}")
            
            # Draw camera position
            try:
                cam_size = 15
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.setBrush(QColor(100, 100, 0))
                painter.drawEllipse(center_x - cam_size//2, center_y - cam_size//2, cam_size, cam_size)
                
                # Draw camera label with simple effect
                camera_font = painter.font()
                camera_font.setBold(True)
                painter.setFont(camera_font)
                
                # Actual text
                painter.setPen(QPen(QColor(255, 255, 0)))
                painter.drawText(center_x + 10, center_y + 15, "Observer")
            except Exception as cam_err:
                self.logger.error(f"Error drawing camera: {cam_err}")
            
            # Draw status information
            try:
                status_font = painter.font()
                status_font.setPointSize(8)
                painter.setFont(status_font)
                painter.setPen(QPen(QColor(200, 200, 200)))
                
                # Bottom info bar
                tracked_count = len(self.tracked_objects) if isinstance(self.tracked_objects, dict) else 0
                info_text = f"UAP Detected: {tracked_count} | View Angle: {self.rotate_angle}°"
                
                if self.voxel_grid is not None and hasattr(self.voxel_grid, 'bounds'):
                    bounds = self.voxel_grid.bounds
                    if isinstance(bounds, tuple) and len(bounds) >= 3 and isinstance(bounds[2], tuple) and len(bounds[2]) >= 2:
                        info_text += f" | Range: {bounds[2][1]:.1f}m"
                
                painter.drawText(10, height - 10, info_text)
            except Exception as status_err:
                self.logger.error(f"Error drawing status: {status_err}")
        
        except Exception as global_err:
            # Global try/except to catch any unexpected errors
            # Log the error, but don't interrupt the UI
            self.logger.error(f"Critical error in paintEvent: {global_err}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # Try to restore the display with a simple error message
            try:
                if painter.isActive():
                    painter.setPen(QPen(QColor(255, 50, 50)))
                    painter.drawText(10, 30, "Rendering error - please restart application")
            except:
                pass  # If even this fails, just give up
        
        finally:
            # Always reset the update flag when we're done
            self.is_updating = False


class PixelMotionTab(QWidget):
    """
    UAP Detection and Analysis Tab for the Pixel2UAP Projector.
    
    This tab provides a comprehensive user interface for analyzing videos 
    to detect and track Unidentified Aerial Phenomena (UAP) using the 
    Pixel Motion Voxel Projection technique. The system analyzes motion 
    across video frames and projects detected phenomena into 3D space
    for tracking and trajectory analysis.
    
    Key Features:
    - UAP video file selection and analysis
    - Advanced motion detection parameter configuration
    - Real-time visualization of UAP motion in 2D and 3D space
    - Interactive 3D radar display for UAP tracking
    - Trajectory analysis and velocity measurements
    - Export of UAP detection data for further investigation
    """
    
    def __init__(self, parent=None):
        """Initialize the UAP Detection and Analysis tab."""
        super().__init__(parent)
        
        # Initialize state variables
        self.video_path = None             # Path to UAP video file
        self.processor_thread = None       # Background processing thread
        
        # Current processing state
        self.current_frame = None          # Current video frame being processed
        self.current_motion_map = None     # Current motion detection map
        self.current_voxel_grid = None     # Current 3D voxel projection
        self.current_objects = []          # List of detected UAP objects
        self.current_tracked_objects = {}  # Dict of tracked UAP objects with IDs
        self.processing_results = {}       # Final processing results
        
        # Default UAP detection settings
        self.default_settings = UAPProcessingSettings()
        
        # Setup the UI components
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Right panel (visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Initial sizes
        
        main_layout.addWidget(splitter)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        
        # Video file selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        file_button = QPushButton("Select Video")
        file_button.clicked.connect(self.select_video_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(file_button)
        input_layout.addLayout(file_layout)
        
        # Video info
        self.video_info_label = QLabel("No video loaded")
        input_layout.addWidget(self.video_info_label)
        
        left_layout.addWidget(input_group)
        
        # Processing parameters
        params_group = QGroupBox("Processing Parameters")
        params_layout = QFormLayout(params_group)
        
        # Skip frames
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(1, 10)
        self.skip_frames_spin.setValue(2)
        self.skip_frames_spin.setToolTip("Process every Nth frame")
        params_layout.addRow("Skip Frames:", self.skip_frames_spin)
        
        # Motion threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 50.0)
        self.threshold_spin.setValue(15.0)
        self.threshold_spin.setToolTip("Threshold for motion detection")
        params_layout.addRow("Motion Threshold:", self.threshold_spin)
        
        # Detection threshold
        self.detection_threshold_spin = QDoubleSpinBox()
        self.detection_threshold_spin.setRange(0.1, 10.0)
        self.detection_threshold_spin.setValue(1.0)
        self.detection_threshold_spin.setToolTip("Threshold for object detection")
        params_layout.addRow("Detection Threshold:", self.detection_threshold_spin)
        
        # Min cluster size
        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setRange(1, 100)
        self.min_cluster_spin.setValue(5)
        self.min_cluster_spin.setToolTip("Minimum size of voxel clusters for object detection")
        params_layout.addRow("Min Cluster Size:", self.min_cluster_spin)
        
        # GPU acceleration
        self.gpu_checkbox = QCheckBox("Use GPU Acceleration")
        self.gpu_checkbox.setChecked(True)
        params_layout.addRow("", self.gpu_checkbox)
        
        left_layout.addWidget(params_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        controls_layout.addWidget(self.process_button)
        controls_layout.addWidget(self.stop_button)
        
        left_layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_label = QLabel("No processing results yet")
        results_layout.addWidget(self.results_label)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_video_button = QPushButton("Export Video")
        self.export_video_button.clicked.connect(self.export_video)
        self.export_video_button.setEnabled(False)
        
        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.export_data)
        self.export_data_button.setEnabled(False)
        
        export_layout.addWidget(self.export_video_button)
        export_layout.addWidget(self.export_data_button)
        
        results_layout.addLayout(export_layout)
        
        left_layout.addWidget(results_group)
        
        # Add stretch to bottom
        left_layout.addStretch()
        
        # Visualization area
        viz_tabs = QTabWidget()
        right_layout.addWidget(viz_tabs)
        
        # Original video tab
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        viz_tabs.addTab(video_tab, "Video")
        
        # Motion map tab
        motion_tab = QWidget()
        motion_layout = QVBoxLayout(motion_tab)
        self.motion_widget = MotionMapWidget()
        motion_layout.addWidget(self.motion_widget)
        viz_tabs.addTab(motion_tab, "Motion Map")
        
        # UAP 3D tracking radar tab
        radar_tab = QWidget()
        radar_layout = QVBoxLayout(radar_tab)
        self.radar_widget = UAPTrackingRadar()
        radar_layout.addWidget(self.radar_widget)
        viz_tabs.addTab(radar_tab, "3D UAP Tracking")
        
        # Status label
        self.status_label = QLabel("Ready")
        right_layout.addWidget(self.status_label)
    
    def select_video_file(self):
        """Open a file dialog to select a UAP video file for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select UAP Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not file_path:
            return  # User canceled
            
        # Update UI to show loading state
        self.file_path_label.setText("Loading video...")
        self.video_info_label.setText("Analyzing video file...")
        self.status_label.setText("Processing video metadata...")
        self.process_button.setEnabled(False)
        
        # Store the path
        self.video_path = file_path
        
        # Try to load the video
        try:
            # Import OpenCV here to give clear error if not available
            try:
                import cv2
            except ImportError:
                self.video_info_label.setText("Error: OpenCV (cv2) is required to process videos")
                return
                
            # Try to open the video file
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                self.file_path_label.setText(os.path.basename(file_path))
                self.video_info_label.setText("Error: Could not open video file. The file may be corrupted or use an unsupported codec.")
                self.status_label.setText("Error opening video file")
                return
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Check for valid properties
            if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
                self.file_path_label.setText(os.path.basename(file_path))
                self.video_info_label.setText("Warning: Video metadata may be incomplete. Try a different video file.")
                self.status_label.setText("Invalid video metadata")
                cap.release()
                return
            
            # Update UI with file name
            self.file_path_label.setText(os.path.basename(file_path))
            
            # Display info with UAP analysis potential
            sky_recording = "High" if height >= 720 else "Medium" if height >= 480 else "Low"
            duration_min = duration / 60
            
            info_text = (
                f"Resolution: {width}x{height} ({sky_recording} analysis quality)\n"
                f"Frame Rate: {fps:.1f} FPS (Optimal: >24 FPS)\n"
                f"Duration: {duration_min:.1f} minutes ({frame_count:,} frames)\n"
                f"File: {os.path.basename(file_path)}"
            )
            self.video_info_label.setText(info_text)
            
            # Load first frame for preview
            ret, frame = cap.read()
            if ret:
                # Try to enhance contrast for better visualization
                try:
                    # Apply mild contrast enhancement for preview
                    frame_float = frame.astype(np.float32) / 255.0
                    frame_enhanced = np.clip(frame_float * 1.2, 0, 1) * 255
                    frame_preview = frame_enhanced.astype(np.uint8)
                    
                    # Draw informative overlay
                    cv2.putText(
                        frame_preview,
                        "UAP Analysis Preview",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 200, 255),
                        2
                    )
                    
                    # Show UAP analysis regions
                    frame_height, frame_width = frame_preview.shape[:2]
                    overlay = frame_preview.copy()
                    
                    # Draw analysis grid - UAPs often appear near horizon or in sky
                    for i in range(0, frame_width, frame_width // 10):
                        cv2.line(overlay, (i, 0), (i, frame_height), (0, 100, 200), 1)
                    for i in range(0, frame_height, frame_height // 8):
                        cv2.line(overlay, (0, i), (frame_width, i), (0, 100, 200), 1)
                    
                    # Apply overlay with transparency
                    alpha = 0.2
                    frame_preview = cv2.addWeighted(overlay, alpha, frame_preview, 1 - alpha, 0)
                    
                    # Show the enhanced preview
                    self.show_frame(frame_preview)
                except Exception as enhance_err:
                    logger.warning(f"Error enhancing preview: {enhance_err}")
                    # Fall back to showing the original frame
                    self.show_frame(frame)
            else:
                # If we couldn't read a frame, show a placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "Video loaded - no preview available",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                self.show_frame(placeholder)
            
            # Enable the process button if everything looks good
            self.process_button.setEnabled(True)
            self.status_label.setText("Ready for UAP detection analysis")
            
            # Clean up
            cap.release()
            
        except Exception as e:
            # Log detailed error
            logger.error(f"Error reading video file: {e}")
            
            # Show user-friendly message
            self.file_path_label.setText(os.path.basename(file_path))
            self.video_info_label.setText(f"Error: {str(e)}\nTry a different video file format.")
            self.status_label.setText("Error processing video")
    
    def start_processing(self):
        """Start processing the selected video file."""
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return
        
        # Disable controls during processing
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Reset results
        self.progress_bar.setValue(0)
        self.results_label.setText("Processing...")
        
        # Create UAP detection settings from UI parameters
        settings = UAPProcessingSettings(
            skip_frames=self.skip_frames_spin.value(),
            threshold=self.threshold_spin.value(),
            detection_threshold=self.detection_threshold_spin.value(),
            min_cluster_size=self.min_cluster_spin.value(),
            use_gpu=self.gpu_checkbox.isChecked(),
            # Use default values for other parameters
            resolution=(100, 100, 100),
            bounds=((-5, 5), (-5, 5), (0, 10)),
            max_ray_distance=10.0,
            history_length=5,
            max_disappeared=10,
            max_distance=1.0,
            min_motion_size=20
        )
        
        # Create and start processing thread
        self.processor_thread = VideoProcessorThread(self.video_path, settings)
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.frame_processed.connect(self.update_visualization)
        self.processor_thread.processing_finished.connect(self.processing_completed)
        self.processor_thread.processing_error.connect(self.processing_error)
        self.processor_thread.start()
        
        self.status_label.setText("Processing video...")
    
    def stop_processing(self):
        """Stop the current processing operation."""
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.abort()
            self.processor_thread.wait()
            
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Processing stopped by user")
    
    def update_progress(self, progress: int):
        """Update the progress bar."""
        self.progress_bar.setValue(progress)
    
    def update_visualization(self, frame, motion_map, voxel_grid, objects, tracked_objects):
        """
        Update all visualizations with new UAP detection data.
        
        This method is called whenever new data is received from the processor thread,
        and updates all visualization components with the latest detection results.
        
        Args:
            frame: The current video frame being processed
            motion_map: The motion detection map showing UAP movement
            voxel_grid: The 3D voxel grid with projected motion
            objects: List of detected UAP objects
            tracked_objects: Dictionary of tracked UAP objects with IDs
        """
        # Store current data for later use (e.g. export)
        self.current_frame = frame
        self.current_motion_map = motion_map
        self.current_voxel_grid = voxel_grid
        self.current_objects = objects
        self.current_tracked_objects = tracked_objects
        
        # Update video frame display with UAP annotations
        self.show_frame(frame, tracked_objects)
        
        # Update 2D motion map visualization
        self.motion_widget.set_motion_map(motion_map)
        
        # Update 3D UAP tracking radar
        self.radar_widget.set_data(voxel_grid, objects, tracked_objects)
        
        # Update status bar with detection count
        self.status_label.setText(f"UAP detected: {len(tracked_objects)} | Processing...")
    
    def show_frame(self, frame, tracked_objects=None):
        """Display a video frame with optional tracked objects overlay."""
        try:
            # Make sure frame is valid
            if frame is None or not hasattr(frame, 'shape') or len(frame.shape) != 3:
                logger.error(f"Invalid frame format: {type(frame)}")
                self.video_label.setText("Error: Invalid frame format")
                return
                
            # Log frame stats for debugging
            logger.debug(f"Frame shape: {frame.shape}, type: {frame.dtype}")
            
            # Make a copy to avoid modifying the original
            try:
                display_frame = frame.copy()
            except Exception as copy_err:
                logger.error(f"Error copying frame: {copy_err}")
                display_frame = frame
            
            # Draw tracked objects on the frame if provided
            if tracked_objects and len(tracked_objects) > 0:
                try:
                    import cv2
                    
                    # Get camera info to project 3D points to 2D
                    height, width = display_frame.shape[:2]
                    camera_info = CameraInfo(
                        position=np.array([0, 0, 0]),
                        orientation=np.eye(3),
                        intrinsic_matrix=np.array([
                            [width, 0, width/2],
                            [0, width, height/2],
                            [0, 0, 1]
                        ]),
                        distortion_coeffs=np.zeros(5),
                        resolution=(width, height),
                        fov=(60, 40),
                        name="VideoCamera"
                    )
                    
                    # Draw objects
                    for obj_id, position in tracked_objects.items():
                        try:
                            # Project 3D position to 2D
                            pixel = camera_info.world_to_pixel(np.append(position, 1.0))
                            
                            # Check if pixel is within image bounds
                            if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                                # Draw circle at position
                                cv2.circle(
                                    display_frame,
                                    (int(pixel[0]), int(pixel[1])),
                                    20,  # radius
                                    (0, 0, 255),  # red in BGR
                                    2  # thickness
                                )
                                
                                # Add ID text
                                cv2.putText(
                                    display_frame,
                                    f"UAP-{obj_id}",
                                    (int(pixel[0]) - 25, int(pixel[1]) - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2
                                )
                        except Exception as proj_err:
                            logger.warning(f"Error projecting object {obj_id}: {proj_err}")
                            continue
                except Exception as draw_err:
                    logger.error(f"Error drawing objects: {draw_err}")
            
            try:
                # Ensure the frame is in RGB format (convert BGR to RGB if from OpenCV)
                import cv2
                if display_frame.shape[2] == 3:  # Has 3 color channels
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = display_frame
                
                # Ensure contiguous memory
                if not rgb_frame.flags['C_CONTIGUOUS']:
                    rgb_frame = np.ascontiguousarray(rgb_frame)
                
                # Convert to QImage
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                
                # Use the correct format without swapping RGB
                qimg = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Scale to fit the label while preserving aspect ratio
                pixmap = QPixmap.fromImage(qimg)
                
                # Make sure the label has a valid size
                if self.video_label.width() > 0 and self.video_label.height() > 0:
                    pixmap = pixmap.scaled(
                        self.video_label.width(), 
                        self.video_label.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                
                # Set the pixmap and update
                self.video_label.setPixmap(pixmap)
                self.video_label.update()
            except Exception as ui_err:
                logger.error(f"Error updating UI with frame: {ui_err}")
                # Create a simple colored placeholder with error message
                try:
                    import cv2
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    placeholder[:, :, 2] = 255  # Red background in BGR
                    cv2.putText(
                        placeholder,
                        "Video Preview Error",
                        (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2
                    )
                    # Try showing the placeholder instead
                    self.show_frame(placeholder)
                except:
                    # If that also fails, use text-based display
                    self.video_label.setText(f"Error: Video preview unavailable")
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            self.video_label.setText(f"Error: {str(e)}")
    
    def processing_completed(self, results: Dict[str, Any]):
        """Handle completion of UAP video processing."""
        self.processing_results = results
        
        # Enable controls for export and reset
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_video_button.setEnabled(True)
        self.export_data_button.setEnabled(True)
        
        # Extract trajectories for analysis
        trajectories = results.get('trajectories', {})
        classifications = results.get('classifications', {})
        objects_detected = results.get('objects_detected', 0)
        
        # Calculate some statistics about UAP movement
        max_velocity = 0
        avg_trajectory_length = 0
        max_trajectory_length = 0
        
        if trajectories:
            # Calculate trajectory lengths and velocities
            trajectory_lengths = []
            for obj_id, positions in trajectories.items():
                if len(positions) > 1:
                    # Calculate length (distance traveled)
                    total_dist = 0
                    for i in range(1, len(positions)):
                        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                        total_dist += dist
                    
                    trajectory_lengths.append(total_dist)
                    
                    # Calculate approximate velocity
                    if len(positions) > 2:
                        # Use first and last position for velocity calculation
                        time_span = results.get('processing_time', 1.0) * (len(positions) / results.get('processed_frames', 1))
                        if time_span > 0:
                            velocity = total_dist / time_span
                            max_velocity = max(max_velocity, velocity)
                
            if trajectory_lengths:
                avg_trajectory_length = sum(trajectory_lengths) / len(trajectory_lengths)
                max_trajectory_length = max(trajectory_lengths)
        
        # Format velocity for display
        velocity_str = f"{max_velocity:.1f}" if max_velocity > 0 else "N/A"
        
        # Show comprehensive analysis results
        result_text = (
            f"UAP ANALYSIS COMPLETE\n"
            f"-------------------------\n"
            f"Total frames analyzed: {results.get('processed_frames', 0):,}/{results.get('total_frames', 0):,}\n"
            f"UAPs detected: {objects_detected}\n"
            f"Max velocity: {velocity_str} m/s\n"
            f"Avg trajectory: {avg_trajectory_length:.2f} m"
        )
        
        # Add classification summary if available
        if classifications:
            class_counts = {}
            for obj_id, cls in classifications.items():
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
            result_text += "\n\nUAP Classifications:\n"
            for cls, count in class_counts.items():
                result_text += f"- {cls}: {count}\n"
        self.results_label.setText(result_text)
        
        self.status_label.setText("Processing completed")
    
    def processing_error(self, error_message: str):
        """Handle processing errors."""
        QMessageBox.critical(self, "Processing Error", f"Error processing video:\n{error_message}")
        
        # Reset controls
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self.status_label.setText("Error: " + error_message)
    
    def export_video(self):
        """Export the processed video with overlays."""
        if not self.processing_results:
            QMessageBox.warning(self, "No Results", "No processing results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Processed Video", "",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # Make sure the file has the correct extension
        if not file_path.lower().endswith(('.mp4', '.avi')):
            file_path += '.mp4'
            
        # TODO: Implement video export
        QMessageBox.information(
            self, 
            "Export Started", 
            f"Exporting processed video to {file_path}.\nThis may take a while."
        )
        
        # This would actually be implemented with a thread to create the video
        self.status_label.setText(f"Video export not yet implemented")
    
    def export_data(self):
        """Export the processing results data."""
        if not self.processing_results:
            QMessageBox.warning(self, "No Results", "No processing results to export.")
            return
        
        # Create a directory for the exports
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", ""
        )
        
        if not directory:
            return
            
        # Create an export manager
        export_manager = ExportManager(directory)
        
        try:
            # Export trajectories
            trajectories = self.processing_results.get('trajectories', {})
            classifications = self.processing_results.get('classifications', {})
            
            if trajectories:
                export_manager.export_trajectory_data(
                    trajectories,
                    classifications,
                    "trajectories.json"
                )
                
                # Create trajectory visualization
                export_manager.create_trajectory_visualization(
                    trajectories,
                    classifications,
                    "trajectory_visualization.png"
                )
            
            # Export last frame data if available
            if self.current_tracked_objects:
                export_manager.export_detection_results(
                    self.current_tracked_objects,
                    None,
                    "detected_objects.json"
                )
            
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Results exported to {directory}"
            )
            
            self.status_label.setText(f"Data exported to {directory}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting data: {str(e)}"
            )
            
            self.status_label.setText("Error exporting data")


# Test function to run the tab standalone
def main():
    """Run the PixelMotionTab as a standalone application."""
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = PixelMotionTab()
    window.setWindowTitle("Pixel Motion Voxel Projection")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()  # Run the PixelMotionTab as standalone for testing UAP detection