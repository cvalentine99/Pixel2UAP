#!/usr/bin/env python
"""
Pixel2UAP - UAP Detection and Analysis Toolkit

This specialized application processes video files or camera feeds to detect and track 
Unidentified Aerial Phenomena (UAP) through advanced motion analysis and 3D spatial tracking.

Usage:
    python pixel2uap_app.py [--video VIDEO_PATH] [--output OUTPUT_DIR]

Features:
- UAP detection in video footage
- Real-time camera feed UAP monitoring
- 3D visualization of UAP trajectories
- Advanced UAP tracking and flight pattern analysis
- Comprehensive data export for further investigation
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Pixel2UAP")

# Import the Pixel2UAP module
try:
    from .pixel_motion import (
        PixelMotionVoxelProjection, CameraInfo, MotionMap, VoxelGrid, UAPVisualizer
    )
    from .pixel_motion.utils import (
        ObjectTracker, MotionFilter, ExportManager
    )
    PIXEL_MOTION_AVAILABLE = True
except ImportError:
    logger.warning("Pixel2UAP module not available. UAP detection will be disabled.")
    PIXEL_MOTION_AVAILABLE = False

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualization will be limited.")
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Video processing will be disabled.")
    OPENCV_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    logger.warning("Open3D not available. 3D visualization will be disabled.")
    OPEN3D_AVAILABLE = False


def create_camera_info(width: int, height: int, name: str = "Camera") -> CameraInfo:
    """
    Create camera information object with default parameters.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        name: Camera name
        
    Returns:
        CameraInfo: Camera information object
    """
    # Default intrinsic parameters (approximate for a standard camera)
    # These values can be tuned based on the actual camera or calibration data
    intrinsic_matrix = np.array([
        [width, 0, width/2],   # fx, 0, cx
        [0, width, height/2],  # 0, fy, cy
        [0, 0, 1]              # 0, 0, 1
    ])
    
    # Create camera info
    camera_info = CameraInfo(
        position=np.array([0, 0, 0]),  # Camera at origin
        orientation=np.eye(3),          # Looking along Z axis
        intrinsic_matrix=intrinsic_matrix,
        distortion_coeffs=np.zeros(5),  # No distortion
        resolution=(width, height),
        fov=(60, 40),                   # Approximate FOV 
        name=name
    )
    
    return camera_info


def process_video_file(video_path: str, 
                      output_dir: Optional[str] = None,
                      settings: Optional[Dict[str, Any]] = None,
                      visualize: bool = True,
                      save_frames: bool = False,
                      save_results: bool = True) -> Dict[str, Any]:
    """
    Process a video file to detect moving objects using Pixel Motion Voxel Projection.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
        settings: Dictionary of processing settings
        visualize: Whether to show visualization during processing
        save_frames: Whether to save visualization frames
        save_results: Whether to save results to files
        
    Returns:
        Dict[str, Any]: Processing results
    """
    if not OPENCV_AVAILABLE:
        logger.error("OpenCV is required for video processing")
        return {}
        
    if not PIXEL_MOTION_AVAILABLE:
        logger.error("Pixel Motion module is required for processing")
        return {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use default settings if not provided
    if settings is None:
        settings = {
            'skip_frames': 2,
            'threshold': 15.0,
            'detection_threshold': 1.0,
            'min_cluster_size': 5,
            'use_gpu': True,
            'resolution': (100, 100, 100),
            'bounds': ((-5, 5), (-5, 5), (0, 10)),
            'max_ray_distance': 10.0,
            'history_length': 5,
            'max_disappeared': 10,
            'max_distance': 1.0,
            'min_motion_size': 20
        }
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return {}
        
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  Total frames: {frame_count}")
    
    # Create camera info
    camera_info = create_camera_info(width, height, "VideoCamera")
    
    # Create voxel projector
    voxel_projector = PixelMotionVoxelProjection(
        grid_resolution=settings.get('resolution', (100, 100, 100)),
        grid_bounds=settings.get('bounds', ((-5, 5), (-5, 5), (0, 10))),
        use_gpu=settings.get('use_gpu', True)
    )
    
    # Add camera to projector
    voxel_projector.add_camera(camera_info)
    
    # Create motion filter
    motion_filter = MotionFilter(history_length=settings.get('history_length', 5))
    
    # Create object tracker
    tracker = ObjectTracker(
        max_disappeared=settings.get('max_disappeared', 10),
        max_distance=settings.get('max_distance', 1.0)
    )
    
    # Create visualizer if requested
    if visualize and MATPLOTLIB_AVAILABLE:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(15, 10))
        
        # Create layout
        gs = plt.GridSpec(2, 2, figure=fig)
        ax_video = fig.add_subplot(gs[0, 0])
        ax_motion = fig.add_subplot(gs[0, 1])
        ax_voxel = fig.add_subplot(gs[1, 0], projection='3d')
        ax_trajectory = fig.add_subplot(gs[1, 1])
        
        # Set up plots
        ax_video.set_title("Video Frame")
        ax_video.axis('off')
        
        ax_motion.set_title("Motion Detection")
        ax_motion.axis('off')
        
        ax_voxel.set_title("3D Voxel Projection")
        ax_voxel.set_xlabel('X')
        ax_voxel.set_ylabel('Z')
        ax_voxel.set_zlabel('Y')
        
        ax_trajectory.set_title("Object Trajectories")
        ax_trajectory.set_xlabel('X')
        ax_trajectory.set_ylabel('Z')
        ax_trajectory.grid(True)
        
        # Initialize plots
        video_plot = ax_video.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        motion_plot = ax_motion.imshow(np.zeros((height, width), dtype=np.uint8), cmap='jet')
        voxel_scatter = None
        trajectory_plots = {}
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    # Create export manager if saving results
    if save_results and output_dir:
        export_manager = ExportManager(output_dir)
    
    # Setup video writer for saving visualization if requested
    if save_frames and output_dir:
        viz_video_path = os.path.join(output_dir, "visualization.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            viz_video_path,
            fourcc,
            fps / settings.get('skip_frames', 2),
            (width, height)
        )
    else:
        video_writer = None
    
    # Processing loop variables
    prev_frame = None
    frame_num = 0
    processed_count = 0
    skip_frames = settings.get('skip_frames', 2)
    trajectories = {}
    processing_start_time = time.time()
    
    try:
        # Main processing loop
        while True:
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
                threshold=settings.get('threshold', 15.0),
                preprocess=True
            )
            
            # Apply temporal filtering
            filtered_map = motion_filter.apply_temporal_filter(motion_map)
            
            # Filter by size to remove small noise
            try:
                filtered_map = motion_filter.filter_by_size(
                    filtered_map, 
                    min_size=settings.get('min_motion_size', 20),
                    max_size=width * height // 4  # Max 1/4 of frame
                )
            except Exception as e:
                logger.warning(f"Size filtering failed: {e}")
            
            # Reset voxel grid
            voxel_projector.reset_voxel_grid()
            
            # Project motion to voxels
            projection = voxel_projector.project_pixels_to_voxels(
                filtered_map,
                max_ray_distance=settings.get('max_ray_distance', 10.0)
            )
            
            # Update voxel grid
            voxel_projector.voxel_grid = projection
            
            # Find objects in voxel space
            objects = voxel_projector.get_detected_objects(
                threshold=settings.get('detection_threshold', 1.0),
                min_cluster_size=settings.get('min_cluster_size', 5)
            )
            
            # Update object tracker
            tracked_objects = tracker.update(objects, timestamp)
            
            # Update visualization if requested
            if visualize and MATPLOTLIB_AVAILABLE:
                # Update video frame
                ax_video.set_title(f"Frame {frame_num}/{frame_count}")
                video_plot.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Update motion map
                motion_vis = cv2.normalize(
                    filtered_map.data, 
                    None, 
                    0, 
                    255, 
                    cv2.NORM_MINMAX
                ).astype(np.uint8)
                motion_plot.set_data(motion_vis)
                
                # Update voxel visualization
                ax_voxel.clear()
                ax_voxel.set_title(f"3D Voxel Projection - {len(objects)} objects")
                ax_voxel.set_xlabel('X')
                ax_voxel.set_ylabel('Z')
                ax_voxel.set_zlabel('Y')
                
                # Extract voxel points
                if voxel_projector.voxel_grid is not None:
                    bounds = voxel_projector.voxel_grid.bounds
                    ax_voxel.set_xlim(bounds[0])
                    ax_voxel.set_ylim(bounds[2])  # Z axis for depth
                    ax_voxel.set_zlim(bounds[1])  # Y axis for height
                    
                    # Extract voxel data points
                    voxel_indices = np.where(voxel_projector.voxel_grid.grid > 0.5)
                    if len(voxel_indices[0]) > 0:
                        voxel_points = []
                        voxel_values = []
                        
                        for i, j, k in zip(*voxel_indices):
                            point = voxel_projector.voxel_grid.voxel_to_world((i, j, k))
                            voxel_points.append(point)
                            voxel_values.append(voxel_projector.voxel_grid.grid[i, j, k])
                            
                        voxel_points = np.array(voxel_points)
                        voxel_values = np.array(voxel_values)
                        
                        # Plot voxels
                        ax_voxel.scatter(
                            voxel_points[:, 0],  # X
                            voxel_points[:, 2],  # Z (depth)
                            voxel_points[:, 1],  # Y (height)
                            c=voxel_values,
                            cmap='hot',
                            alpha=0.3,
                            s=10
                        )
                
                # Plot camera and detected objects
                ax_voxel.scatter(0, 0, 0, c='yellow', marker='^', s=100, label='Camera')
                
                if objects:
                    ax_voxel.scatter(
                        [obj[0] for obj in objects],
                        [obj[2] for obj in objects],
                        [obj[1] for obj in objects],
                        c='green',
                        marker='o',
                        s=50,
                        label='Detected'
                    )
                
                if tracked_objects:
                    ax_voxel.scatter(
                        [pos[0] for pos in tracked_objects.values()],
                        [pos[2] for pos in tracked_objects.values()],
                        [pos[1] for pos in tracked_objects.values()],
                        c='red',
                        marker='s',
                        s=80,
                        label='Tracked'
                    )
                    
                    # Add ID labels
                    for obj_id, pos in tracked_objects.items():
                        ax_voxel.text(
                            pos[0], pos[2], pos[1],
                            f"ID:{obj_id}",
                            color='white',
                            fontsize=8,
                            horizontalalignment='center'
                        )
                
                ax_voxel.legend(loc='upper left')
                
                # Update trajectories
                ax_trajectory.clear()
                ax_trajectory.set_title("Object Trajectories (Top View)")
                ax_trajectory.set_xlabel('X')
                ax_trajectory.set_ylabel('Z')
                ax_trajectory.grid(True)
                
                # Get updated trajectories
                trajectories = tracker.get_trajectories()
                
                # Plot camera position
                ax_trajectory.plot(0, 0, 'y^', markersize=10, label='Camera')
                
                # Set bounds
                bounds = voxel_projector.voxel_grid.bounds
                ax_trajectory.set_xlim(bounds[0])
                ax_trajectory.set_ylim(bounds[2])
                
                # Plot trajectories
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
                for i, (obj_id, traj) in enumerate(trajectories.items()):
                    if len(traj) > 1:
                        traj = np.array(traj)
                        color_idx = i % 10
                        
                        # Plot trajectory line
                        ax_trajectory.plot(
                            traj[:, 0], 
                            traj[:, 2],
                            '-', 
                            color=colors[color_idx],
                            linewidth=2,
                            label=f"ID:{obj_id}"
                        )
                        
                        # Mark start and end points
                        ax_trajectory.plot(
                            traj[0, 0],
                            traj[0, 2],
                            'o',
                            color=colors[color_idx],
                            markersize=6
                        )
                        
                        ax_trajectory.plot(
                            traj[-1, 0],
                            traj[-1, 2],
                            's',
                            color=colors[color_idx],
                            markersize=8
                        )
                
                ax_trajectory.legend(loc='upper left', fontsize='small')
                
                # Update display
                plt.draw()
                plt.pause(0.001)
                
                # Save visualization frame if requested
                if save_frames and output_dir:
                    fig.savefig(os.path.join(output_dir, f"frame_{frame_num:04d}.png"), dpi=100)
            
            # Create visualization frame for video writer
            if video_writer is not None:
                try:
                    # Create a copy of the frame for visualization
                    viz_frame = frame.copy()
                    
                    # Draw detected motion
                    motion_vis = cv2.normalize(
                        filtered_map.data, 
                        None, 
                        0, 
                        255, 
                        cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    
                    motion_color = cv2.applyColorMap(motion_vis, cv2.COLORMAP_JET)
                    
                    # Add motion overlay with transparency
                    mask = (motion_vis > 50).astype(np.float32) * 0.3
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    viz_frame = viz_frame * (1 - mask) + motion_color * mask
                    viz_frame = viz_frame.astype(np.uint8)
                    
                    # Draw tracked objects
                    for obj_id, position in tracked_objects.items():
                        # Project 3D position to 2D
                        pixel = camera_info.world_to_pixel(np.append(position, 1.0))
                        
                        # Check if pixel is within image bounds
                        if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                            # Draw circle
                            cv2.circle(
                                viz_frame,
                                (int(pixel[0]), int(pixel[1])),
                                20,  # radius
                                (0, 0, 255),  # red in BGR
                                2  # thickness
                            )
                            
                            # Add ID text
                            cv2.putText(
                                viz_frame,
                                f"ID:{obj_id}",
                                (int(pixel[0]) - 20, int(pixel[1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )
                    
                    # Add frame information
                    cv2.putText(
                        viz_frame,
                        f"Frame: {frame_num}/{frame_count} | Objects: {len(tracked_objects)}",
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    # Write frame to video
                    video_writer.write(viz_frame)
                    
                except Exception as e:
                    logger.error(f"Error creating visualization frame: {e}")
            
            # Save detection results periodically
            if save_results and output_dir and processed_count % 30 == 0:
                try:
                    # Export detection results
                    export_manager.export_detection_results(
                        tracked_objects,
                        {obj_id: timestamp for obj_id in tracked_objects},
                        f"detections_{frame_num:04d}.json"
                    )
                except Exception as e:
                    logger.error(f"Error exporting detection results: {e}")
            
            # Update previous frame
            prev_frame = frame.copy()
            
            # Print progress
            progress = frame_num / frame_count * 100
            elapsed = time.time() - processing_start_time
            if processed_count % 10 == 0:
                logger.info(f"Progress: {progress:.1f}% ({processed_count} frames, {len(tracked_objects)} objects)")
    
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        
        if visualize and MATPLOTLIB_AVAILABLE:
            plt.ioff()
        
        # Save final results
        if save_results and output_dir:
            try:
                # Get final trajectories and classifications
                trajectories = tracker.get_trajectories()
                classifications = tracker.classify_trajectories()
                
                # Export trajectory data
                export_manager.export_trajectory_data(
                    trajectories,
                    classifications,
                    "trajectories_final.json"
                )
                
                # Create trajectory visualization
                export_manager.create_trajectory_visualization(
                    trajectories,
                    classifications,
                    "trajectory_visualization.png"
                )
                
                logger.info(f"Results exported to {output_dir}")
            except Exception as e:
                logger.error(f"Error exporting final results: {e}")
    
    # Calculate processing stats
    processing_time = time.time() - processing_start_time
    fps_processing = processed_count / processing_time if processing_time > 0 else 0
    
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(f"Processed {processed_count}/{frame_count} frames ({processed_count/frame_count*100:.1f}%)")
    logger.info(f"Processing speed: {fps_processing:.2f} fps")
    logger.info(f"Detected {len(trajectories)} unique objects")
    
    # Return results
    results = {
        'video_path': video_path,
        'total_frames': frame_count,
        'processed_frames': processed_count,
        'processing_time': processing_time,
        'processing_fps': fps_processing,
        'trajectories': trajectories,
        'objects_detected': len(trajectories)
    }
    
    return results


def process_camera_feed(camera_id: int = 0,
                       output_dir: Optional[str] = None,
                       settings: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None) -> Dict[str, Any]:
    """
    Process a live camera feed for object detection using Pixel Motion Voxel Projection.
    
    Args:
        camera_id: Camera device ID
        output_dir: Directory to save output files
        settings: Dictionary of processing settings
        duration: Maximum duration to process in seconds (None for indefinite)
        
    Returns:
        Dict[str, Any]: Processing results
    """
    if not OPENCV_AVAILABLE:
        logger.error("OpenCV is required for camera processing")
        return {}
        
    if not PIXEL_MOTION_AVAILABLE:
        logger.error("Pixel Motion module is required for processing")
        return {}
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use default settings if not provided
    if settings is None:
        settings = {
            'threshold': 15.0,
            'detection_threshold': 1.0,
            'min_cluster_size': 5,
            'use_gpu': True,
            'resolution': (100, 100, 100),
            'bounds': ((-5, 5), (-5, 5), (0, 10)),
            'max_ray_distance': 10.0,
            'history_length': 5,
            'max_disappeared': 10,
            'max_distance': 1.0,
            'min_motion_size': 20
        }
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_id}")
        return {}
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera {camera_id} opened")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    
    # Create camera info
    camera_info = create_camera_info(width, height, f"Camera{camera_id}")
    
    # Create voxel projector
    voxel_projector = PixelMotionVoxelProjection(
        grid_resolution=settings.get('resolution', (100, 100, 100)),
        grid_bounds=settings.get('bounds', ((-5, 5), (-5, 5), (0, 10))),
        use_gpu=settings.get('use_gpu', True)
    )
    
    # Add camera to projector
    voxel_projector.add_camera(camera_info)
    
    # Create motion filter
    motion_filter = MotionFilter(history_length=settings.get('history_length', 5))
    
    # Create object tracker
    tracker = ObjectTracker(
        max_disappeared=settings.get('max_disappeared', 10),
        max_distance=settings.get('max_distance', 1.0)
    )
    
    # Create UAP visualizer if Open3D is available
    visualizer = None
    if OPEN3D_AVAILABLE:
        visualizer = UAPVisualizer(use_vulkan=True)
        if not visualizer.initialize_visualizer():
            logger.warning("Failed to initialize UAP visualization system")
            visualizer = None
    
    # Set up windows for UAP visualization
    cv2.namedWindow("UAP Detection Monitor", cv2.WINDOW_NORMAL)
    cv2.namedWindow("UAP Motion Signature", cv2.WINDOW_NORMAL)
    
    # Setup video writer if output directory is specified
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(output_dir, f"camera_{camera_id}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps if fps > 0 else 30,
            (width, height)
        )
    else:
        video_writer = None
    
    # Processing loop variables
    prev_frame = None
    frame_count = 0
    processing_start_time = time.time()
    end_time = None if duration is None else processing_start_time + duration
    
    try:
        # Main processing loop
        while True:
            # Check if duration exceeded
            if end_time is not None and time.time() > end_time:
                logger.info(f"Processing duration ({duration}s) reached")
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Skip first frame (no previous frame to compare)
            if prev_frame is None:
                prev_frame = frame.copy()
                continue
            
            # Get current timestamp
            timestamp = time.time()
            
            # Process frame difference
            motion_map = voxel_projector.process_frame_difference(
                prev_frame,
                frame,
                camera_info.name,
                timestamp,
                threshold=settings.get('threshold', 15.0),
                preprocess=True
            )
            
            # Apply temporal filtering
            filtered_map = motion_filter.apply_temporal_filter(motion_map)
            
            # Filter by size to remove small noise
            try:
                filtered_map = motion_filter.filter_by_size(
                    filtered_map, 
                    min_size=settings.get('min_motion_size', 20),
                    max_size=width * height // 4  # Max 1/4 of frame
                )
            except Exception as e:
                logger.warning(f"Size filtering failed: {e}")
            
            # Reset voxel grid
            voxel_projector.reset_voxel_grid()
            
            # Project motion to voxels
            projection = voxel_projector.project_pixels_to_voxels(
                filtered_map,
                max_ray_distance=settings.get('max_ray_distance', 10.0)
            )
            
            # Update voxel grid
            voxel_projector.voxel_grid = projection
            
            # Find objects in voxel space
            objects = voxel_projector.get_detected_objects(
                threshold=settings.get('detection_threshold', 1.0),
                min_cluster_size=settings.get('min_cluster_size', 5)
            )
            
            # Update object tracker
            tracked_objects = tracker.update(objects, timestamp)
            
            # Visualize voxel grid with Open3D if available
            if visualizer is not None:
                visualizer.visualize_voxel_grid(voxel_projector.voxel_grid, threshold=0.5)
                
                if objects:
                    visualizer.add_detected_objects(objects, color=(1, 0, 0), label="Objects")
            
            # Create visualization frame
            motion_vis = cv2.normalize(
                filtered_map.data, 
                None, 
                0, 
                255, 
                cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            motion_color = cv2.applyColorMap(motion_vis, cv2.COLORMAP_JET)
            
            # Create visualization frame for display
            viz_frame = frame.copy()
            
            # Draw tracked objects
            for obj_id, position in tracked_objects.items():
                # Project 3D position to 2D
                pixel = camera_info.world_to_pixel(np.append(position, 1.0))
                
                # Check if pixel is within image bounds
                if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    # Draw circle
                    cv2.circle(
                        viz_frame,
                        (int(pixel[0]), int(pixel[1])),
                        20,  # radius
                        (0, 0, 255),  # red in BGR
                        2  # thickness
                    )
                    
                    # Add ID text
                    cv2.putText(
                        viz_frame,
                        f"ID:{obj_id}",
                        (int(pixel[0]) - 20, int(pixel[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            # Add info text
            cv2.putText(
                viz_frame,
                f"Frame: {frame_count} | Objects: {len(tracked_objects)}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show UAP detection frames
            cv2.imshow("UAP Detection Monitor", viz_frame)
            cv2.imshow("UAP Motion Signature", motion_color)
            
            # Write frame to video if recording
            if video_writer is not None:
                video_writer.write(viz_frame)
            
            # Save frames periodically if requested
            if output_dir and frame_count % 30 == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, viz_frame)
                
                motion_path = os.path.join(output_dir, f"motion_{frame_count:04d}.jpg")
                cv2.imwrite(motion_path, motion_color)
            
            # Update previous frame
            prev_frame = frame.copy()
            
            # Print stats periodically
            if frame_count % 30 == 0:
                elapsed = time.time() - processing_start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames ({current_fps:.2f} fps), {len(tracked_objects)} objects")
            
            # Check for keypress (ESC or 'q' to exit)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                logger.info("User requested stop")
                break
    
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        if visualizer is not None:
            visualizer.close()
        
        # Calculate processing stats
        processing_time = time.time() - processing_start_time
        fps_processing = frame_count / processing_time if processing_time > 0 else 0
        
        logger.info(f"Camera processing completed in {processing_time:.2f} seconds")
        logger.info(f"Processed {frame_count} frames at {fps_processing:.2f} fps")
        logger.info(f"Detected {len(tracker.get_trajectories())} unique objects")
        
        # Save final results if output directory specified
        if output_dir:
            try:
                # Create export manager
                export_manager = ExportManager(output_dir)
                
                # Get trajectories and classifications
                trajectories = tracker.get_trajectories()
                classifications = tracker.classify_trajectories()
                
                # Export trajectory data
                export_manager.export_trajectory_data(
                    trajectories,
                    classifications,
                    "trajectories_final.json"
                )
                
                # Create trajectory visualization
                export_manager.create_trajectory_visualization(
                    trajectories,
                    classifications,
                    "trajectory_visualization.png"
                )
                
                logger.info(f"Results exported to {output_dir}")
            except Exception as e:
                logger.error(f"Error exporting final results: {e}")
    
    # Return results
    results = {
        'camera_id': camera_id,
        'total_frames': frame_count,
        'processing_time': processing_time,
        'processing_fps': fps_processing,
        'trajectories': tracker.get_trajectories(),
        'objects_detected': len(tracker.get_trajectories())
    }
    
    return results


def main():
    """Main function for the Pixel2UAP standalone application."""
    parser = argparse.ArgumentParser(description="Pixel2UAP - UAP Detection and Analysis Toolkit")
    parser.add_argument("--video", type=str, help="Path to video file containing potential UAP footage")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID for live UAP monitoring")
    parser.add_argument("--output", type=str, help="Output directory for UAP analysis results")
    parser.add_argument("--duration", type=float, help="Maximum duration for UAP monitoring session (seconds)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization during UAP analysis")
    parser.add_argument("--threshold", type=float, default=15.0, help="UAP motion detection sensitivity (lower = more sensitive)")
    parser.add_argument("--detection", type=float, default=1.0, help="UAP detection confidence threshold")
    args = parser.parse_args()
    
    # Check if required modules are available
    if not OPENCV_AVAILABLE:
        print("ERROR: OpenCV is required for video processing.")
        return 1
        
    if not PIXEL_MOTION_AVAILABLE:
        print("ERROR: Pixel Motion module is required for processing.")
        return 1
    
    # Create settings
    settings = {
        'threshold': args.threshold,
        'detection_threshold': args.detection,
        'skip_frames': 2,
        'min_cluster_size': 5,
        'use_gpu': True,
        'resolution': (100, 100, 100),
        'bounds': ((-5, 5), (-5, 5), (0, 10)),
        'max_ray_distance': 10.0,
        'history_length': 5,
        'max_disappeared': 10,
        'max_distance': 1.0,
        'min_motion_size': 20
    }
    
    # Create output directory if specified
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.video:
            video_name = os.path.splitext(os.path.basename(args.video))[0]
            output_dir = f"output_{video_name}_{timestamp}"
        else:
            output_dir = f"output_camera{args.camera}_{timestamp}"
    
    # Process video file if specified
    if args.video:
        process_video_file(
            args.video,
            output_dir=output_dir,
            settings=settings,
            visualize=not args.no_viz,
            save_frames=True,
            save_results=True
        )
    else:
        # Process camera feed
        process_camera_feed(
            args.camera,
            output_dir=output_dir,
            settings=settings,
            duration=args.duration
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())