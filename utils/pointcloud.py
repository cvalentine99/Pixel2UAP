"""
Point cloud generation and processing functionality for Voxel Projector

This module provides utilities for generating point clouds from depth data,
particularly from OAK-D cameras. It supports both on-device and host-based
point cloud generation methods.
"""

import os
import numpy as np
import cv2
import logging
from pathlib import Path

# Optional imports that may not be available
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class PointCloudVisualizer:
    """Point cloud visualization and processing using Open3D

    This class handles the creation and visualization of point clouds from depth data,
    supporting RGB-D projection and basic point cloud operations like filtering and
    downsampling.
    
    Attributes:
        pcl (o3d.geometry.PointCloud): The point cloud object
        pinhole_camera_intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics
        vis (o3d.visualization.Visualizer): Open3D visualizer object
        R_camera_to_world (np.ndarray): Rotation matrix to transform from camera to world coordinates
    """
    def __init__(self, intrinsic_matrix=None, width=None, height=None):
        """Initialize the point cloud visualizer

        Args:
            intrinsic_matrix: Camera intrinsic matrix (3x3)
            width: Image width
            height: Image height
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("open3d is required for point cloud visualization")
        
        # Define rotation matrix to convert from camera to world coordinates
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.depth_map = None
        self.rgb = None
        self.pcl = o3d.geometry.PointCloud()
        
        # Initialize visualization window with proper error handling
        try:
            # Check if we should use Vulkan renderer
            if os.environ.get("OPEN3D_ENABLE_VULKAN", "0") == "1":
                logger.info("Using Vulkan renderer for point cloud visualization")
                # Vulkan settings could be configured here if needed
            
            self.vis = o3d.visualization.Visualizer()
            window_created = self.vis.create_window(window_name="Point Cloud", width=800, height=600, visible=True)
            
            if not window_created:
                logger.warning("Failed to create Open3D visualization window. Will operate in headless mode.")
                # Create offscreen renderer as fallback
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(width=800, height=600, visible=False)
                self.headless_mode = True
            else:
                self.headless_mode = False
                
            self.vis.add_geometry(self.pcl)
            
            # Add coordinate frame
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            
            # Configure view
            view_control = self.vis.get_view_control()
            view_control.set_constant_z_far(1000)
            
            self.vis_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing Open3D visualizer: {e}")
            self.vis_initialized = False
            self.headless_mode = True
        
        # Set up camera intrinsics if provided
        if intrinsic_matrix is not None and width is not None and height is not None:
            self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                intrinsic_matrix[0][0],
                intrinsic_matrix[1][1],
                intrinsic_matrix[0][2],
                intrinsic_matrix[1][2]
            )
        else:
            self.pinhole_camera_intrinsic = None
    
    def setup_intrinsics(self, intrinsic_matrix, width, height):
        """Set or update the camera intrinsics

        Args:
            intrinsic_matrix: Camera intrinsic matrix (3x3)
            width: Image width
            height: Image height
        """
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic_matrix[0][0],
            intrinsic_matrix[1][1],
            intrinsic_matrix[0][2],
            intrinsic_matrix[1][2]
        )
    
    def rgbd_to_projection(self, depth_map, rgb, downsample=True, remove_noise=False):
        """Convert RGB-D data to a point cloud

        Args:
            depth_map: Depth map (2D numpy array)
            rgb: RGB image (3D numpy array)
            downsample: Whether to downsample the point cloud for better performance
            remove_noise: Whether to remove statistical outliers
            
        Returns:
            o3d.geometry.PointCloud: The generated point cloud
        """
        if self.pinhole_camera_intrinsic is None:
            raise ValueError("Camera intrinsics must be set before creating point cloud")
        
        # Create Open3D images
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, 
            convert_rgb_to_intensity=(len(rgb.shape) != 3), 
            depth_trunc=20000, 
            depth_scale=1000.0
        )
        
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            self.pinhole_camera_intrinsic
        )
        
        # Downsample for better performance
        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # Remove noise
        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]
        
        # Update internal point cloud
        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors
        
        # Transform to world coordinates
        self.pcl.rotate(self.R_camera_to_world, center=np.array([0, 0, 0], dtype=np.float64))
        
        return self.pcl
    
    def visualize_pcd(self):
        """Update and visualize the point cloud"""
        if not hasattr(self, 'vis_initialized') or not self.vis_initialized:
            logger.warning("Cannot visualize point cloud: visualizer not properly initialized")
            return False
            
        try:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()
            return True
        except Exception as e:
            logger.error(f"Error visualizing point cloud: {e}")
            return False
    
    def save_pointcloud(self, filename):
        """Save the point cloud to a file

        Args:
            filename: Output filename (should end with .pcd, .ply, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            o3d.io.write_point_cloud(filename, self.pcl, compressed=True)
            return True
        except Exception as e:
            logger.error(f"Error saving point cloud: {e}")
            return False
    
    def close_window(self):
        """Close the visualization window"""
        if hasattr(self, 'vis') and self.vis is not None:
            try:
                self.vis.destroy_window()
                logger.info("Point cloud visualization window closed")
            except Exception as e:
                logger.error(f"Error closing point cloud window: {e}")


class DevicePointCloudGenerator:
    """On-device point cloud generation using OAK-D's depth

    This class handles point cloud generation directly on the OAK-D device
    using its neural engine, which is more efficient than host-based generation.
    
    Attributes:
        device (dai.Device): DepthAI device instance
        pipeline (dai.Pipeline): DepthAI pipeline for point cloud generation
        xyz (np.ndarray): XYZ coordinate grid
    """
    def __init__(self, device=None, resolution=(640, 400)):
        """Initialize the point cloud generator

        Args:
            device: DepthAI device instance (if None, a new one will be created)
            resolution: Resolution for the depth map (width, height)
        """
        if not DEPTHAI_AVAILABLE:
            raise ImportError("depthai is required for on-device point cloud generation")
            
        self.resolution = resolution
        self.device = device
        self.pipeline = None
        self.xyz = None
        self.model_path = None
        
        # Create XYZ grid coordinate system
        self._create_xyz_grid()
    
    def _create_xyz_grid(self):
        """Create the XYZ coordinate grid based on camera intrinsics"""
        # We'll need actual device intrinsics, but for now use a placeholder
        # This will be updated when a device is available
        fx, fy = 500.0, 500.0  # placeholder focal lengths
        cx, cy = self.resolution[0]/2, self.resolution[1]/2  # placeholder centers
        
        # Create grid
        xs = np.linspace(0, self.resolution[0] - 1, self.resolution[0], dtype=np.float32)
        ys = np.linspace(0, self.resolution[1] - 1, self.resolution[1], dtype=np.float32)
        
        # Generate grid by stacking coordinates
        base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
        points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2
        
        # Unpack coordinates
        u_coord = points_2d[..., 0]
        v_coord = points_2d[..., 1]
        
        # Projective transformation
        x_coord = (u_coord - cx) / fx
        y_coord = (v_coord - cy) / fy
        
        # Create xyz coordinates
        self.xyz = np.stack([x_coord, y_coord], axis=-1)
        self.xyz = np.pad(self.xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    
    def update_intrinsics(self, camera_matrix):
        """Update the XYZ grid with actual camera intrinsics

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        # Extract camera parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Recreate grid with actual parameters
        xs = np.linspace(0, self.resolution[0] - 1, self.resolution[0], dtype=np.float32)
        ys = np.linspace(0, self.resolution[1] - 1, self.resolution[1], dtype=np.float32)
        
        base_grid = np.stack(np.meshgrid(xs, ys))
        points_2d = base_grid.transpose(1, 2, 0)
        
        u_coord = points_2d[..., 0]
        v_coord = points_2d[..., 1]
        
        x_coord = (u_coord - cx) / fx
        y_coord = (v_coord - cy) / fy
        
        self.xyz = np.stack([x_coord, y_coord], axis=-1)
        self.xyz = np.pad(self.xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
    
    def setup_pipeline(self):
        """Set up the DepthAI pipeline for point cloud generation
        
        This creates a pipeline with:
        1. Stereo depth nodes for depth calculation
        2. Neural network node for on-device point cloud generation
        3. Color camera node for RGB data (optional)
        
        Returns:
            dai.Pipeline: The configured pipeline
        """
        if not DEPTHAI_AVAILABLE:
            raise ImportError("depthai is required for on-device point cloud generation")
            
        # Create pipeline
        pipeline = dai.Pipeline()
        
        # Configure Camera Properties
        left = pipeline.createMonoCamera()
        left.setResolution(self._get_resolution_enum(self.resolution[1]))
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        
        right = pipeline.createMonoCamera()
        right.setResolution(self._get_resolution_enum(self.resolution[1]))
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Create stereo depth
        stereo = pipeline.createStereoDepth()
        self._configure_depth_postprocessing(stereo)
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        
        # Color camera (optional)
        camRgb = pipeline.createColorCamera()
        camRgb.setIspScale(1, 3)
        rgbOut = pipeline.createXLinkOut()
        rgbOut.setStreamName("rgb")
        camRgb.isp.link(rgbOut.input)
        
        # Get or create the point cloud model
        model_path = self._get_model_path()
        
        # Neural network for point cloud generation
        nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(model_path)
        stereo.depth.link(nn.inputs["depth"])
        
        # Input for XYZ grid
        xyz_in = pipeline.createXLinkIn()
        xyz_in.setMaxDataSize(6144000)
        xyz_in.setStreamName("xyz_in")
        xyz_in.out.link(nn.inputs["xyz"])
        
        # Only send xyz data once, and always reuse the message
        nn.inputs["xyz"].setReusePreviousMessage(True)
        
        # Output for point cloud
        pointsOut = pipeline.createXLinkOut()
        pointsOut.setStreamName("pcl")
        nn.out.link(pointsOut.input)
        
        self.pipeline = pipeline
        return pipeline
    
    def _get_resolution_enum(self, height):
        """Convert height value to DepthAI resolution enum
        
        Args:
            height: Height in pixels
            
        Returns:
            dai.MonoCameraProperties.SensorResolution: Resolution enum
        """
        if height == 480:
            return dai.MonoCameraProperties.SensorResolution.THE_480_P
        elif height == 720:
            return dai.MonoCameraProperties.SensorResolution.THE_720_P
        elif height == 800:
            return dai.MonoCameraProperties.SensorResolution.THE_800_P
        else:
            return dai.MonoCameraProperties.SensorResolution.THE_400_P
    
    def _get_model_path(self):
        """Get or create the neural network model for point cloud generation
        
        Returns:
            str: Path to the model blob file
        """
        # TODO: Implement model creation here (deferred until depthai integration)
        # This would require heavy dependencies like torch, onnx, etc.
        # For now, we'll assume the model is available at a predefined path
        
        # Path to look for and store model
        model_dir = Path.home() / ".voxel_projector" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = f"pointcloud_{self.resolution[0]}x{self.resolution[1]}.blob"
        model_path = model_dir / model_name
        
        # Check if model exists, otherwise it needs to be created
        if not model_path.exists():
            raise FileNotFoundError(
                f"Point cloud model not found at {model_path}. "
                "Please download the model or create it using the depthai-experiments repository."
            )
        
        self.model_path = str(model_path)
        return self.model_path
    
    def _configure_depth_postprocessing(self, stereo_depth_node):
        """Configure depth post-processing parameters
        
        Args:
            stereo_depth_node: StereoDepth node to configure
        """
        stereo_depth_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        
        config = stereo_depth_node.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 60
        config.postProcessing.temporalFilter.enable = True
        
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = 700  # mm
        config.postProcessing.thresholdFilter.maxRange = 4000  # mm
        config.censusTransform.enableMeanMode = True
        config.costMatching.linearEquationParameters.alpha = 0
        config.costMatching.linearEquationParameters.beta = 2
        
        stereo_depth_node.initialConfig.set(config)
        stereo_depth_node.setLeftRightCheck(True)
        stereo_depth_node.setExtendedDisparity(False)
        stereo_depth_node.setSubpixel(True)
        stereo_depth_node.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    
    def start(self):
        """Start point cloud generation with the device
        
        Returns:
            tuple: (queue, device) where queue is for receiving point cloud data
        """
        if self.device is None:
            # Create new device with the pipeline
            if self.pipeline is None:
                self.setup_pipeline()
            
            self.device = dai.Device(self.pipeline)
        
        # Get camera intrinsics from the device
        calib_data = self.device.readCalibration()
        intrinsics = calib_data.getCameraIntrinsics(
            dai.CameraBoardSocket.RIGHT,
            dai.Size2f(self.resolution[0], self.resolution[1])
        )
        
        # Update our XYZ grid with the actual intrinsics
        self.update_intrinsics(np.array(intrinsics).reshape(3, 3))
        
        # Send the XYZ grid to the device
        matrix = np.array([self.xyz], dtype=np.float16).view(np.int8)
        buff = dai.Buffer()
        buff.setData(matrix)
        self.device.getInputQueue("xyz_in").send(buff)
        
        # Get output queue for point cloud data
        queue = self.device.getOutputQueue("pcl", maxSize=8, blocking=False)
        rgb_queue = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
        
        return queue, rgb_queue
    
    def process_results(self, pcl_data):
        """Process point cloud data from the device
        
        Args:
            pcl_data: Raw point cloud data from the device
            
        Returns:
            np.ndarray: Processed point cloud data (Nx3)
        """
        # Reshape and scale the point cloud data
        points = np.array(pcl_data.getFirstLayerFp16()).reshape(
            1, 3, self.resolution[1], self.resolution[0]
        )
        
        # Convert to Nx3 format and scale to meters
        points = points.reshape(3, -1).T.astype(np.float64) / 1000.0
        
        return points