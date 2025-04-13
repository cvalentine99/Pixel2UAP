"""
Visualization utilities for UAP Detection and Tracking (Pixel2UAP).

This module provides specialized visualization tools for displaying UAP detection data,
motion maps, and tracked aerial phenomena in 3D space with trajectory analysis.
"""

import numpy as np
import logging
import os
import platform
from typing import List, Tuple, Dict, Optional, Union, Any

# Setup logger first to avoid undefined issues
logger = logging.getLogger(__name__)

# Disable Vulkan for headless environments or if running in a VM
is_headless = not os.environ.get('DISPLAY', '')
is_vm = platform.platform().lower().find('vm') != -1 or platform.platform().lower().find('virtual') != -1

# Configure renderer based on environment
if is_headless or is_vm:
    os.environ["OPEN3D_ENABLE_VULKAN"] = "0"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Software rendering for OpenGL
    os.environ["MPLBACKEND"] = "Agg"  # Non-interactive Matplotlib backend
else:
    # Try to use Vulkan but have a fallback
    os.environ["OPEN3D_ENABLE_VULKAN"] = "1"

from .data_structures import VoxelGrid, CameraInfo, MotionMap

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D is not available. 3D visualization will be limited.")

try:
    import matplotlib
    # Don't try to set a specific backend as it may conflict with Qt
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib is not available. 2D visualization will be limited.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV is not available. Video processing will be limited.")


class UAPVisualizer:
    """
    Advanced visualization tools for UAP detection and trajectory analysis.
    
    Provides methods for rendering UAP data in 3D space, including detected objects,
    movement trajectories, and observation positions using Open3D or Matplotlib.
    
    Attributes:
        use_vulkan (bool): Whether to use Vulkan rendering backend for enhanced performance
        visualizer (Any): Open3D visualizer object when using Open3D
        figure (Any): Matplotlib figure when using Matplotlib
        point_cloud (Any): Open3D point cloud for the UAP detection data
        object_markers (List): UAP objects added to the visualization
    """
    
    def __init__(self, use_vulkan: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            use_vulkan (bool): Whether to use Vulkan rendering backend
        """
        self.use_vulkan = use_vulkan
        
        if use_vulkan:
            os.environ["OPEN3D_ENABLE_VULKAN"] = "1"
        else:
            os.environ["OPEN3D_ENABLE_VULKAN"] = "0"
            
        self.visualizer = None
        self.figure = None
        self.point_cloud = None
        self.object_markers = []
        
    def initialize_visualizer(self) -> bool:
        """
        Set up the visualization environment.
        
        Returns:
            bool: True if initialization was successful
        """
        # Always use matplotlib for initial visualization to avoid window issues
        # Open3D visualizer can be unstable in some environments
        return self._initialize_matplotlib()
        
        # The code below is preserved but disabled for better stability
        # If you want to try Open3D visualization, you can modify this method
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available for 3D visualization")
            return self._initialize_matplotlib()
            
        try:
            # Initialize Open3D visualizer
            self.visualizer = o3d.visualization.Visualizer()
            success = self.visualizer.create_window(
                window_name="Pixel2UAP - UAP Detection & Analysis",
                width=1024,
                height=768
            )
            
            if not success:
                logger.warning("Failed to create Open3D window, falling back to Matplotlib")
                return self._initialize_matplotlib()
                
            # Set render options for enhanced UAP visualization
            opt = self.visualizer.get_render_option()
            opt.background_color = np.array([0.05, 0.05, 0.12])  # Deep space blue background
            opt.point_size = 6.0  # Larger points for better UAP visibility
            opt.light_on = True   # Enable lighting for better 3D perception
            
            # Add coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            self.visualizer.add_geometry(frame)
            
            # Create empty point cloud for voxel data
            self.point_cloud = o3d.geometry.PointCloud()
            self.visualizer.add_geometry(self.point_cloud)
            
            # Set up view
            view_control = self.visualizer.get_view_control()
            view_control.set_front([0, 0, -1])  # Look toward negative z-axis
            view_control.set_up([0, -1, 0])     # Up is negative y-axis
            view_control.set_zoom(0.8)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Open3D visualizer: {e}")
            return self._initialize_matplotlib()
        """
    
    def _initialize_matplotlib(self) -> bool:
        """
        Initialize Matplotlib for fallback visualization.
        
        Returns:
            bool: True if initialization was successful
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Neither Open3D nor Matplotlib available for visualization")
            return False
            
        try:
            # Use non-interactive renderer that works with Qt
            import matplotlib
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            # Create a matplotlib figure that doesn't require a GUI backend
            self.figure = Figure(figsize=(12, 9))
            self.canvas = FigureCanvas(self.figure)
            ax = self.figure.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('UAP Detection & Trajectory Analysis')
            
            # Store axes for later use
            self.axes = ax
            
            # Set up a nice dark theme for the plot with high contrast for UAP visibility
            self.figure.patch.set_facecolor('black')
            ax.set_facecolor((0.05, 0.05, 0.1))  # Deep space blue
            ax.grid(False)
            
            # Make axes and grid more visible
            try:
                # Set colors and tick parameters
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                
                ax.xaxis.pane.set_edgecolor('white')
                ax.yaxis.pane.set_edgecolor('white')
                ax.zaxis.pane.set_edgecolor('white')
            except Exception as pane_err:
                # Some versions of matplotlib have different APIs
                logger.debug(f"Could not set pane colors: {pane_err}")
                # Alternative method for older matplotlib
                try:
                    ax.w_xaxis.set_pane_color((0.1, 0.1, 0.2, 0.3))
                    ax.w_yaxis.set_pane_color((0.1, 0.1, 0.2, 0.3))
                    ax.w_zaxis.set_pane_color((0.1, 0.1, 0.2, 0.3))
                except:
                    pass  # Ignore if this fails too
            
            # These should work on all matplotlib versions
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='z', colors='white')
            
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            
            # Add a "no data" message until we have data
            ax.text(0, 0, 0, "No UAP data available yet", 
                   color='white', fontsize=12, ha='center')
            
            # Set empty lists for scatter plots
            self.scatter_points = None
            self.object_markers = []
            
            # Create initial render
            self.canvas.draw()
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Matplotlib visualizer: {e}")
            # Log the detailed error for debugging
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def visualize_voxel_grid(self, voxel_grid: VoxelGrid, threshold: float = 0.1) -> bool:
        """
        Visualize a voxel grid.
        
        Args:
            voxel_grid (VoxelGrid): The voxel grid to visualize
            threshold (float): Value threshold for considering voxels
            
        Returns:
            bool: True if visualization was successful
        """
        if self.visualizer is None and self.figure is None:
            if not self.initialize_visualizer():
                return False
                
        if self.visualizer is not None:
            return self._visualize_with_open3d(voxel_grid, threshold)
        else:
            return self._visualize_with_matplotlib(voxel_grid, threshold)
    
    def _visualize_with_open3d(self, voxel_grid: VoxelGrid, threshold: float) -> bool:
        """
        Visualize using Open3D.
        
        Args:
            voxel_grid (VoxelGrid): The voxel grid to visualize
            threshold (float): Value threshold for considering voxels
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Update point cloud from voxel grid
            new_cloud = voxel_grid.to_point_cloud()
            
            if new_cloud is None:
                logger.warning("Failed to convert voxel grid to point cloud")
                return False
                
            self.point_cloud.points = new_cloud.points
            self.point_cloud.colors = new_cloud.colors
            self.visualizer.update_geometry(self.point_cloud)
            
            # Update visualization
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Open3D visualization: {e}")
            return False
    
    def _visualize_with_matplotlib(self, voxel_grid: VoxelGrid, threshold: float) -> bool:
        """
        Visualize using Matplotlib.
        
        Args:
            voxel_grid (VoxelGrid): The voxel grid to visualize
            threshold (float): Value threshold for considering voxels
            
        Returns:
            bool: True if visualization was successful
        """
        try:
            # Clear current plot
            self.axes.clear()
            
            # Basic setup
            self.axes.set_xlabel('X')
            self.axes.set_ylabel('Y')
            self.axes.set_zlabel('Z')
            self.axes.set_title('UAP Detection Space', color='white')
            
            # Find non-zero voxels
            indices = np.where(voxel_grid.grid > threshold)
            
            if len(indices[0]) == 0:
                # No points to visualize, but still have a nice empty view
                self.axes.text(0, 0, 0, "No UAP signatures detected", 
                              color='white', fontsize=12, ha='center')
                
                # Set consistent bounds for empty grid
                bounds = voxel_grid.bounds
                if bounds:
                    self.axes.set_xlim(bounds[0])
                    self.axes.set_ylim(bounds[1])  
                    self.axes.set_zlim(bounds[2])
                else:
                    self.axes.set_xlim([-5, 5])
                    self.axes.set_ylim([-5, 5])
                    self.axes.set_zlim([0, 10])
                    
                # Add grid and camera origin for reference
                self.axes.scatter(0, 0, 0, c='yellow', marker='^', s=100, label='Observer')
                
                # Draw grid lines
                for i in range(-5, 6, 1):
                    self.axes.plot([-5, 5], [i, i], [0, 0], 'gray', alpha=0.3)
                    self.axes.plot([i, i], [-5, 5], [0, 0], 'gray', alpha=0.3)
                
            else:
                # Convert indices to world coordinates
                points = []
                values = []
                
                for i, j, k in zip(indices[0], indices[1], indices[2]):
                    point = voxel_grid.voxel_to_world((i, j, k))
                    points.append(point)
                    values.append(voxel_grid.grid[i, j, k])
                    
                points = np.array(points)
                values = np.array(values)
                
                # Normalize values for color mapping
                if np.max(values) > 0:
                    norm_values = values / np.max(values)
                else:
                    norm_values = values
                
                # Create scatter plot with enhanced UAP colors
                self.scatter_points = self.axes.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=norm_values, 
                    cmap='plasma',  # Use a more celestial/alien colormap
                    alpha=0.7,
                    s=50 * norm_values + 10,  # Size based on value with minimum size
                    edgecolors='white',  # Add white outlines
                    linewidths=0.3      # Thin outlines
                )
                
                # Set view limits based on data and voxel grid bounds
                bounds = voxel_grid.bounds
                self.axes.set_xlim(bounds[0])
                self.axes.set_ylim(bounds[1])
                self.axes.set_zlim(bounds[2])
                
                # Add observer point (camera)
                self.axes.scatter(0, 0, 0, c='yellow', marker='^', s=100, label='Observer')
                
                # Add count label
                self.axes.text(
                    bounds[0][0], bounds[1][0], bounds[2][1] * 0.9,
                    f"UAP Signatures: {len(points)}",
                    color='white', fontsize=10, ha='left'
                )
            
            # Consistent styling
            # These should work on all matplotlib versions
            self.axes.tick_params(axis='x', colors='white')
            self.axes.tick_params(axis='y', colors='white')
            self.axes.tick_params(axis='z', colors='white')
            
            self.axes.xaxis.label.set_color('white')
            self.axes.yaxis.label.set_color('white')
            self.axes.zaxis.label.set_color('white')
            
            # Create rendering (using FigureCanvasAgg)
            self.canvas.draw()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating Matplotlib visualization: {e}")
            # Log the detailed error for debugging
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def add_cameras(self, cameras: Dict[str, CameraInfo]) -> bool:
        """
        Add camera positions and orientations to the visualization.
        
        Args:
            cameras (Dict[str, CameraInfo]): Dictionary of cameras to visualize
            
        Returns:
            bool: True if cameras were added successfully
        """
        if self.visualizer is None and self.figure is None:
            if not self.initialize_visualizer():
                return False
                
        if self.visualizer is not None:
            return self._add_cameras_open3d(cameras)
        else:
            return self._add_cameras_matplotlib(cameras)
    
    def _add_cameras_open3d(self, cameras: Dict[str, CameraInfo]) -> bool:
        """
        Add cameras to Open3D visualization.
        
        Args:
            cameras (Dict[str, CameraInfo]): Dictionary of cameras to visualize
            
        Returns:
            bool: True if cameras were added successfully
        """
        try:
            # Add a camera frustum for each camera
            for name, camera in cameras.items():
                # Create a camera frustum mesh
                frustum = self._create_camera_frustum(camera)
                self.visualizer.add_geometry(frustum)
                
                # Add to object markers list to track added objects
                self.object_markers.append(frustum)
                
            # Update visualization
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding cameras to Open3D visualization: {e}")
            return False
    
    def _create_camera_frustum(self, camera: CameraInfo) -> o3d.geometry.TriangleMesh:
        """
        Create a camera frustum mesh for visualization.
        
        Args:
            camera (CameraInfo): Camera information
            
        Returns:
            o3d.geometry.TriangleMesh: Camera frustum mesh
        """
        # Create a simple camera frustum representation
        # Size based on focal length
        size = 0.2
        
        # Camera pyramid vertices
        vertices = np.array([
            [0, 0, 0],  # Camera position
            [size, size, size],  # Top-right
            [size, -size, size],  # Bottom-right
            [-size, -size, size],  # Bottom-left
            [-size, size, size]  # Top-left
        ])
        
        # Apply camera orientation and position
        for i in range(1, 5):
            vertices[i] = camera.orientation @ vertices[i] + camera.position
            
        # Camera position
        vertices[0] = camera.position
        
        # Create mesh with triangles
        frustum = o3d.geometry.TriangleMesh()
        
        # Set vertices
        frustum.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Set triangles (faces of the pyramid)
        triangles = np.array([
            [0, 1, 2],  # Right face
            [0, 2, 3],  # Bottom face
            [0, 3, 4],  # Left face
            [0, 4, 1],  # Top face
            [1, 4, 3],  # Front-top
            [1, 3, 2]   # Front-bottom
        ])
        
        frustum.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Set color based on camera name (hash to get consistent color)
        name_hash = hash(camera.name) % 1000 / 1000.0
        color = np.array([0.7 + 0.3 * name_hash, 0.3, 0.3 + 0.7 * (1 - name_hash)])
        frustum.paint_uniform_color(color)
        
        # Compute normals for proper rendering
        frustum.compute_vertex_normals()
        
        return frustum
    
    def _add_cameras_matplotlib(self, cameras: Dict[str, CameraInfo]) -> bool:
        """
        Add cameras to Matplotlib visualization.
        
        Args:
            cameras (Dict[str, CameraInfo]): Dictionary of cameras to visualize
            
        Returns:
            bool: True if cameras were added successfully
        """
        try:
            # Add a marker and frustum for each camera
            for name, camera in cameras.items():
                # Add camera position
                self.axes.scatter(
                    camera.position[0], 
                    camera.position[1], 
                    camera.position[2],
                    color='yellow',
                    marker='o',
                    s=100,
                    label=name
                )
                
                # Add simple frustum lines
                self._add_camera_frustum_matplotlib(camera)
                
            # Add legend
            self.axes.legend()
            
            # Update figure
            plt.draw()
            plt.pause(0.001)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding cameras to Matplotlib visualization: {e}")
            return False
    
    def _add_camera_frustum_matplotlib(self, camera: CameraInfo) -> None:
        """
        Add a camera frustum to the Matplotlib visualization.
        
        Args:
            camera (CameraInfo): Camera information
        """
        # Create a simple camera frustum representation
        size = 0.2
        
        # Camera pyramid vertices
        vertices = np.array([
            [0, 0, 0],  # Camera position
            [size, size, size],  # Top-right
            [size, -size, size],  # Bottom-right
            [-size, -size, size],  # Bottom-left
            [-size, size, size]  # Top-left
        ])
        
        # Apply camera orientation and position
        for i in range(1, 5):
            vertices[i] = camera.orientation @ vertices[i] + camera.position
            
        # Camera position
        vertices[0] = camera.position
        
        # Add lines from camera to frustum corners
        for i in range(1, 5):
            self.axes.plot(
                [vertices[0][0], vertices[i][0]],
                [vertices[0][1], vertices[i][1]],
                [vertices[0][2], vertices[i][2]],
                color='yellow',
                linewidth=1
            )
            
        # Add lines between frustum corners (front face)
        for i in range(1, 5):
            next_i = 1 if i == 4 else i + 1
            self.axes.plot(
                [vertices[i][0], vertices[next_i][0]],
                [vertices[i][1], vertices[next_i][1]],
                [vertices[i][2], vertices[next_i][2]],
                color='yellow',
                linewidth=1
            )
    
    def visualize_motion_map(self, motion_map: MotionMap, title: str = "UAP Motion Detection") -> None:
        """
        Visualize a 2D UAP motion detection map using Matplotlib or OpenCV.
        
        The visualization shows areas where UAP movement has been detected in the
        video frame, with color intensity indicating the strength of the detection signal.
        
        Args:
            motion_map (MotionMap): The UAP motion map to visualize
            title (str): Title for the visualization
        """
        if not MATPLOTLIB_AVAILABLE and not OPENCV_AVAILABLE:
            logger.warning("Neither Matplotlib nor OpenCV available for motion map visualization")
            return
            
        # Normalize motion map for visualization
        motion_data = motion_map.data.copy()
        if np.max(motion_data) > 0:
            motion_data = motion_data / np.max(motion_data) * 255
            
        motion_data = motion_data.astype(np.uint8)
        
        if MATPLOTLIB_AVAILABLE:
            # Create a new figure for the motion map
            plt.figure(figsize=(10, 8))
            plt.imshow(motion_data, cmap='jet')
            plt.colorbar(label='Motion Intensity')
            plt.title(title)
            
            # Get motion pixels and highlight them
            motion_pixels = motion_map.get_motion_pixels()
            if len(motion_pixels) > 0:
                plt.scatter(
                    motion_pixels[:, 0],
                    motion_pixels[:, 1],
                    s=10,
                    color='white',
                    alpha=0.5,
                    marker='.',
                    label='Motion Pixels'
                )
                
            plt.legend()
            plt.show()
        else:
            # Use OpenCV for visualization
            colored_motion = cv2.applyColorMap(motion_data, cv2.COLORMAP_JET)
            
            # Add text
            cv2.putText(
                colored_motion,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
            
            # Show image
            cv2.imshow(title, colored_motion)
            cv2.waitKey(1)  # Small wait time to allow rendering
    
    def add_detected_objects(self, objects: List[np.ndarray], 
                            color: Tuple[float, float, float] = (0.0, 1.0, 0.5),
                            label: str = "Detected UAPs") -> bool:
        """
        Add detected UAP objects to the 3D visualization.
        
        This method adds visual markers for each detected UAP at its 3D position,
        allowing for spatial tracking and trajectory analysis.
        
        Args:
            objects (List[np.ndarray]): List of 3D UAP positions
            color (Tuple[float, float, float]): RGB color for UAP markers (default: green)
            label (str): Label for the UAPs in the visualization legend
            
        Returns:
            bool: True if UAP objects were added successfully
        """
        if self.visualizer is None and self.figure is None:
            if not self.initialize_visualizer():
                return False
                
        if self.visualizer is not None:
            return self._add_objects_open3d(objects, color, label)
        else:
            return self._add_objects_matplotlib(objects, color, label)
    
    def _add_objects_open3d(self, objects: List[np.ndarray], 
                           color: Tuple[float, float, float],
                           label: str) -> bool:
        """
        Add objects to Open3D visualization.
        
        Args:
            objects (List[np.ndarray]): List of 3D object positions
            color (Tuple[float, float, float]): RGB color for object markers
            label (str): Label for the objects in the visualization
            
        Returns:
            bool: True if objects were added successfully
        """
        try:
            if not objects:
                return True
                
            # Create a sphere for each UAP object with glowing effect
            for i, obj_pos in enumerate(objects):
                # Create larger sphere for UAP visualization
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
                sphere.translate(obj_pos)
                
                # Use brighter color for UAP visualization
                uap_color = color
                if color == (0.0, 1.0, 0.5):  # Default UAP color
                    # Make color more vibrant for visibility
                    uap_color = (0.0, 1.0, 0.7)
                
                sphere.paint_uniform_color(uap_color)
                
                self.visualizer.add_geometry(sphere)
                self.object_markers.append(sphere)
                
            # Update visualization
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding objects to Open3D visualization: {e}")
            return False
    
    def _add_objects_matplotlib(self, objects: List[np.ndarray], 
                              color: Tuple[float, float, float],
                              label: str) -> bool:
        """
        Add objects to Matplotlib visualization.
        
        Args:
            objects (List[np.ndarray]): List of 3D object positions
            color (Tuple[float, float, float]): RGB color for object markers
            label (str): Label for the objects in the visualization
            
        Returns:
            bool: True if objects were added successfully
        """
        try:
            if not objects:
                return True
                
            # Convert list of positions to array
            positions = np.array(objects)
            
            # Add scatter plot for UAP objects with enhanced visibility
            self.axes.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                marker='*',     # Star marker for UAPs
                s=150,          # Larger size
                label=label,
                edgecolors='white',
                linewidth=1.5,  # Thicker outline
                alpha=0.9       # Slightly transparent
            )
            
            # Update legend
            self.axes.legend()
            
            # Update figure
            plt.draw()
            plt.pause(0.001)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding objects to Matplotlib visualization: {e}")
            return False
    
    def save_screenshot(self, filename: str) -> bool:
        """
        Save a screenshot of the current visualization.
        
        Args:
            filename (str): Path to save the screenshot
            
        Returns:
            bool: True if screenshot was saved successfully
        """
        if self.visualizer is not None:
            try:
                return self.visualizer.capture_screen_image(filename)
            except Exception as e:
                logger.error(f"Error saving Open3D screenshot: {e}")
                return False
        elif self.figure is not None:
            try:
                self.figure.savefig(filename, dpi=300)
                return True
            except Exception as e:
                logger.error(f"Error saving Matplotlib screenshot: {e}")
                return False
        else:
            logger.warning("No visualization active, cannot save screenshot")
            return False
    
    def create_animation(self, voxel_grids: List[VoxelGrid], 
                        output_path: str, 
                        threshold: float = 0.1,
                        fps: int = 15) -> bool:
        """
        Create a UAP trajectory animation from a sequence of detection frames.
        
        This generates a high-quality video showing the movement of UAPs through 3D space
        over time, ideal for analyzing flight patterns and trajectory characteristics.
        
        Args:
            voxel_grids (List[VoxelGrid]): Sequence of UAP detection voxel grids
            output_path (str): Path to save the animation video
            threshold (float): Value threshold for UAP detection sensitivity
            fps (int): Frames per second (higher values create smoother trajectories)
            
        Returns:
            bool: True if UAP trajectory animation was created successfully
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV required for animation creation")
            return False
            
        try:
            # Initialize visualizer if needed
            if self.visualizer is None and self.figure is None:
                if not self.initialize_visualizer():
                    return False
            
            # Create a temporary directory for frames
            import tempfile
            import os
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Visualize each frame and save screenshot
                frame_files = []
                
                for i, grid in enumerate(voxel_grids):
                    # Update visualization
                    self.visualize_voxel_grid(grid, threshold)
                    
                    # Save screenshot
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    self.save_screenshot(frame_path)
                    frame_files.append(frame_path)
                    
                    logger.info(f"Rendered frame {i+1}/{len(voxel_grids)}")
                
                # Combine frames into video
                if not frame_files:
                    logger.warning("No frames rendered for animation")
                    return False
                    
                # Read first frame to get dimensions
                first_frame = cv2.imread(frame_files[0])
                height, width, _ = first_frame.shape
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Add frames to video
                for frame_path in frame_files:
                    video.write(cv2.imread(frame_path))
                    
                # Release video writer
                video.release()
                
                logger.info(f"Animation saved to {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
            return False
    
    def close(self) -> None:
        """Close the visualizer and release resources."""
        if self.visualizer is not None:
            self.visualizer.destroy_window()
            self.visualizer = None
            
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None