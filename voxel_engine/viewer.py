"""
Visualization components for voxel data
"""

import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from ..utils.profiling import timeit
from ..utils.errors import RenderingError

logger = logging.getLogger(__name__)

try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
    logger.info("PyVista visualization available")
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista not available, falling back to Matplotlib")

try:
    from pyvistaqt import QtInteractor, BackgroundPlotter
    PYVISTAQT_AVAILABLE = True
    logger.info("PyVistaQt available for Qt integration")
except ImportError:
    PYVISTAQT_AVAILABLE = False
    logger.warning("PyVistaQt not available, Qt integration limited")


class MatplotlibVoxelViewer:
    """
    Matplotlib-based voxel visualization
    
    Used as a fallback when PyVista is not available
    """
    
    def __init__(self):
        """Initialize the Matplotlib viewer"""
        self.points = None
        self.intensities = None
        self.fig = None
        self.canvas = None
    
    def create_canvas(self):
        """Create a Matplotlib canvas for embedding in Qt"""
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        return self.canvas
    
    @timeit
    def plot_3d_voxels(self, points, intensities, marker_size=5, alpha=0.5):
        """
        Plot voxel data in 3D using Matplotlib
        
        Args:
            points: Nx3 array of point coordinates
            intensities: N array of point intensities
            marker_size: Size of markers
            alpha: Transparency (0-1)
        """
        if points is None or intensities is None or len(points) == 0:
            logger.warning("No voxel data to visualize")
            return
        
        # Store the data
        self.points = points
        self.intensities = intensities
        
        # Clear the figure
        self.fig.clear()
        
        # Create 3D axis
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        
        # Find the brightest point
        brightest_idx = np.argmax(intensities) if len(intensities) > 0 else None
        
        # Scatter plot with intensity as color
        sc = ax.scatter(
            x_coords, y_coords, z_coords,
            c=intensities, cmap='hot', marker='o',
            s=marker_size, alpha=alpha
        )
        
        # Highlight the brightest point if we have data
        if brightest_idx is not None:
            brightest_x = x_coords[brightest_idx]
            brightest_y = y_coords[brightest_idx]
            brightest_z = z_coords[brightest_idx]
            ax.scatter(
                [brightest_x], [brightest_y], [brightest_z],
                c='blue', marker='*', s=100, label='Brightest Point'
            )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Voxel Visualization')
        
        # Add colorbar
        self.fig.colorbar(sc, ax=ax, label='Intensity')
        
        # Add legend
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Update the canvas
        self.canvas.draw()
    
    @timeit
    def plot_2d_slice(self, slice_axis='z', slice_index=None):
        """
        Plot a 2D slice through the voxel data
        
        Args:
            slice_axis: Axis to slice along ('x', 'y', or 'z')
            slice_index: Position along the axis to slice (None for middle)
        """
        if self.points is None or self.intensities is None or len(self.points) == 0:
            logger.warning("No voxel data to visualize")
            return
        
        # Clear the figure
        self.fig.clear()
        
        # Create axis
        ax = self.fig.add_subplot(111)
        
        # Extract coordinates
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        z_coords = self.points[:, 2]
        
        # Determine the slice index if not provided
        if slice_index is None:
            if slice_axis == 'x':
                slice_index = np.mean([np.min(x_coords), np.max(x_coords)])
            elif slice_axis == 'y':
                slice_index = np.mean([np.min(y_coords), np.max(y_coords)])
            else:  # 'z'
                slice_index = np.mean([np.min(z_coords), np.max(z_coords)])
        
        # Find points near the slice
        slice_thickness = (np.max(self.points) - np.min(self.points)) * 0.01  # 1% thickness
        
        if slice_axis == 'x':
            mask = np.abs(x_coords - slice_index) < slice_thickness
            plot_x, plot_y = y_coords[mask], z_coords[mask]
            plot_intensity = self.intensities[mask]
            xlabel, ylabel = 'Y', 'Z'
        elif slice_axis == 'y':
            mask = np.abs(y_coords - slice_index) < slice_thickness
            plot_x, plot_y = x_coords[mask], z_coords[mask]
            plot_intensity = self.intensities[mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # 'z'
            mask = np.abs(z_coords - slice_index) < slice_thickness
            plot_x, plot_y = x_coords[mask], y_coords[mask]
            plot_intensity = self.intensities[mask]
            xlabel, ylabel = 'X', 'Y'
        
        # Create scatter plot
        sc = ax.scatter(plot_x, plot_y, c=plot_intensity, cmap='hot', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'2D Slice at {slice_axis.upper()} = {slice_index:.2f}')
        
        # Add colorbar
        self.fig.colorbar(sc, ax=ax, label='Intensity')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True)
        
        # Update the canvas
        self.canvas.draw()
    
    def save_figure(self, output_path):
        """
        Save the current figure to a file
        
        Args:
            output_path: Path to save the figure
            
        Returns:
            Success status (boolean)
        """
        if self.fig is not None:
            self.fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {output_path}")
            return True
        return False


class PyVistaVoxelViewer:
    """
    PyVista-based voxel visualization
    
    Provides advanced 3D visualization capabilities
    """
    
    def __init__(self, use_qt=False):
        """
        Initialize the PyVista viewer
        
        Args:
            use_qt: Whether to use Qt integration
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for PyVistaVoxelViewer")
        
        self.points = None
        self.intensities = None
        self.plotter = None
        self.grid = None
        self.mesh = None
        self.volume = None
        
        # Settings
        self.use_qt = use_qt and PYVISTAQT_AVAILABLE
        self.theme = "document"  # "document", "dark", "paraview", etc.
        
        # Default camera position
        self.camera_position = "iso"  # PyVista's built-in positions
        
        # Rendering settings
        self.point_size = 5
        self.opacity = 0.5
        self.colormap = "rainbow"
        
        # Initialize renderer system
        self._initialize_plotter()
    
    def _initialize_plotter(self):
        """Initialize the PyVista plotter with appropriate settings"""
        try:
            # Choose plotter type based on Qt availability
            if self.use_qt:
                logger.debug("Creating PyVista QtInteractor")
                self.plotter = QtInteractor(theme=self.theme)
            else:
                logger.debug("Creating standard PyVista Plotter")
                self.plotter = pv.Plotter()
            
            # Set default theme
            self.plotter.set_background('white')
            
            logger.info("PyVista plotter initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PyVista plotter: {e}")
            raise RenderingError(f"Failed to initialize PyVista plotter: {e}")
    
    def get_widget(self):
        """Get the Qt widget for embedding in the GUI"""
        if self.use_qt and self.plotter:
            return self.plotter
        else:
            logger.warning("Qt integration not available, no widget to return")
            return None
    
    @timeit
    def plot_points(self, points, intensities, reset_camera=True):
        """
        Plot points with intensity values
        
        Args:
            points: Nx3 array of point coordinates
            intensities: N array of point intensities
            reset_camera: Whether to reset the camera view
        """
        if points is None or intensities is None or len(points) == 0:
            logger.warning("No points to visualize")
            return
        
        try:
            # Store the data
            self.points = points
            self.intensities = intensities
            
            # Clear existing points
            self.plotter.clear()
            
            # Create point cloud
            cloud = pv.PolyData(points)
            cloud.point_data["intensity"] = intensities
            
            # Add points to the renderer
            self.plotter.add_points(
                cloud,
                render_points_as_spheres=True,
                point_size=self.point_size,
                opacity=self.opacity,
                scalar_bar_args={'title': 'Intensity'},
                cmap=self.colormap,
                show_scalar_bar=True,
                reset_camera=reset_camera
            )
            
            # Add brightest point as a marker
            brightest_idx = np.argmax(intensities)
            brightest_point = points[brightest_idx]
            
            # Add a marker for the brightest point
            brightest_mesh = pv.Sphere(radius=0.05, center=brightest_point)
            self.plotter.add_mesh(
                brightest_mesh, 
                color='blue', 
                render_points_as_spheres=True,
                point_size=10
            )
            
            # Update the view
            if reset_camera:
                self.plotter.reset_camera()
                self.plotter.view_isometric()
            
            # Add axes for reference
            self.plotter.add_axes(interactive=True)
            
            # Render
            self.plotter.update()
            
            logger.info("Points visualization updated")
            
        except Exception as e:
            logger.error(f"Failed to plot points: {e}")
            raise RenderingError(f"Failed to plot points: {e}")
    
    @timeit
    def plot_volume(self, voxel_grid, voxel_grid_extent, reset_camera=True):
        """
        Render a volume directly from a voxel grid
        
        Args:
            voxel_grid: 3D numpy array of voxel values
            voxel_grid_extent: Extents of the voxel grid in 3D space
            reset_camera: Whether to reset the camera view
        """
        if voxel_grid is None or np.all(voxel_grid == 0):
            logger.warning("No voxel data to visualize")
            return
        
        try:
            # Clear existing data
            self.plotter.clear()
            
            # Create a uniform grid for the voxels
            grid = pv.UniformGrid()
            
            # Set dimensions (add 1 because UniformGrid dimensions are number of points, not cells)
            grid.dimensions = np.array(voxel_grid.shape) + 1
            
            # Set the grid origin and spacing
            x_min, x_max = voxel_grid_extent[0]
            y_min, y_max = voxel_grid_extent[1]
            z_min, z_max = voxel_grid_extent[2]
            
            grid.origin = [x_min, y_min, z_min]
            grid.spacing = [(x_max - x_min) / voxel_grid.shape[0],
                          (y_max - y_min) / voxel_grid.shape[1],
                          (z_max - z_min) / voxel_grid.shape[2]]
            
            # Add the data to the grid
            # Note: PyVista's UniformGrid expects cell data with shape matching dimensions-1
            grid.cell_data["values"] = voxel_grid.flatten(order="F")
            
            # Store the grid
            self.grid = grid
            
            # Add volume rendering
            self.plotter.add_volume(
                grid, 
                cmap=self.colormap,
                opacity='sigmoid',
                show_scalar_bar=True,
                reset_camera=reset_camera
            )
            
            # Update the view
            if reset_camera:
                self.plotter.reset_camera()
                self.plotter.view_isometric()
            
            # Add axes for reference
            self.plotter.add_axes(interactive=True)
            
            # Render
            self.plotter.update()
            
            logger.info("Volume visualization updated")
            
        except Exception as e:
            logger.error(f"Failed to plot volume: {e}")
            raise RenderingError(f"Failed to plot volume: {e}")
    
    @timeit
    def plot_mesh(self, mesh, reset_camera=True):
        """
        Plot a surface mesh
        
        Args:
            mesh: PyVista mesh object or (vertices, faces) tuple
            reset_camera: Whether to reset the camera view
        """
        try:
            # Clear existing data
            self.plotter.clear()
            
            # Convert mesh if needed
            if isinstance(mesh, tuple) and len(mesh) == 2:
                # Convert (vertices, faces) to PyVista mesh
                vertices, faces = mesh
                
                # Create mesh with triangular faces
                faces_with_counts = np.column_stack((
                    np.full(len(faces), 3),  # 3 vertices per face
                    faces
                )).flatten()
                
                pv_mesh = pv.PolyData(vertices, faces_with_counts)
                self.mesh = pv_mesh
            else:
                # Assume it's already a PyVista mesh
                pv_mesh = mesh
                self.mesh = mesh
            
            # Add mesh to plotter
            self.plotter.add_mesh(
                pv_mesh,
                show_edges=True,
                color='tan',
                opacity=0.8,
                reset_camera=reset_camera
            )
            
            # Update the view
            if reset_camera:
                self.plotter.reset_camera()
                self.plotter.view_isometric()
            
            # Add axes for reference
            self.plotter.add_axes(interactive=True)
            
            # Render
            self.plotter.update()
            
            logger.info("Mesh visualization updated")
            
        except Exception as e:
            logger.error(f"Failed to plot mesh: {e}")
            raise RenderingError(f"Failed to plot mesh: {e}")
    
    @timeit
    def add_clip_plane_widget(self, callback=None):
        """
        Add an interactive clipping plane widget
        
        Args:
            callback: Function to call when the plane is moved (optional)
        """
        if self.grid is None and self.mesh is None:
            logger.warning("No data to clip")
            return
        
        try:
            # Determine which data to clip
            if self.mesh is not None:
                # Clip mesh
                self.plotter.add_mesh_clip_plane(self.mesh, show_edges=True, interaction_event=callback)
            elif self.grid is not None:
                # Clip volume
                self.plotter.add_volume_clip_plane(self.grid, cmap=self.colormap, interaction_event=callback)
            
            # Render
            self.plotter.update()
            
            logger.info("Clip plane widget added")
            
        except Exception as e:
            logger.error(f"Failed to add clip plane widget: {e}")
            raise RenderingError(f"Failed to add clip plane widget: {e}")
    
    @timeit
    def add_slice_widget(self, callback=None):
        """
        Add an interactive slice widget
        
        Args:
            callback: Function to call when the slice is moved (optional)
        """
        if self.grid is None:
            logger.warning("No volume data to slice")
            return
        
        try:
            # Add a slice widget
            self.plotter.add_mesh_slice(
                self.grid,
                cmap=self.colormap,
                interaction_event=callback
            )
            
            # Render
            self.plotter.update()
            
            logger.info("Slice widget added")
            
        except Exception as e:
            logger.error(f"Failed to add slice widget: {e}")
            raise RenderingError(f"Failed to add slice widget: {e}")
    
    @timeit
    def add_measurement_widget(self, callback=None):
        """
        Add a measurement widget for distance calculation
        
        Args:
            callback: Function to call when measurement changes (optional)
        """
        try:
            self.plotter.add_measurement_widget(callback)
            
            # Render
            self.plotter.update()
            
            logger.info("Measurement widget added")
            
        except Exception as e:
            logger.error(f"Failed to add measurement widget: {e}")
            raise RenderingError(f"Failed to add measurement widget: {e}")
    
    @timeit
    def add_axes_widget(self):
        """Add an interactive axes widget"""
        try:
            self.plotter.add_axes_at_origin(interactive=True)
            
            # Render
            self.plotter.update()
            
            logger.info("Axes widget added")
            
        except Exception as e:
            logger.error(f"Failed to add axes widget: {e}")
            raise RenderingError(f"Failed to add axes widget: {e}")
    
    def screenshot(self, filename=None):
        """
        Take a screenshot of the current view
        
        Args:
            filename: Path to save the screenshot, or None for memory buffer
            
        Returns:
            Image data or success status
        """
        try:
            if filename:
                self.plotter.screenshot(filename)
                logger.info(f"Screenshot saved to {filename}")
                return True
            else:
                img = self.plotter.screenshot(return_img=True)
                return img
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise RenderingError(f"Failed to take screenshot: {e}")
    
    def close(self):
        """Close the plotter and release resources"""
        if self.plotter:
            self.plotter.close()
            logger.info("PyVista plotter closed")