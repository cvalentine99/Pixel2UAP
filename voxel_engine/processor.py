"""
Core voxel processing logic
"""

import os
import numpy as np
import logging
from pathlib import Path
import subprocess
import sys
import scipy.ndimage as ndimage
from ..utils.profiling import timeit, log_memory_usage, optimize_voxel_grid
from ..utils.errors import ProcessingError, InputError

logger = logging.getLogger(__name__)

try:
    import process_image_cpp
    PROCESS_IMAGE_CPP_AVAILABLE = True
    logger.info("C++ extensions successfully loaded")
except ImportError:
    logger.warning("C++ extensions not found. Using Python fallback implementations.")
    PROCESS_IMAGE_CPP_AVAILABLE = False

# Check for PyVista availability
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    logger.info("PyVista is available for mesh/volume rendering")
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista is not installed. 3D rendering features will be limited")


class VoxelProcessor:
    """
    Handle the processing of various input types to voxel representations
    """
    
    def __init__(self):
        """Initialize the voxel processor with default parameters"""
        # Default processing parameters
        self.voxel_grid_size = (256, 256, 256)  # Number of voxels in (x, y, z) directions
        self.grid_extent = 1.0  # Half the length of one side of the voxel cube
        self.distance_from_origin = 1.0  # Distance from origin to grid center
        self.center_orientation = (0.0, 0.0, 1.0)  # Default orientation vector
        self.brightness_threshold_percentile = 0.1
        self.max_distance = 10.0  # Maximum distance for ray casting
        self.num_steps = 10000  # Number of steps along each ray
        
        # Rendering parameters
        self.angular_width = 5.0  # Angular width of viewing patch in degrees
        self.angular_height = 5.0  # Angular height of viewing patch in degrees
        self.texture_width = 1024  # Texture width (pixels)
        self.texture_height = 1024  # Texture height (pixels)
        
        # Post-processing parameters
        self.smoothing_sigma = 0.5  # Gaussian smoothing sigma
        self.apply_smoothing = False
        self.apply_thresholding = False
        self.threshold_value = 0.5
        
        # Performance parameters
        self.use_gpu = False
        self.use_sparse = False
        
        # Storage for processed data
        self.reset_voxel_grid()
    
    def reset_voxel_grid(self):
        """Reset the voxel grid and textures to zeros"""
        logger.debug("Resetting voxel grid")
        
        # Create and store empty voxel grid
        self.voxel_grid = np.zeros(self.voxel_grid_size, dtype=np.float32)
        
        # Create empty celestial texture
        self.celestial_sphere_texture = np.zeros(
            (self.texture_height, self.texture_width), dtype=np.float32)
        
        # Calculate grid extents
        self.voxel_grid_extent = self._compute_voxel_grid_extent()
        
        # Reset processed files list
        self.processed_files = []
        
        # Reset output mesh
        self.output_mesh = None
    
    def _compute_voxel_grid_extent(self):
        """
        Compute the voxel grid spatial extents based on center coordinates
        """
        # Convert orientation vector to a direction
        direction_vector = np.array(self.center_orientation)
        
        # Normalize direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # Compute the center of the voxel grid in space
        voxel_grid_center = direction_vector * self.distance_from_origin
        
        # Define the extents of the voxel grid
        return (
            (voxel_grid_center[0] - self.grid_extent, voxel_grid_center[0] + self.grid_extent),
            (voxel_grid_center[1] - self.grid_extent, voxel_grid_center[1] + self.grid_extent),
            (voxel_grid_center[2] - self.grid_extent, voxel_grid_center[2] + self.grid_extent)
        )
    
    @timeit
    def process_input(self, input_path, method="projection"):
        """
        Process an input file or directory using the specified method
        
        Args:
            input_path: Path to input file or directory
            method: Processing method ('projection', 'space_carving', etc.)
            
        Returns:
            Success status
        """
        logger.info(f"Processing input: {input_path} using method: {method}")
        
        input_path = Path(input_path)
        if not input_path.exists():
            msg = f"Input path does not exist: {input_path}"
            logger.error(msg)
            raise InputError(msg)
        
        # Determine input type and call appropriate method
        if input_path.is_file():
            return self._process_file(input_path, method)
        elif input_path.is_dir():
            return self._process_directory(input_path, method)
        else:
            msg = f"Unsupported input type: {input_path}"
            logger.error(msg)
            raise InputError(msg)
    
    def _process_file(self, file_path, method):
        """Process a single input file"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Dispatch to appropriate method based on file extension
        if extension in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            return self.process_image_file(file_path, method)
        elif extension in ('.mp4', '.avi', '.mov', '.mkv'):
            return self.process_video_file(file_path, method)
        elif extension == '.dcm':
            return self.process_dicom_file(file_path, method)
        elif extension in ('.nii', '.nii.gz'):
            return self.process_nifti_file(file_path, method)
        elif extension in ('.ply', '.obj', '.stl', '.pcd'):
            return self.process_mesh_or_point_cloud(file_path, method)
        elif extension == '.json':
            return self.process_motion_data(file_path, method)
        else:
            msg = f"Unsupported file type: {extension}"
            logger.error(msg)
            raise InputError(msg)
    
    def _process_directory(self, dir_path, method):
        """Process a directory of files"""
        dir_path = Path(dir_path)
        
        # Check if this is a DICOM directory
        dicom_files = list(dir_path.glob('*.dcm'))
        if dicom_files:
            return self.process_dicom_series(dir_path, method)
        
        # Check for motion data
        json_files = list(dir_path.glob('*.json'))
        if json_files:
            # Look for metadata.json or similar
            metadata_file = next((f for f in json_files if "meta" in f.name.lower()), None)
            if metadata_file:
                return self.process_motion_data(metadata_file, method, image_dir=dir_path)
        
        # Otherwise, process as a batch of individual images
        image_files = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg')) + \
                     list(dir_path.glob('*.jpeg')) + list(dir_path.glob('*.tif')) + \
                     list(dir_path.glob('*.tiff'))
        
        if not image_files:
            msg = f"No supported files found in directory: {dir_path}"
            logger.warning(msg)
            raise InputError(msg)
        
        success_count = 0
        for img_file in image_files:
            try:
                if self.process_image_file(img_file, method):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        
        return success_count > 0
    
    @timeit
    def process_image_file(self, image_path, method="projection"):
        """
        Process a single image file and update the voxel grid
        
        Args:
            image_path: Path to the image file
            method: Processing method ('projection', 'space_carving', etc.)
            
        Returns:
            Success status (boolean)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Load and preprocess the image
        try:
            from PIL import Image
            image = self._load_image(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise InputError(f"Failed to load image: {e}", image_path=str(image_path))
        
        if image is None:
            return False
        
        # Mock earth position and pointing direction for demonstration
        # In a real application, this would come from the FITS header or user input
        earth_position = [0.0, 0.0, 0.0]  # Origin
        pointing_direction = [0.0, 0.0, 1.0]  # Looking along z-axis
        
        # Default field of view (in radians)
        fov_rad = np.deg2rad(1.0)  # 1 degree field of view
        
        # Get image dimensions
        height, width = image.shape
        
        # Prepare parameters for C++ processing or Python fallback
        voxel_grid_extent_list = [
            (self.voxel_grid_extent[0][0], self.voxel_grid_extent[0][1]),
            (self.voxel_grid_extent[1][0], self.voxel_grid_extent[1][1]),
            (self.voxel_grid_extent[2][0], self.voxel_grid_extent[2][1])
        ]
        
        # Define sky patch in radians (for celestial texture)
        c_ra_rad = np.deg2rad(0.0)  # Center RA (arbitrary for single image)
        c_dec_rad = np.deg2rad(0.0)  # Center Dec (arbitrary for single image)
        aw_rad = np.deg2rad(self.angular_width)
        ah_rad = np.deg2rad(self.angular_height)
        
        try:
            # Call C++ function to process the image if available
            if PROCESS_IMAGE_CPP_AVAILABLE and method == "projection":
                logger.debug("Using C++ implementation for projection")
                process_image_cpp.process_image_cpp(
                    image.astype(np.float64),
                    earth_position,
                    pointing_direction,
                    fov_rad,
                    width,
                    height,
                    self.voxel_grid,
                    voxel_grid_extent_list,
                    self.max_distance,
                    self.num_steps,
                    self.celestial_sphere_texture,
                    c_ra_rad,
                    c_dec_rad,
                    aw_rad,
                    ah_rad,
                    True,   # update_celestial_sphere
                    False   # perform_background_subtraction
                )
            else:
                # Python fallback implementation
                logger.debug(f"Using Python implementation for {method}")
                
                if method == "projection":
                    self._process_image_projection_python(
                        image, earth_position, pointing_direction, fov_rad)
                elif method == "space_carving":
                    self._process_image_space_carving_python(
                        image, earth_position, pointing_direction, fov_rad)
                else:
                    msg = f"Unsupported method: {method}"
                    logger.error(msg)
                    raise ProcessingError(msg)
        
        except Exception as e:
            logger.error(f"Error during image processing: {e}")
            raise ProcessingError(f"Error during image processing: {e}")
        
        # Add to processed files
        self.processed_files.append(str(image_path))
        
        # Apply post-processing if enabled
        self._apply_post_processing()
        
        return True
    
    def _process_image_projection_python(self, image, position, direction, fov):
        """
        Python implementation of the image projection algorithm
        
        This is a fallback if the C++ extension is not available.
        """
        logger.info("Using Python implementation for image projection")
        
        height, width = image.shape
        focal_length = (width / 2.0) / np.tan(fov / 2.0)
        
        # Calculate principal point (image center)
        cx = width / 2.0
        cy = height / 2.0
        
        # Normalize direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        # Define coordinate axes
        z_axis = direction
        up = np.array([0.0, 0.0, 1.0])
        
        # Check if direction is parallel to up vector
        if np.abs(np.dot(z_axis, up)) > 0.99:
            # Use a different up vector
            up = np.array([0.0, 1.0, 0.0])
        
        # Compute camera coordinate system
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Extract grid parameters
        nx, ny, nz = self.voxel_grid.shape
        x_min, x_max = self.voxel_grid_extent[0]
        y_min, y_max = self.voxel_grid_extent[1]
        z_min, z_max = self.voxel_grid_extent[2]
        
        # Initialize ray origin (camera position)
        ray_origin = np.array(position, dtype=np.float64)
        
        # Create AABB for voxel grid
        box_min = np.array([x_min, y_min, z_min])
        box_max = np.array([x_max, y_max, z_max])
        
        # Simple progress logging
        log_interval = max(1, height // 10)
        
        # For each pixel in the image
        for i in range(height):
            if i % log_interval == 0:
                logger.debug(f"Processing row {i}/{height}")
                
            for j in range(width):
                brightness = image[i, j]
                
                if brightness > 0:
                    # Compute ray direction in camera coordinates
                    x_cam = (j - cx)
                    y_cam = (i - cy)
                    z_cam = focal_length
                    
                    norm = np.sqrt(x_cam*x_cam + y_cam*y_cam + z_cam*z_cam)
                    dir_camera = np.array([x_cam/norm, y_cam/norm, z_cam/norm])
                    
                    # Transform to world coordinates
                    ray_direction = (x_axis * dir_camera[0] + 
                                    y_axis * dir_camera[1] + 
                                    z_axis * dir_camera[2])
                    
                    # Ray-AABB intersection
                    t_entry, t_exit = self._ray_aabb_intersection(
                        ray_origin, ray_direction, box_min, box_max)
                    
                    if t_entry <= t_exit:
                        # Ray intersects the box
                        t_entry = max(t_entry, 0.0)
                        t_exit = min(t_exit, self.max_distance)
                        
                        # Cast ray through voxel grid
                        step_size = (t_exit - t_entry) / self.num_steps
                        
                        for s in range(self.num_steps):
                            t = t_entry + s * step_size
                            point = ray_origin + t * ray_direction
                            
                            # Convert point to voxel indices
                            x_norm = (point[0] - x_min) / (x_max - x_min)
                            y_norm = (point[1] - y_min) / (y_max - y_min)
                            z_norm = (point[2] - z_min) / (z_max - z_min)
                            
                            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1 and 0 <= z_norm <= 1:
                                x_idx = int(x_norm * nx)
                                y_idx = int(y_norm * ny)
                                z_idx = int(z_norm * nz)
                                
                                # Clamp indices to grid boundaries
                                x_idx = min(max(x_idx, 0), nx - 1)
                                y_idx = min(max(y_idx, 0), ny - 1)
                                z_idx = min(max(z_idx, 0), nz - 1)
                                
                                # Update voxel grid
                                self.voxel_grid[x_idx, y_idx, z_idx] += brightness
        
        logger.info("Python projection completed")

    def _process_image_space_carving_python(self, image, position, direction, fov):
        """
        Python implementation of the Space Carving algorithm
        
        This requires multiple images with known positions to be useful.
        For a single image, it simply marks voxels as occupied along the ray.
        """
        logger.info("Using Python implementation for space carving")
        
        # For demonstration - this would be more useful with multiple images
        height, width = image.shape
        focal_length = (width / 2.0) / np.tan(fov / 2.0)
        
        # Calculate principal point (image center)
        cx = width / 2.0
        cy = height / 2.0
        
        # Create a binary segmentation mask (simple thresholding for demonstration)
        threshold = np.mean(image)
        mask = image > threshold
        
        logger.debug(f"Created segmentation mask with {np.sum(mask)} foreground pixels")
        
        # For a real space carving algorithm, we would:
        # 1. Project each voxel onto the image
        # 2. Check if it falls within the segmentation mask
        # 3. If not, carve it away (set to 0)
        
        # Since this is just a demonstration, we'll do a simplified version
        
        # Initialize all voxels as potentially occupied
        if len(self.processed_files) == 0:
            # First image - initialize all voxels as potentially occupied
            self.voxel_grid.fill(1.0)
        
        # Normalize direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        # Define camera coordinate system
        z_axis = direction
        up = np.array([0.0, 0.0, 1.0])
        
        # Check if direction is parallel to up vector
        if np.abs(np.dot(z_axis, up)) > 0.99:
            # Use a different up vector
            up = np.array([0.0, 1.0, 0.0])
        
        # Compute camera coordinate system
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # Extract grid parameters
        nx, ny, nz = self.voxel_grid.shape
        x_min, x_max = self.voxel_grid_extent[0]
        y_min, y_max = self.voxel_grid_extent[1]
        z_min, z_max = self.voxel_grid_extent[2]
        
        # For each voxel
        for xi in range(nx):
            if xi % 10 == 0:
                logger.debug(f"Processing voxel layer {xi}/{nx}")
                
            for yi in range(ny):
                for zi in range(nz):
                    # Skip already carved voxels
                    if self.voxel_grid[xi, yi, zi] == 0:
                        continue
                    
                    # Convert voxel indices to world coordinates
                    x = x_min + (x_max - x_min) * (xi + 0.5) / nx
                    y = y_min + (y_max - y_min) * (yi + 0.5) / ny
                    z = z_min + (z_max - z_min) * (zi + 0.5) / nz
                    
                    # Voxel center in world coordinates
                    voxel_center = np.array([x, y, z])
                    
                    # Vector from camera to voxel
                    cam_to_voxel = voxel_center - np.array(position)
                    
                    # Project voxel onto image plane
                    # First, get coordinates in camera space
                    x_cam = np.dot(cam_to_voxel, x_axis)
                    y_cam = np.dot(cam_to_voxel, y_axis)
                    z_cam = np.dot(cam_to_voxel, z_axis)
                    
                    # Check if voxel is behind camera
                    if z_cam <= 0:
                        continue
                    
                    # Project to pixel coordinates
                    j = int(cx + (x_cam / z_cam) * focal_length)
                    i = int(cy + (y_cam / z_cam) * focal_length)
                    
                    # Check if projection is within image bounds
                    if 0 <= i < height and 0 <= j < width:
                        # Check if projection is outside the silhouette
                        if not mask[i, j]:
                            # Carve the voxel (set to 0)
                            self.voxel_grid[xi, yi, zi] = 0
        
        logger.info("Space carving completed")
    
    def _ray_aabb_intersection(self, origin, direction, box_min, box_max):
        """
        Calculate the intersection of a ray with an axis-aligned bounding box.
        
        Args:
            origin: Ray origin (3D point)
            direction: Ray direction (3D vector)
            box_min: Minimum point of the box (3D point)
            box_max: Maximum point of the box (3D point)
            
        Returns:
            (t_entry, t_exit) - entry and exit distances, or (inf, -inf) if no intersection
        """
        t_min = -np.inf
        t_max = np.inf
        
        for i in range(3):
            if abs(direction[i]) > 1e-8:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
            elif origin[i] < box_min[i] or origin[i] > box_max[i]:
                # Ray is parallel to this slab and outside the box
                return np.inf, -np.inf
        
        if t_min > t_max:
            return np.inf, -np.inf
            
        return t_min, t_max
    
    def _apply_post_processing(self):
        """Apply post-processing to the voxel grid based on current settings"""
        if self.apply_smoothing:
            logger.debug(f"Applying Gaussian smoothing with sigma={self.smoothing_sigma}")
            self.voxel_grid = ndimage.gaussian_filter(
                self.voxel_grid, sigma=self.smoothing_sigma)
        
        if self.apply_thresholding:
            logger.debug(f"Applying thresholding with value={self.threshold_value}")
            self.voxel_grid = (self.voxel_grid > self.threshold_value).astype(np.float32)
    
    def _load_image(self, image_path):
        """
        Load and preprocess an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized image as a 2D numpy array
        """
        try:
            from PIL import Image
            import numpy as np
            
            with Image.open(image_path) as img:
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array
                image = np.array(img, dtype=np.float32)
                
                # Normalize to [0, 1] range
                image_min = np.min(image)
                image_max = np.max(image)
                if image_max - image_min == 0:
                    logger.warning(f"Image {image_path} has zero dynamic range")
                    return None
                
                image = (image - image_min) / (image_max - image_min)
                return image
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise InputError(f"Failed to load image: {e}")
    
    @timeit
    def process_video_file(self, video_path, method="projection"):
        """
        Process a video file by extracting frames and processing each one
        
        Args:
            video_path: Path to the video file
            method: Processing method
            
        Returns:
            Success status (boolean)
        """
        try:
            import cv2
            
            logger.info(f"Processing video: {video_path}")
            
            # Open the video file
            video = cv2.VideoCapture(str(video_path))
            
            if not video.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            # Get video properties
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video has {frame_count} frames at {fps} FPS")
            
            # Determine frame extraction rate (extract 1 frame per second)
            frame_interval = max(1, int(fps))
            
            success_count = 0
            frame_idx = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                # Process every n-th frame
                if frame_idx % frame_interval == 0:
                    # Convert BGR to grayscale
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Convert to numpy array
                    image = np.array(gray_frame, dtype=np.float32)
                    
                    # Normalize to [0, 1] range
                    image_min = np.min(image)
                    image_max = np.max(image)
                    if image_max - image_min == 0:
                        logger.warning(f"Frame {frame_idx} has zero dynamic range")
                        continue
                    
                    image = (image - image_min) / (image_max - image_min)
                    
                    # Create a mock camera position based on frame index
                    # In a real scenario, this would come from camera motion estimation
                    position = [0.0, 0.0, 0.0]
                    direction = [0.0, 0.0, 1.0]
                    
                    # Process the frame
                    fov_rad = np.deg2rad(60.0)  # Typical video camera FOV
                    
                    if method == "projection":
                        self._process_image_projection_python(
                            image, position, direction, fov_rad)
                    elif method == "space_carving":
                        self._process_image_space_carving_python(
                            image, position, direction, fov_rad)
                    
                    success_count += 1
                    
                    logger.debug(f"Processed frame {frame_idx}/{frame_count}")
                
                frame_idx += 1
            
            # Release video resources
            video.release()
            
            # Add to processed files
            self.processed_files.append(str(video_path))
            
            # Apply post-processing
            self._apply_post_processing()
            
            logger.info(f"Processed {success_count} frames from video")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise ProcessingError(f"Error processing video: {e}")
        
    @timeit
    def process_dicom_file(self, dicom_path, method="projection"):
        """
        Process a single DICOM file
        
        Args:
            dicom_path: Path to the DICOM file
            method: Processing method
            
        Returns:
            Success status (boolean)
        """
        try:
            import pydicom
            
            logger.info(f"Processing DICOM file: {dicom_path}")
            
            # Load DICOM data
            ds = pydicom.dcmread(dicom_path)
            
            # Get pixel data
            pixel_data = ds.pixel_array
            
            # Apply rescale slope and intercept if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_data = pixel_data * ds.RescaleSlope + ds.RescaleIntercept
            
            # Normalize to [0, 1] range
            pixel_min = np.min(pixel_data)
            pixel_max = np.max(pixel_data)
            if pixel_max - pixel_min == 0:
                logger.warning(f"DICOM {dicom_path} has zero dynamic range")
                return False
            
            image = (pixel_data - pixel_min) / (pixel_max - pixel_min)
            
            # Process as a regular image
            position = [0.0, 0.0, 0.0]
            direction = [0.0, 0.0, 1.0]
            fov_rad = np.deg2rad(1.0)
            
            if method == "projection":
                self._process_image_projection_python(
                    image, position, direction, fov_rad)
            elif method == "space_carving":
                self._process_image_space_carving_python(
                    image, position, direction, fov_rad)
            
            # Add to processed files
            self.processed_files.append(str(dicom_path))
            
            # Apply post-processing
            self._apply_post_processing()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing DICOM {dicom_path}: {e}")
            raise ProcessingError(f"Error processing DICOM: {e}")
    
    @timeit
    def process_dicom_series(self, dicom_dir, method="direct_import"):
        """
        Process a directory of DICOM files as a 3D volume
        
        Args:
            dicom_dir: Directory containing DICOM files
            method: Processing method ('direct_import' for direct volume import)
            
        Returns:
            Success status (boolean)
        """
        try:
            import pydicom
            
            logger.info(f"Processing DICOM series from: {dicom_dir}")
            
            # Find all DICOM files in the directory
            dicom_files = list(Path(dicom_dir).glob('*.dcm'))
            
            if not dicom_files:
                logger.warning(f"No DICOM files found in: {dicom_dir}")
                return False
            
            logger.info(f"Found {len(dicom_files)} DICOM files")
            
            # Read all files and extract position information
            slices = []
            for dcm_file in dicom_files:
                try:
                    ds = pydicom.dcmread(dcm_file)
                    
                    # Check if this is a suitable imaging dataset
                    if not hasattr(ds, 'ImagePositionPatient') or not hasattr(ds, 'ImageOrientationPatient'):
                        logger.warning(f"DICOM file missing required tags: {dcm_file}")
                        continue
                    
                    # All checks passed, add to slices
                    slices.append(ds)
                except Exception as e:
                    logger.warning(f"Error reading DICOM file {dcm_file}: {e}")
            
            if not slices:
                logger.error("No valid DICOM slices found")
                return False
            
            # Sort slices by position
            sorted_slices = self._sort_dicom_slices(slices)
            
            # Extract pixel data and stack into a 3D volume
            pixel_arrays = []
            for ds in sorted_slices:
                # Apply rescaling if available
                pixel_array = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                
                pixel_arrays.append(pixel_array)
            
            # Stack arrays to create 3D volume
            volume = np.stack(pixel_arrays)
            
            # Normalize to [0, 1] range
            volume_min = np.min(volume)
            volume_max = np.max(volume)
            if volume_max - volume_min == 0:
                logger.warning("DICOM volume has zero dynamic range")
                return False
            
            volume = (volume - volume_min) / (volume_max - volume_min)
            
            # For direct import method, reshape the volume to fit our voxel grid
            if method == "direct_import":
                # Resize volume to match voxel grid dimensions
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                src_shape = np.array(volume.shape)
                dst_shape = np.array(self.voxel_grid_size)
                factors = dst_shape / src_shape
                
                # Resize the volume
                logger.info(f"Resizing DICOM volume from {src_shape} to {dst_shape}")
                resized_volume = zoom(volume, factors)
                
                # Replace our voxel grid with the resized volume
                self.voxel_grid = resized_volume
                
                logger.info("DICOM volume directly imported as voxel grid")
            else:
                # Other methods would process the volume differently
                logger.warning(f"Method {method} not implemented for DICOM series")
                return False
            
            # Add to processed files
            self.processed_files.append(str(dicom_dir))
            
            # Apply post-processing
            self._apply_post_processing()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing DICOM series {dicom_dir}: {e}")
            raise ProcessingError(f"Error processing DICOM series: {e}")
    
    def _sort_dicom_slices(self, slices):
        """Sort DICOM slices by spatial position"""
        # Get image orientation (direction cosines)
        orientation = slices[0].ImageOrientationPatient
        
        # Calculate slice normal vector (cross product of orientation vectors)
        row_vec = np.array(orientation[:3])
        col_vec = np.array(orientation[3:])
        slice_normal = np.cross(row_vec, col_vec)
        
        # Calculate slice position along normal
        def slice_position(s):
            pos = np.array(s.ImagePositionPatient)
            return np.dot(pos, slice_normal)
        
        # Sort slices by position
        return sorted(slices, key=slice_position)
    
    @timeit
    def process_nifti_file(self, nifti_path, method="direct_import"):
        """
        Process a NIFTI file as a 3D volume
        
        Args:
            nifti_path: Path to the NIFTI file
            method: Processing method ('direct_import' for direct volume import)
            
        Returns:
            Success status (boolean)
        """
        try:
            import nibabel as nib
            
            logger.info(f"Processing NIFTI file: {nifti_path}")
            
            # Load NIFTI data
            nifti_img = nib.load(nifti_path)
            
            # Get the data as a numpy array
            volume = nifti_img.get_fdata()
            
            # Get orientation and voxel spacing information
            affine = nifti_img.affine
            
            logger.debug(f"NIFTI volume shape: {volume.shape}")
            logger.debug(f"NIFTI affine matrix: {affine}")
            
            # Normalize to [0, 1] range
            volume_min = np.min(volume)
            volume_max = np.max(volume)
            if volume_max - volume_min == 0:
                logger.warning(f"NIFTI {nifti_path} has zero dynamic range")
                return False
            
            volume = (volume - volume_min) / (volume_max - volume_min)
            
            # For direct import method, reshape the volume to fit our voxel grid
            if method == "direct_import":
                # Resize volume to match voxel grid dimensions
                from scipy.ndimage import zoom
                
                # Calculate zoom factors
                src_shape = np.array(volume.shape)
                dst_shape = np.array(self.voxel_grid_size)
                factors = dst_shape / src_shape
                
                # Resize the volume
                logger.info(f"Resizing NIFTI volume from {src_shape} to {dst_shape}")
                resized_volume = zoom(volume, factors)
                
                # Replace our voxel grid with the resized volume
                self.voxel_grid = resized_volume
                
                logger.info("NIFTI volume directly imported as voxel grid")
            else:
                # Other methods would process the volume differently
                logger.warning(f"Method {method} not implemented for NIFTI")
                return False
            
            # Add to processed files
            self.processed_files.append(str(nifti_path))
            
            # Apply post-processing
            self._apply_post_processing()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing NIFTI {nifti_path}: {e}")
            raise ProcessingError(f"Error processing NIFTI: {e}")
    
    @timeit
    def process_mesh_or_point_cloud(self, mesh_path, method="voxelization"):
        """
        Process a 3D mesh or point cloud file
        
        Args:
            mesh_path: Path to the mesh or point cloud file
            method: Processing method ('voxelization' for converting to voxels)
            
        Returns:
            Success status (boolean)
        """
        try:
            logger.info(f"Processing mesh/point cloud: {mesh_path}")
            
            # Try to use PyVista if available
            if PYVISTA_AVAILABLE:
                import pyvista as pv
                
                # Load the mesh
                mesh = pv.read(mesh_path)
                
                logger.debug(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
                
                if method == "voxelization":
                    # Voxelize the mesh
                    logger.info(f"Voxelizing mesh to {self.voxel_grid_size} grid")
                    
                    # Get bounds
                    bounds = mesh.bounds
                    x_min, x_max, y_min, y_max, z_min, z_max = bounds
                    
                    # Create uniform grid for voxelization
                    grid = pv.UniformGrid(
                        dimensions=self.voxel_grid_size,
                        spacing=((x_max-x_min)/self.voxel_grid_size[0],
                                (y_max-y_min)/self.voxel_grid_size[1],
                                (z_max-z_min)/self.voxel_grid_size[2]),
                        origin=(x_min, y_min, z_min)
                    )
                    
                    # Generate voxels
                    voxels = grid.sample_points_to_volume(mesh.points, radius=1.0)
                    
                    # Get the voxel data
                    volume_data = voxels.point_data["values"]
                    
                    # Reshape to match our grid size
                    volume = volume_data.reshape(self.voxel_grid_size, order='F')
                    
                    # Normalize
                    volume_min = np.min(volume)
                    volume_max = np.max(volume)
                    if volume_max > volume_min:
                        volume = (volume - volume_min) / (volume_max - volume_min)
                    
                    # Replace our voxel grid
                    self.voxel_grid = volume
                    
                    logger.info("Mesh successfully voxelized")
                else:
                    logger.warning(f"Method {method} not implemented for mesh/point cloud")
                    return False
            else:
                logger.error("PyVista is required for mesh/point cloud processing but not available")
                return False
            
            # Add to processed files
            self.processed_files.append(str(mesh_path))
            
            # Apply post-processing
            self._apply_post_processing()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing mesh/point cloud {mesh_path}: {e}")
            raise ProcessingError(f"Error processing mesh/point cloud: {e}")
    
    @timeit
    def process_motion_data(self, json_path, method="projection", image_dir=None):
        """
        Process motion data from a JSON file with image references
        
        Args:
            json_path: Path to the JSON metadata file
            method: Processing method
            image_dir: Directory containing image files (if different from JSON location)
            
        Returns:
            Success status (boolean)
        """
        try:
            import json
            
            logger.info(f"Processing motion data: {json_path}")
            
            # Load JSON metadata
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            if not isinstance(metadata, list):
                logger.error(f"Invalid motion data format in {json_path}")
                return False
            
            # Determine image directory
            if image_dir is None:
                image_dir = Path(json_path).parent
            else:
                image_dir = Path(image_dir)
            
            # Process each frame in metadata
            success_count = 0
            for frame in metadata:
                try:
                    # Extract frame data
                    image_file = frame.get('image_file')
                    if not image_file:
                        logger.warning("Frame missing image_file field, skipping")
                        continue
                    
                    # Construct full image path
                    image_path = image_dir / image_file
                    if not image_path.exists():
                        logger.warning(f"Image file not found: {image_path}")
                        continue
                    
                    # Extract camera parameters
                    camera_position = frame.get('camera_position', [0.0, 0.0, 0.0])
                    
                    # Extract orientation (could be Euler angles or quaternion)
                    yaw = frame.get('yaw', 0.0)
                    pitch = frame.get('pitch', 0.0)
                    roll = frame.get('roll', 0.0)
                    
                    # Convert Euler angles to direction vector (simplistic)
                    import math
                    yaw_rad = math.radians(yaw)
                    pitch_rad = math.radians(pitch)
                    
                    direction = [
                        math.sin(yaw_rad) * math.cos(pitch_rad),
                        math.sin(pitch_rad),
                        math.cos(yaw_rad) * math.cos(pitch_rad)
                    ]
                    
                    # Get field of view
                    fov_degrees = frame.get('fov_degrees', 60.0)
                    fov_rad = math.radians(fov_degrees)
                    
                    # Load the image
                    image = self._load_image(image_path)
                    if image is None:
                        continue
                    
                    # Process the image with motion data
                    if method == "projection":
                        self._process_image_projection_python(
                            image, camera_position, direction, fov_rad)
                    elif method == "space_carving":
                        self._process_image_space_carving_python(
                            image, camera_position, direction, fov_rad)
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing frame: {e}")
            
            if success_count == 0:
                logger.error("No frames were successfully processed")
                return False
            
            # Add to processed files
            self.processed_files.append(str(json_path))
            
            # Apply post-processing
            self._apply_post_processing()
            
            logger.info(f"Processed {success_count} frames from motion data")
            return True
            
        except Exception as e:
            logger.error(f"Error processing motion data {json_path}: {e}")
            raise ProcessingError(f"Error processing motion data: {e}")
    
    @timeit
    def extract_mesh(self, method="marching_cubes", level=0.5):
        """
        Extract a surface mesh from the voxel grid
        
        Args:
            method: Extraction method ('marching_cubes', 'contour')
            level: Isovalue for surface extraction
            
        Returns:
            The extracted mesh object or None if failed
        """
        try:
            logger.info(f"Extracting mesh using {method} at level {level}")
            
            if method == "marching_cubes":
                from skimage import measure
                
                # Extract the isosurface
                verts, faces, normals, values = measure.marching_cubes(
                    self.voxel_grid, level=level)
                
                # Scale vertices to match voxel grid extents
                x_min, x_max = self.voxel_grid_extent[0]
                y_min, y_max = self.voxel_grid_extent[1]
                z_min, z_max = self.voxel_grid_extent[2]
                
                verts[:, 0] = x_min + verts[:, 0] * (x_max - x_min) / self.voxel_grid.shape[0]
                verts[:, 1] = y_min + verts[:, 1] * (y_max - y_min) / self.voxel_grid.shape[1]
                verts[:, 2] = z_min + verts[:, 2] * (z_max - z_min) / self.voxel_grid.shape[2]
                
                # Store mesh in PyVista format if available
                if PYVISTA_AVAILABLE:
                    import pyvista as pv
                    
                    # Prepare faces for PyVista (need to handle different face formats)
                    # scikit-image's marching_cubes returns triangular faces
                    if faces.shape[1] == 3:
                        # Faces are already triangles, need to add count (3) to each face
                        face_array = np.column_stack((np.full(len(faces), 3), faces))
                        face_array = face_array.flatten()
                    else:
                        # Faces include count in first column
                        face_array = faces.flatten()
                        
                    # Create polydata mesh
                    mesh = pv.PolyData(verts, face_array)
                    mesh["Normals"] = normals
                    
                    self.output_mesh = mesh
                    logger.info(f"Extracted mesh with {mesh.n_points} points and {mesh.n_faces} faces")
                    return mesh
                else:
                    # Return basic mesh data
                    logger.info(f"Extracted mesh with {len(verts)} vertices and {len(faces)} faces")
                    self.output_mesh = (verts, faces)
                    return self.output_mesh
            
            elif method == "contour" and PYVISTA_AVAILABLE:
                import pyvista as pv
                
                # Create a PyVista UniformGrid
                grid = pv.UniformGrid()
                grid.dimensions = np.array(self.voxel_grid.shape) + 1
                
                # Set the grid origin and spacing
                x_min, x_max = self.voxel_grid_extent[0]
                y_min, y_max = self.voxel_grid_extent[1]
                z_min, z_max = self.voxel_grid_extent[2]
                
                grid.origin = [x_min, y_min, z_min]
                grid.spacing = [(x_max - x_min) / self.voxel_grid.shape[0],
                              (y_max - y_min) / self.voxel_grid.shape[1],
                              (z_max - z_min) / self.voxel_grid.shape[2]]
                
                # Add the data to the grid
                grid.cell_data["values"] = self.voxel_grid.flatten(order="F")
                
                # Extract the isosurface
                mesh = grid.contour([level])
                
                self.output_mesh = mesh
                logger.info(f"Extracted mesh with {mesh.n_points} points and {mesh.n_faces} faces")
                return mesh
            
            else:
                logger.error(f"Unsupported mesh extraction method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting mesh: {e}")
            raise ProcessingError(f"Error extracting mesh: {e}")
    
    def get_voxel_data(self):
        """
        Get the current voxel grid data and extents for visualization
        
        Returns:
            points, intensities - 3D coordinates and intensity values for visualization
        """
        # If we've processed files, normalize the voxel grid by the number of files
        if self.processed_files:
            voxel_grid_avg = self.voxel_grid / len(self.processed_files)
        else:
            voxel_grid_avg = self.voxel_grid
        
        # Apply threshold to find significant voxels
        if np.any(voxel_grid_avg > 0):
            # If we have actual range of intensities, use percentile
            if np.max(voxel_grid_avg) > np.min(voxel_grid_avg[voxel_grid_avg > 0]):
                threshold = np.percentile(
                    voxel_grid_avg[voxel_grid_avg > 0], 
                    self.brightness_threshold_percentile)
            else:
                # If all values are the same (like binary mask), use a small value
                threshold = np.min(voxel_grid_avg[voxel_grid_avg > 0]) * 0.5
        else:
            threshold = 0
        
        object_voxels = voxel_grid_avg > threshold
        x_indices, y_indices, z_indices = np.nonzero(object_voxels)
        
        if len(x_indices) == 0:
            logger.warning("No significant voxels found to visualize.")
            return None, None
        
        # Convert voxel indices to spatial coordinates
        nx, ny, nz = voxel_grid_avg.shape
        x_min, x_max = self.voxel_grid_extent[0]
        y_min, y_max = self.voxel_grid_extent[1]
        z_min, z_max = self.voxel_grid_extent[2]
        
        x_coords = x_indices / nx * (x_max - x_min) + x_min
        y_coords = y_indices / ny * (y_max - y_min) + y_min
        z_coords = z_indices / nz * (z_max - z_min) + z_min
        
        intensities = voxel_grid_avg[object_voxels]
        
        # Stack coordinates into points array
        points = np.column_stack((x_coords, y_coords, z_coords))
        
        return points, intensities
    
    @timeit
    def export_voxel_data(self, output_path, format="vti"):
        """
        Export the voxel data to a file
        
        Args:
            output_path: Path to save the file
            format: Export format ('vti', 'nrrd', 'nii', 'raw')
            
        Returns:
            Success status (boolean)
        """
        try:
            output_path = Path(output_path)
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on format
            if format == "vti" and PYVISTA_AVAILABLE:
                import pyvista as pv
                
                # Create a PyVista UniformGrid
                grid = pv.UniformGrid()
                grid.dimensions = np.array(self.voxel_grid.shape) + 1
                
                # Set the grid origin and spacing
                x_min, x_max = self.voxel_grid_extent[0]
                y_min, y_max = self.voxel_grid_extent[1]
                z_min, z_max = self.voxel_grid_extent[2]
                
                grid.origin = [x_min, y_min, z_min]
                grid.spacing = [(x_max - x_min) / self.voxel_grid.shape[0],
                              (y_max - y_min) / self.voxel_grid.shape[1],
                              (z_max - z_min) / self.voxel_grid.shape[2]]
                
                # Add the data to the grid
                grid.cell_data["values"] = self.voxel_grid.flatten(order="F")
                
                # Save to VTI file
                grid.save(str(output_path))
                logger.info(f"Exported voxel data to VTI file: {output_path}")
                
            elif format == "nrrd":
                import nrrd
                
                # Create header
                header = {
                    'space': 'right-anterior-superior',
                    'space directions': [
                        [(self.voxel_grid_extent[0][1] - self.voxel_grid_extent[0][0]) / self.voxel_grid.shape[0], 0, 0],
                        [0, (self.voxel_grid_extent[1][1] - self.voxel_grid_extent[1][0]) / self.voxel_grid.shape[1], 0],
                        [0, 0, (self.voxel_grid_extent[2][1] - self.voxel_grid_extent[2][0]) / self.voxel_grid.shape[2]]
                    ],
                    'space origin': [
                        self.voxel_grid_extent[0][0],
                        self.voxel_grid_extent[1][0],
                        self.voxel_grid_extent[2][0]
                    ]
                }
                
                # Write NRRD file
                nrrd.write(str(output_path), self.voxel_grid, header)
                logger.info(f"Exported voxel data to NRRD file: {output_path}")
                
            elif format == "nii":
                import nibabel as nib
                
                # Create affine matrix
                affine = np.eye(4)
                
                # Set voxel spacing
                affine[0, 0] = (self.voxel_grid_extent[0][1] - self.voxel_grid_extent[0][0]) / self.voxel_grid.shape[0]
                affine[1, 1] = (self.voxel_grid_extent[1][1] - self.voxel_grid_extent[1][0]) / self.voxel_grid.shape[1]
                affine[2, 2] = (self.voxel_grid_extent[2][1] - self.voxel_grid_extent[2][0]) / self.voxel_grid.shape[2]
                
                # Set origin
                affine[0, 3] = self.voxel_grid_extent[0][0]
                affine[1, 3] = self.voxel_grid_extent[1][0]
                affine[2, 3] = self.voxel_grid_extent[2][0]
                
                # Create NIfTI image
                nifti_img = nib.Nifti1Image(self.voxel_grid, affine)
                
                # Save to file
                nib.save(nifti_img, str(output_path))
                logger.info(f"Exported voxel data to NIfTI file: {output_path}")
                
            elif format == "raw":
                # Save raw binary data
                with open(output_path, 'wb') as f:
                    # Write dimensions as integers
                    for dim in self.voxel_grid.shape:
                        f.write(np.int32(dim).tobytes())
                    
                    # Write grid extents as floats
                    for extent in self.voxel_grid_extent:
                        for value in extent:
                            f.write(np.float64(value).tobytes())
                    
                    # Write voxel data
                    self.voxel_grid.tofile(f)
                
                logger.info(f"Exported voxel data to raw file: {output_path}")
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting voxel data: {e}")
            raise ProcessingError(f"Error exporting voxel data: {e}")
    
    @timeit
    def export_mesh(self, output_path, format="stl"):
        """
        Export the extracted mesh to a file
        
        Args:
            output_path: Path to save the file
            format: Export format ('stl', 'obj', 'ply', 'vtp')
            
        Returns:
            Success status (boolean)
        """
        try:
            output_path = Path(output_path)
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Make sure we have a mesh
            if self.output_mesh is None:
                # Extract mesh if not done yet
                self.extract_mesh()
                
            if self.output_mesh is None:
                logger.error("No mesh to export")
                return False
            
            # PyVista mesh
            if PYVISTA_AVAILABLE and hasattr(self.output_mesh, "save"):
                # Save using PyVista
                self.output_mesh.save(str(output_path))
                logger.info(f"Exported mesh to {format} file: {output_path}")
                return True
            # Tuple of (vertices, faces)
            elif isinstance(self.output_mesh, tuple) and len(self.output_mesh) == 2:
                verts, faces = self.output_mesh
                
                if format == "stl":
                    # Export as STL
                    self._export_mesh_to_stl(verts, faces, output_path)
                elif format == "obj":
                    # Export as OBJ
                    self._export_mesh_to_obj(verts, faces, output_path)
                elif format == "ply":
                    # Export as PLY
                    self._export_mesh_to_ply(verts, faces, output_path)
                else:
                    logger.error(f"Unsupported export format for basic mesh: {format}")
                    return False
                
                logger.info(f"Exported mesh to {format} file: {output_path}")
                return True
            else:
                logger.error("Mesh in unrecognized format")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting mesh: {e}")
            raise ProcessingError(f"Error exporting mesh: {e}")
    
    def _export_mesh_to_stl(self, verts, faces, output_path):
        """Export mesh as STL file"""
        # For simple STL export
        with open(output_path, 'w') as f:
            f.write("solid voxel_mesh\n")
            
            for face in faces:
                # Get vertices of this face
                v1 = verts[face[0]]
                v2 = verts[face[1]]
                v3 = verts[face[2]]
                
                # Calculate face normal (simplified)
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Write face
                f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("  outer loop\n")
                f.write(f"    vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"    vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write(f"    vertex {v3[0]} {v3[1]} {v3[2]}\n")
                f.write("  endloop\n")
                f.write("endfacet\n")
            
            f.write("endsolid voxel_mesh\n")
    
    def _export_mesh_to_obj(self, verts, faces, output_path):
        """Export mesh as OBJ file"""
        with open(output_path, 'w') as f:
            f.write("# Voxel Projector Mesh\n")
            
            # Write vertices
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def _export_mesh_to_ply(self, verts, faces, output_path):
        """Export mesh as PLY file"""
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")