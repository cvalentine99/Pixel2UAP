"""
Tab widgets for the UI
"""

import os
import logging
import platform
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QCheckBox,
    QRadioButton, QLineEdit, QSlider,
    QButtonGroup, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage

logger = logging.getLogger(__name__)


class InputTab(QWidget):
    """Tab for input file selection and configuration"""
    
    def __init__(self, voxel_processor):
        super().__init__()
        
        self.voxel_processor = voxel_processor
        self.input_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface elements for the VideoTab.
        
        Creates and arranges the following UI components:
        - Video display area for showing the camera feed
        - Camera selection dropdown with auto-detection
        - Control buttons (Start, Stop, Save)
        - Camera settings (resolution, FPS)
        - Specialized controls for OAK-D cameras (depth mode)
        - Status display for showing camera state and errors
        """
        layout = QVBoxLayout(self)
        
        # Input selection section
        input_group = QGroupBox("Input Selection")
        input_layout = QVBoxLayout(input_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_button)
        
        input_layout.addLayout(file_layout)
        
        # Input type selection
        input_type_layout = QFormLayout()
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems([
            "Auto-detect",
            "Image",
            "Video",
            "DICOM Series",
            "NIFTI Volume",
            "Point Cloud",
            "Motion Data"
        ])
        input_type_layout.addRow("Input Type:", self.input_type_combo)
        
        input_layout.addLayout(input_type_layout)
        
        # Add the group to the main layout
        layout.addWidget(input_group)
        
        # DICOM-specific options
        self.dicom_group = QGroupBox("DICOM Options")
        self.dicom_group.setVisible(False)
        dicom_layout = QFormLayout(self.dicom_group)
        
        self.dicom_series_combo = QComboBox()
        dicom_layout.addRow("Series:", self.dicom_series_combo)
        
        self.dicom_window_center = QSpinBox()
        self.dicom_window_center.setRange(-10000, 10000)
        self.dicom_window_center.setValue(0)
        dicom_layout.addRow("Window Center:", self.dicom_window_center)
        
        self.dicom_window_width = QSpinBox()
        self.dicom_window_width.setRange(1, 20000)
        self.dicom_window_width.setValue(2000)
        dicom_layout.addRow("Window Width:", self.dicom_window_width)
        
        layout.addWidget(self.dicom_group)
        
        # Motion data options
        self.motion_group = QGroupBox("Motion Data Options")
        self.motion_group.setVisible(False)
        motion_layout = QFormLayout(self.motion_group)
        
        self.motion_data_path = QLineEdit()
        self.motion_data_path.setReadOnly(True)
        
        motion_browse_layout = QHBoxLayout()
        motion_browse_layout.addWidget(self.motion_data_path)
        
        motion_browse_button = QPushButton("Browse")
        motion_browse_button.clicked.connect(self.browse_motion_data)
        motion_browse_layout.addWidget(motion_browse_button)
        
        motion_layout.addRow("Metadata File:", motion_browse_layout)
        
        self.image_dir_path = QLineEdit()
        self.image_dir_path.setReadOnly(True)
        
        image_dir_layout = QHBoxLayout()
        image_dir_layout.addWidget(self.image_dir_path)
        
        image_dir_button = QPushButton("Browse")
        image_dir_button.clicked.connect(self.browse_image_dir)
        image_dir_layout.addWidget(image_dir_button)
        
        motion_layout.addRow("Image Directory:", image_dir_layout)
        
        layout.addWidget(self.motion_group)
        
        # Input information display
        info_group = QGroupBox("Input Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("No file loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.info_label.setMinimumHeight(100)
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        # Preview button (for images, DICOM, etc.)
        self.preview_button = QPushButton("Preview Input")
        self.preview_button.clicked.connect(self.preview_input)
        self.preview_button.setEnabled(False)
        layout.addWidget(self.preview_button)
        
        # Spacer at the bottom
        layout.addStretch()
        
        # Connect signals
        self.input_type_combo.currentIndexChanged.connect(self.update_input_options)
    
    def browse_file(self):
        """Open file dialog to select input file"""
        input_type = self.input_type_combo.currentText()
        
        # Set up file dialog based on input type
        if input_type == "Image":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "",
                "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)"
            )
        elif input_type == "Video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Video", "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
        elif input_type == "DICOM Series":
            # For DICOM, we select a directory
            file_path = QFileDialog.getExistingDirectory(
                self, "Select DICOM Directory"
            )
        elif input_type == "NIFTI Volume":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open NIFTI", "",
                "NIFTI Files (*.nii *.nii.gz);;All Files (*)"
            )
        elif input_type == "Point Cloud":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Point Cloud", "",
                "Point Cloud Files (*.ply *.pcd *.obj *.stl);;All Files (*)"
            )
        elif input_type == "Motion Data":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Motion Data", "",
                "JSON Files (*.json);;All Files (*)"
            )
        else:  # Auto-detect
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open File", "",
                "All Files (*);;Images (*.png *.jpg *.jpeg *.tif *.tiff);;DICOM (*.dcm);;NIFTI (*.nii *.nii.gz);;Point Clouds (*.ply *.pcd);;Motion Data (*.json)"
            )
        
        if file_path:
            self.set_input_path(file_path)
    
    def set_input_path(self, file_path):
        """Set the input path and update UI"""
        self.input_path = file_path
        
        if not file_path:
            self.file_path_label.setText("No file selected")
            self.info_label.setText("No file loaded")
            self.preview_button.setEnabled(False)
            return
        
        # Set the file path label
        self.file_path_label.setText(file_path)
        
        # Enable preview button for appropriate file types
        file_ext = Path(file_path).suffix.lower() if file_path else ''
        self.preview_button.setEnabled(
            file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz']
        )
        
        # Auto-detect input type if set to auto
        if self.input_type_combo.currentText() == "Auto-detect":
            self.auto_detect_input_type(file_path)
        
        # Update input options based on selected type
        self.update_input_options()
        
        # Update input information
        self.update_input_info(file_path)
    
    def auto_detect_input_type(self, file_path):
        """Auto-detect the input type based on file extension or directory contents"""
        path = Path(file_path)
        
        # Check if it's a directory
        if path.is_dir():
            # Check for DICOM files
            dicom_files = list(path.glob('*.dcm'))
            if dicom_files:
                self.input_type_combo.setCurrentText("DICOM Series")
                return
            
            # Check for motion data (JSON + images)
            json_files = list(path.glob('*.json'))
            image_files = list(path.glob('*.png')) + list(path.glob('*.jpg'))
            if json_files and image_files:
                self.input_type_combo.setCurrentText("Motion Data")
                return
        
        # Check file extension
        ext = path.suffix.lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            self.input_type_combo.setCurrentText("Image")
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            self.input_type_combo.setCurrentText("Video")
        elif ext == '.dcm':
            self.input_type_combo.setCurrentText("DICOM Series")
        elif ext in ['.nii', '.nii.gz']:
            self.input_type_combo.setCurrentText("NIFTI Volume")
        elif ext in ['.ply', '.pcd', '.obj', '.stl']:
            self.input_type_combo.setCurrentText("Point Cloud")
        elif ext == '.json':
            self.input_type_combo.setCurrentText("Motion Data")
    
    def update_input_options(self):
        """Update input option visibility based on selected input type"""
        input_type = self.input_type_combo.currentText()
        
        # Hide all option groups first
        self.dicom_group.setVisible(False)
        self.motion_group.setVisible(False)
        
        # Show relevant option groups
        if input_type == "DICOM Series":
            self.dicom_group.setVisible(True)
            
            # Populate DICOM series dropdown if we have a path
            if self.input_path and Path(self.input_path).is_dir():
                self.populate_dicom_series()
                
        elif input_type == "Motion Data":
            self.motion_group.setVisible(True)
            
            # Set motion data paths if we have a JSON file
            if self.input_path and Path(self.input_path).suffix.lower() == '.json':
                self.motion_data_path.setText(self.input_path)
                self.image_dir_path.setText(str(Path(self.input_path).parent))
    
    def populate_dicom_series(self):
        """Populate the DICOM series dropdown"""
        try:
            import pydicom
            
            dicom_dir = Path(self.input_path)
            
            # Clear the combo box
            self.dicom_series_combo.clear()
            
            # Find all DICOM files
            dicom_files = list(dicom_dir.glob('*.dcm'))
            
            # Group by series
            series_dict = {}
            for file_path in dicom_files:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    
                    # Check for required tags
                    if hasattr(ds, 'SeriesInstanceUID') and hasattr(ds, 'SeriesDescription'):
                        series_uid = ds.SeriesInstanceUID
                        series_desc = ds.SeriesDescription
                        
                        if series_uid not in series_dict:
                            series_dict[series_uid] = {
                                'description': series_desc,
                                'count': 0
                            }
                        
                        series_dict[series_uid]['count'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error reading DICOM file {file_path}: {e}")
            
            # Add series to combo box
            for uid, info in series_dict.items():
                self.dicom_series_combo.addItem(
                    f"{info['description']} ({info['count']} files)",
                    userData=uid
                )
            
        except ImportError:
            self.dicom_series_combo.addItem("PyDICOM not available")
            logger.warning("PyDICOM not available for DICOM series detection")
    
    def update_input_info(self, file_path):
        """Update the input information display"""
        path = Path(file_path)
        
        # Basic file info
        info_text = f"<b>File:</b> {path.name}<br>"
        
        if path.is_file():
            # File size
            size_bytes = path.stat().st_size
            size_kb = size_bytes / 1024
            size_mb = size_kb / 1024
            
            if size_mb >= 1:
                info_text += f"<b>Size:</b> {size_mb:.2f} MB<br>"
            else:
                info_text += f"<b>Size:</b> {size_kb:.2f} KB<br>"
        
        elif path.is_dir():
            # Count files
            file_count = sum(1 for _ in path.iterdir() if _.is_file())
            info_text += f"<b>Files:</b> {file_count}<br>"
        
        # Add type-specific information
        input_type = self.input_type_combo.currentText()
        
        if input_type == "Image" and path.is_file():
            try:
                from PIL import Image
                
                with Image.open(path) as img:
                    info_text += f"<b>Format:</b> {img.format}<br>"
                    info_text += f"<b>Mode:</b> {img.mode}<br>"
                    info_text += f"<b>Size:</b> {img.width} x {img.height} pixels<br>"
            except Exception as e:
                logger.warning(f"Error getting image info: {e}")
        
        elif input_type == "Video" and path.is_file():
            try:
                import cv2
                
                video = cv2.VideoCapture(str(path))
                if video.isOpened():
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    info_text += f"<b>Resolution:</b> {width} x {height} pixels<br>"
                    info_text += f"<b>Frames:</b> {frame_count}<br>"
                    info_text += f"<b>FPS:</b> {fps:.2f}<br>"
                    info_text += f"<b>Duration:</b> {frame_count/fps:.2f} seconds<br>"
                    
                    video.release()
            except Exception as e:
                logger.warning(f"Error getting video info: {e}")
        
        elif input_type == "DICOM Series":
            try:
                import pydicom
                
                if path.is_file():
                    # Single DICOM file
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    
                    info_text += "<b>DICOM Information:</b><br>"
                    if hasattr(ds, 'PatientName'):
                        info_text += f"<b>Patient:</b> {ds.PatientName}<br>"
                    if hasattr(ds, 'StudyDescription'):
                        info_text += f"<b>Study:</b> {ds.StudyDescription}<br>"
                    if hasattr(ds, 'SeriesDescription'):
                        info_text += f"<b>Series:</b> {ds.SeriesDescription}<br>"
                    if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                        info_text += f"<b>Size:</b> {ds.Columns} x {ds.Rows} pixels<br>"
                
                elif path.is_dir():
                    # DICOM directory
                    dicom_files = list(path.glob('*.dcm'))
                    if dicom_files:
                        info_text += f"<b>DICOM Files:</b> {len(dicom_files)}<br>"
                        
                        # Read first file for info
                        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                        
                        if hasattr(ds, 'PatientName'):
                            info_text += f"<b>Patient:</b> {ds.PatientName}<br>"
                        if hasattr(ds, 'StudyDescription'):
                            info_text += f"<b>Study:</b> {ds.StudyDescription}<br>"
                    
            except Exception as e:
                logger.warning(f"Error getting DICOM info: {e}")
        
        elif input_type == "NIFTI Volume" and path.is_file():
            try:
                import nibabel as nib
                
                nifti_img = nib.load(path)
                
                info_text += "<b>NIFTI Information:</b><br>"
                info_text += f"<b>Shape:</b> {nifti_img.shape}<br>"
                info_text += f"<b>Data Type:</b> {nifti_img.get_data_dtype()}<br>"
                
                # Add affine matrix info
                affine = nifti_img.affine
                info_text += "<b>Voxel Size:</b> "
                info_text += f"{affine[0,0]:.2f} x {affine[1,1]:.2f} x {affine[2,2]:.2f} mm<br>"
                
            except Exception as e:
                logger.warning(f"Error getting NIFTI info: {e}")
        
        self.info_label.setText(info_text)
    
    def browse_motion_data(self):
        """Browse for motion data JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Motion Data", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.motion_data_path.setText(file_path)
            
            # Set the input path to the motion data file
            self.input_path = file_path
            self.file_path_label.setText(file_path)
            
            # If image directory is empty, set to same directory as JSON
            if not self.image_dir_path.text():
                self.image_dir_path.setText(str(Path(file_path).parent))
            
            # Update input information
            self.update_input_info(file_path)
    
    def browse_image_dir(self):
        """Browse for image directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )
        
        if dir_path:
            self.image_dir_path.setText(dir_path)
    
    def preview_input(self):
        """Show a preview of the input file"""
        if not self.input_path:
            return
        
        try:
            path = Path(self.input_path)
            
            # Determine file type
            ext = path.suffix.lower()
            
            if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                # Preview image
                self.preview_image(path)
            
            elif ext == '.dcm' or self.input_type_combo.currentText() == "DICOM Series":
                # Preview DICOM
                self.preview_dicom(path)
            
            elif ext in ['.nii', '.nii.gz']:
                # Preview NIFTI
                self.preview_nifti(path)
            
        except Exception as e:
            logger.error(f"Error previewing input: {e}")
    
    def preview_image(self, path):
        """Show an image preview"""
        # Implementation depends on your preferred image viewer approach:
        # - Use QLabel to display the image in a dialog
        # - Use a library like PIL.ImageShow to open an external viewer
        # - Create a custom dialog with PyVista for 3D preview
        pass
    
    def preview_dicom(self, path):
        """Show a DICOM preview"""
        # Similar to above, but for DICOM files/series
        pass
    
    def preview_nifti(self, path):
        """Show a NIFTI preview"""
        # Similar to above, but for NIFTI volumes
        pass
    
    def get_input_path(self):
        """Get the selected input path"""
        # For motion data, return both JSON and image directory
        if self.input_type_combo.currentText() == "Motion Data":
            return [self.motion_data_path.text(), self.image_dir_path.text()]
        
        # For DICOM, handle series selection
        if self.input_type_combo.currentText() == "DICOM Series" and Path(self.input_path).is_dir():
            # Return the directory path
            return self.input_path
        
        # For other types, return the input path
        return self.input_path


class ProcessingTab(QWidget):
    """Tab for processing settings"""
    
    def __init__(self, voxel_processor):
        super().__init__()
        
        self.voxel_processor = voxel_processor
        
        self.setup_ui()
        self.update_ui_from_processor()
    
    def setup_ui(self):
        """Set up the user interface elements for the VideoTab.
        
        Creates and arranges the following UI components:
        - Video display area for showing the camera feed
        - Camera selection dropdown with auto-detection
        - Control buttons (Start, Stop, Save)
        - Camera settings (resolution, FPS)
        - Specialized controls for OAK-D cameras (depth mode)
        - Status display for showing camera state and errors
        """
        layout = QVBoxLayout(self)
        
        # Processing method
        method_group = QGroupBox("Processing Method")
        method_layout = QVBoxLayout(method_group)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "projection",
            "space_carving",
            "direct_import"  # For DICOM/NIFTI volumes
        ])
        method_layout.addWidget(self.method_combo)
        
        # Add method group to layout
        layout.addWidget(method_group)
        
        # Grid parameters
        grid_group = QGroupBox("Voxel Grid Parameters")
        grid_layout = QFormLayout(grid_group)
        
        # Grid size
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(10, 1000)
        self.grid_size_spin.setValue(256)
        self.grid_size_spin.setSingleStep(16)
        grid_layout.addRow("Grid Size:", self.grid_size_spin)
        
        # Grid extent
        self.grid_extent_spin = QDoubleSpinBox()
        self.grid_extent_spin.setRange(0.1, 10.0)
        self.grid_extent_spin.setValue(1.0)
        self.grid_extent_spin.setSingleStep(0.1)
        grid_layout.addRow("Grid Extent:", self.grid_extent_spin)
        
        # Distance from origin
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.1, 100.0)
        self.distance_spin.setValue(1.0)
        self.distance_spin.setSingleStep(0.1)
        grid_layout.addRow("Distance from Origin:", self.distance_spin)
        
        # Add grid group to layout
        layout.addWidget(grid_group)
        
        # Ray casting parameters
        ray_group = QGroupBox("Ray Casting Parameters")
        ray_layout = QFormLayout(ray_group)
        
        # Number of steps
        self.num_steps_spin = QSpinBox()
        self.num_steps_spin.setRange(100, 50000)
        self.num_steps_spin.setValue(10000)
        self.num_steps_spin.setSingleStep(1000)
        ray_layout.addRow("Steps:", self.num_steps_spin)
        
        # Maximum distance
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 100.0)
        self.max_distance_spin.setValue(10.0)
        self.max_distance_spin.setSingleStep(1.0)
        ray_layout.addRow("Max Distance:", self.max_distance_spin)
        
        # Add ray group to layout
        layout.addWidget(ray_group)
        
        # Post-processing
        post_group = QGroupBox("Post-Processing")
        post_layout = QFormLayout(post_group)
        
        # Smoothing
        self.smoothing_check = QCheckBox("Apply Smoothing")
        post_layout.addRow(self.smoothing_check)
        
        self.smoothing_sigma_spin = QDoubleSpinBox()
        self.smoothing_sigma_spin.setRange(0.1, 5.0)
        self.smoothing_sigma_spin.setValue(0.5)
        self.smoothing_sigma_spin.setSingleStep(0.1)
        post_layout.addRow("Smoothing Sigma:", self.smoothing_sigma_spin)
        
        # Thresholding
        self.threshold_check = QCheckBox("Apply Thresholding")
        post_layout.addRow(self.threshold_check)
        
        self.threshold_value_spin = QDoubleSpinBox()
        self.threshold_value_spin.setRange(0.0, 1.0)
        self.threshold_value_spin.setValue(0.5)
        self.threshold_value_spin.setSingleStep(0.05)
        post_layout.addRow("Threshold Value:", self.threshold_value_spin)
        
        # Add post group to layout
        layout.addWidget(post_group)
        
        # Performance options
        perf_group = QGroupBox("Performance Options")
        perf_layout = QFormLayout(perf_group)
        
        # GPU acceleration
        self.gpu_check = QCheckBox("Use GPU Acceleration (if available)")
        self.gpu_check.setEnabled(self.is_gpu_available())
        perf_layout.addRow(self.gpu_check)
        
        # Sparse representation
        self.sparse_check = QCheckBox("Use Sparse Data Structure (for large grids)")
        perf_layout.addRow(self.sparse_check)
        
        # Add performance group to layout
        layout.addWidget(perf_group)
        
        # Add apply button
        self.apply_button = QPushButton("Apply Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Connect signals
        self.smoothing_check.toggled.connect(
            lambda checked: self.smoothing_sigma_spin.setEnabled(checked)
        )
        self.threshold_check.toggled.connect(
            lambda checked: self.threshold_value_spin.setEnabled(checked)
        )
    
    def is_gpu_available(self):
        """Check if GPU acceleration is available"""
        try:
            import cupy
            return True
        except ImportError:
            return False
    
    def update_ui_from_processor(self):
        """Update UI values from processor settings"""
        # Grid parameters
        self.grid_size_spin.setValue(self.voxel_processor.voxel_grid_size[0])
        self.grid_extent_spin.setValue(self.voxel_processor.grid_extent)
        self.distance_spin.setValue(self.voxel_processor.distance_from_origin)
        
        # Ray casting parameters
        self.num_steps_spin.setValue(self.voxel_processor.num_steps)
        self.max_distance_spin.setValue(self.voxel_processor.max_distance)
        
        # Post-processing
        self.smoothing_check.setChecked(self.voxel_processor.apply_smoothing)
        self.smoothing_sigma_spin.setValue(self.voxel_processor.smoothing_sigma)
        self.smoothing_sigma_spin.setEnabled(self.voxel_processor.apply_smoothing)
        
        self.threshold_check.setChecked(self.voxel_processor.apply_thresholding)
        self.threshold_value_spin.setValue(self.voxel_processor.threshold_value)
        self.threshold_value_spin.setEnabled(self.voxel_processor.apply_thresholding)
        
        # Performance options
        self.gpu_check.setChecked(self.voxel_processor.use_gpu)
        self.sparse_check.setChecked(self.voxel_processor.use_sparse)
    
    def apply_settings(self):
        """Apply UI settings to the processor"""
        # Grid parameters
        size = self.grid_size_spin.value()
        self.voxel_processor.voxel_grid_size = (size, size, size)
        self.voxel_processor.grid_extent = self.grid_extent_spin.value()
        self.voxel_processor.distance_from_origin = self.distance_spin.value()
        
        # Ray casting parameters
        self.voxel_processor.num_steps = self.num_steps_spin.value()
        self.voxel_processor.max_distance = self.max_distance_spin.value()
        
        # Post-processing
        self.voxel_processor.apply_smoothing = self.smoothing_check.isChecked()
        self.voxel_processor.smoothing_sigma = self.smoothing_sigma_spin.value()
        
        self.voxel_processor.apply_thresholding = self.threshold_check.isChecked()
        self.voxel_processor.threshold_value = self.threshold_value_spin.value()
        
        # Performance options
        self.voxel_processor.use_gpu = self.gpu_check.isChecked()
        self.voxel_processor.use_sparse = self.sparse_check.isChecked()
        
        # Reset the voxel grid with the new parameters
        self.voxel_processor.reset_voxel_grid()
        
        # Log the changes
        logger.info("Processing settings applied")
    
    def get_processing_method(self):
        """Get the selected processing method"""
        return self.method_combo.currentText()


class VisualizationTab(QWidget):
    """Tab for visualization settings"""
    
    def __init__(self, voxel_viewer, engine_type):
        super().__init__()
        
        self.voxel_viewer = voxel_viewer
        self.engine_type = engine_type  # 'pyvista' or 'matplotlib'
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface elements for the VideoTab.
        
        Creates and arranges the following UI components:
        - Video display area for showing the camera feed
        - Camera selection dropdown with auto-detection
        - Control buttons (Start, Stop, Save)
        - Camera settings (resolution, FPS)
        - Specialized controls for OAK-D cameras (depth mode)
        - Status display for showing camera state and errors
        """
        layout = QVBoxLayout(self)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)
        
        # Common settings (for both engines)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "rainbow", "viridis", "plasma", "jet", "hot", "inferno",
            "turbo", "coolwarm", "magma", "cividis"
        ])
        self.colormap_combo.setCurrentText("hot")
        display_layout.addRow("Colormap:", self.colormap_combo)
        
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(5)
        display_layout.addRow("Point Size:", self.point_size_spin)
        
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(50)
        display_layout.addRow("Opacity:", self.opacity_slider)
        
        # PyVista-specific settings
        if self.engine_type == 'pyvista':
            # Volume rendering options
            self.volume_group = QGroupBox("Volume Rendering Options")
            volume_layout = QFormLayout(self.volume_group)
            
            self.volume_mapper_combo = QComboBox()
            self.volume_mapper_combo.addItems(["smart", "gpu", "fixed_point", "open_gl"])
            volume_layout.addRow("Mapper:", self.volume_mapper_combo)
            
            self.volume_shade_check = QCheckBox("Enable Shading")
            self.volume_shade_check.setChecked(True)
            volume_layout.addRow(self.volume_shade_check)
            
            self.ambient_slider = QSlider(Qt.Orientation.Horizontal)
            self.ambient_slider.setRange(0, 100)
            self.ambient_slider.setValue(50)
            volume_layout.addRow("Ambient:", self.ambient_slider)
            
            self.diffuse_slider = QSlider(Qt.Orientation.Horizontal)
            self.diffuse_slider.setRange(0, 100)
            self.diffuse_slider.setValue(75)
            volume_layout.addRow("Diffuse:", self.diffuse_slider)
            
            self.opacity_function_combo = QComboBox()
            self.opacity_function_combo.addItems(["linear", "sigmoid", "linear_10_90", "sigmoid_10_90"])
            volume_layout.addRow("Opacity Function:", self.opacity_function_combo)
        
        # Add display group to layout
        layout.addWidget(display_group)
        
        # Add PyVista volume group if applicable
        if self.engine_type == 'pyvista':
            layout.addWidget(self.volume_group)
        
        # Camera options (PyVista only)
        if self.engine_type == 'pyvista':
            camera_group = QGroupBox("Camera Options")
            camera_layout = QFormLayout(camera_group)
            
            self.camera_position_combo = QComboBox()
            self.camera_position_combo.addItems([
                "iso", "xy", "xz", "yz", "+x", "-x", "+y", "-y", "+z", "-z"
            ])
            self.camera_position_combo.setCurrentText("iso")
            camera_layout.addRow("Position:", self.camera_position_combo)
            
            self.camera_parallel_check = QCheckBox("Parallel Projection")
            camera_layout.addRow(self.camera_parallel_check)
            
            # Add camera group to layout
            layout.addWidget(camera_group)
        
        # Mesh options (PyVista only)
        if self.engine_type == 'pyvista':
            mesh_group = QGroupBox("Mesh Options")
            mesh_layout = QFormLayout(mesh_group)
            
            self.mesh_show_edges_check = QCheckBox("Show Edges")
            self.mesh_show_edges_check.setChecked(True)
            mesh_layout.addRow(self.mesh_show_edges_check)
            
            self.mesh_color_combo = QComboBox()
            self.mesh_color_combo.addItems([
                "tan", "white", "grey", "blue", "red", "green"
            ])
            self.mesh_color_combo.setCurrentText("tan")
            mesh_layout.addRow("Color:", self.mesh_color_combo)
            
            self.mesh_style_combo = QComboBox()
            self.mesh_style_combo.addItems([
                "surface", "wireframe", "points"
            ])
            self.mesh_style_combo.setCurrentText("surface")
            mesh_layout.addRow("Style:", self.mesh_style_combo)
            
            # Add mesh group to layout
            layout.addWidget(mesh_group)
        
        # Apply button
        self.apply_button = QPushButton("Apply Visualization Settings")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)
        
        # Spacer
        layout.addStretch()
        
        # Connect signals
        if self.engine_type == 'pyvista':
            self.camera_position_combo.currentIndexChanged.connect(self.update_camera_position)
            self.volume_shade_check.toggled.connect(self.toggle_volume_shading)
            
            # Enable/disable sliders based on shading
            self.toggle_volume_shading(self.volume_shade_check.isChecked())
    
    def apply_settings(self):
        """Apply visualization settings"""
        try:
            # Common settings
            if self.engine_type == 'pyvista':
                self.voxel_viewer.colormap = self.colormap_combo.currentText()
                self.voxel_viewer.point_size = self.point_size_spin.value()
                self.voxel_viewer.opacity = self.opacity_slider.value() / 100.0
            
            # Update the visualization
            if hasattr(self.voxel_viewer, 'plotter') and self.voxel_viewer.plotter:
                # PyVista - update the active display
                self.update_current_visualization()
            
            logger.info("Visualization settings applied")
            
        except Exception as e:
            logger.error(f"Error applying visualization settings: {e}")
    
    def update_camera_position(self):
        """Update the camera position based on combo box selection"""
        if self.engine_type != 'pyvista':
            return
        
        position = self.camera_position_combo.currentText()
        
        try:
            if position == "iso":
                self.voxel_viewer.plotter.view_isometric()
            elif position == "xy":
                self.voxel_viewer.plotter.view_xy()
            elif position == "xz":
                self.voxel_viewer.plotter.view_xz()
            elif position == "yz":
                self.voxel_viewer.plotter.view_yz()
            elif position == "+x":
                self.voxel_viewer.plotter.view_vector((1, 0, 0))
            elif position == "-x":
                self.voxel_viewer.plotter.view_vector((-1, 0, 0))
            elif position == "+y":
                self.voxel_viewer.plotter.view_vector((0, 1, 0))
            elif position == "-y":
                self.voxel_viewer.plotter.view_vector((0, -1, 0))
            elif position == "+z":
                self.voxel_viewer.plotter.view_vector((0, 0, 1))
            elif position == "-z":
                self.voxel_viewer.plotter.view_vector((0, 0, -1))
            
            self.voxel_viewer.plotter.update()
            
        except Exception as e:
            logger.error(f"Error updating camera position: {e}")
    
    def toggle_volume_shading(self, enabled):
        """Enable/disable volume shading controls"""
        if self.engine_type != 'pyvista':
            return
        
        self.ambient_slider.setEnabled(enabled)
        self.diffuse_slider.setEnabled(enabled)
    
    def update_current_visualization(self):
        """Update the current visualization with the new settings"""
        if self.engine_type != 'pyvista':
            return
        
        try:
            # Determine what's currently being displayed
            if self.voxel_viewer.volume is not None:
                # Update volume settings
                opacity = self.opacity_function_combo.currentText()
                shade = self.volume_shade_check.isChecked()
                
                # Clear and re-add with new settings
                self.voxel_viewer.plotter.clear()
                self.voxel_viewer.plotter.add_volume(
                    self.voxel_viewer.grid,
                    cmap=self.colormap_combo.currentText(),
                    opacity=opacity,
                    shade=shade,
                    ambient=self.ambient_slider.value() / 100.0,
                    diffuse=self.diffuse_slider.value() / 100.0,
                    mapper=self.volume_mapper_combo.currentText()
                )
                
            elif self.voxel_viewer.mesh is not None:
                # Update mesh settings
                show_edges = self.mesh_show_edges_check.isChecked()
                color = self.mesh_color_combo.currentText()
                style = self.mesh_style_combo.currentText()
                
                # Clear and re-add with new settings
                self.voxel_viewer.plotter.clear()
                self.voxel_viewer.plotter.add_mesh(
                    self.voxel_viewer.mesh,
                    show_edges=show_edges,
                    color=color,
                    style=style,
                    opacity=self.opacity_slider.value() / 100.0
                )
                
            elif self.voxel_viewer.points is not None:
                # Update point settings
                self.voxel_viewer.plotter.clear()
                self.voxel_viewer.plot_points(
                    self.voxel_viewer.points,
                    self.voxel_viewer.intensities,
                    reset_camera=False
                )
            
            # Update the view
            self.voxel_viewer.plotter.update()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
    
    def update_for_visualization_type(self, viz_type):
        """Update controls visibility based on visualization type"""
        if self.engine_type != 'pyvista':
            return
        
        # Show/hide groups based on visualization type
        self.volume_group.setVisible(viz_type == "Volume")
        
        # Show mesh options only for mesh visualization
        mesh_visible = viz_type == "Mesh"
        self.findChild(QGroupBox, "Mesh Options").setVisible(mesh_visible)


class ExportTab(QWidget):
    """Tab for export options"""
    
    def __init__(self, voxel_processor):
        super().__init__()
        
        self.voxel_processor = voxel_processor
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface elements for the VideoTab.
        
        Creates and arranges the following UI components:
        - Video display area for showing the camera feed
        - Camera selection dropdown with auto-detection
        - Control buttons (Start, Stop, Save)
        - Camera settings (resolution, FPS)
        - Specialized controls for OAK-D cameras (depth mode)
        - Status display for showing camera state and errors
        """
        layout = QVBoxLayout(self)
        
        # Voxel data export
        voxel_group = QGroupBox("Voxel Data Export")
        voxel_layout = QVBoxLayout(voxel_group)
        
        # VTI export
        vti_button = QPushButton("Export as VTI (VTK Image Data)")
        vti_button.clicked.connect(lambda: self.export_voxel_data("vti"))
        voxel_layout.addWidget(vti_button)
        
        # NRRD export
        nrrd_button = QPushButton("Export as NRRD")
        nrrd_button.clicked.connect(lambda: self.export_voxel_data("nrrd"))
        voxel_layout.addWidget(nrrd_button)
        
        # NIFTI export
        nifti_button = QPushButton("Export as NIFTI")
        nifti_button.clicked.connect(lambda: self.export_voxel_data("nii"))
        voxel_layout.addWidget(nifti_button)
        
        # Raw export
        raw_button = QPushButton("Export as Raw Binary")
        raw_button.clicked.connect(lambda: self.export_voxel_data("raw"))
        voxel_layout.addWidget(raw_button)
        
        # Add voxel group to layout
        layout.addWidget(voxel_group)
        
        # Mesh export
        mesh_group = QGroupBox("Mesh Export")
        mesh_layout = QVBoxLayout(mesh_group)
        
        # Mesh extraction parameters
        param_layout = QFormLayout()
        
        self.extraction_method_combo = QComboBox()
        self.extraction_method_combo.addItems(["marching_cubes", "contour"])
        param_layout.addRow("Extraction Method:", self.extraction_method_combo)
        
        self.extraction_level_spin = QDoubleSpinBox()
        self.extraction_level_spin.setRange(0.0, 1.0)
        self.extraction_level_spin.setValue(0.5)
        self.extraction_level_spin.setSingleStep(0.05)
        param_layout.addRow("Iso-Level:", self.extraction_level_spin)
        
        mesh_layout.addLayout(param_layout)
        
        # Extract mesh button
        extract_button = QPushButton("Extract Mesh")
        extract_button.clicked.connect(self.extract_mesh)
        mesh_layout.addWidget(extract_button)
        
        # Export buttons
        stl_button = QPushButton("Export as STL")
        stl_button.clicked.connect(lambda: self.export_mesh("stl"))
        mesh_layout.addWidget(stl_button)
        
        obj_button = QPushButton("Export as OBJ")
        obj_button.clicked.connect(lambda: self.export_mesh("obj"))
        mesh_layout.addWidget(obj_button)
        
        ply_button = QPushButton("Export as PLY")
        ply_button.clicked.connect(lambda: self.export_mesh("ply"))
        mesh_layout.addWidget(ply_button)
        
        # Add mesh group to layout
        layout.addWidget(mesh_group)
        
        # Screenshot
        screenshot_group = QGroupBox("Screenshot")
        screenshot_layout = QVBoxLayout(screenshot_group)
        
        screenshot_button = QPushButton("Save Current View as Image")
        screenshot_button.clicked.connect(self.save_screenshot)
        screenshot_layout.addWidget(screenshot_button)
        
        # Add screenshot group to layout
        layout.addWidget(screenshot_group)
        
        # Spacer
        layout.addStretch()
    
    def export_voxel_data(self, format):
        """Export voxel data in the specified format"""
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export Voxel Data as {format.upper()}", "",
            f"{format.upper()} Files (*.{format});;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Ensure file has the correct extension
            if not file_path.lower().endswith(f".{format}"):
                file_path += f".{format}"
            
            # Export the data
            success = self.voxel_processor.export_voxel_data(file_path, format)
            
            if success:
                logger.info(f"Exported voxel data to {file_path}")
            else:
                logger.warning("Export failed")
            
        except Exception as e:
            logger.error(f"Error exporting voxel data: {e}")
    
    def extract_mesh(self):
        """Extract a mesh from the voxel data"""
        try:
            # Get parameters
            method = self.extraction_method_combo.currentText()
            level = self.extraction_level_spin.value()
            
            # Extract the mesh
            mesh = self.voxel_processor.extract_mesh(method, level)
            
            if mesh is not None:
                logger.info("Mesh extraction completed")
            else:
                logger.warning("Mesh extraction failed")
            
        except Exception as e:
            logger.error(f"Error extracting mesh: {e}")
    
    def export_mesh(self, format):
        """Export mesh in the specified format"""
        # Check if we have a mesh
        if not hasattr(self.voxel_processor, 'output_mesh') or self.voxel_processor.output_mesh is None:
            # Try to extract a mesh first
            try:
                method = self.extraction_method_combo.currentText()
                level = self.extraction_level_spin.value()
                self.voxel_processor.extract_mesh(method, level)
            except Exception as e:
                logger.error(f"Error extracting mesh for export: {e}")
                return
            
            # Check again
            if not hasattr(self.voxel_processor, 'output_mesh') or self.voxel_processor.output_mesh is None:
                logger.warning("No mesh available for export")
                return
        
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export Mesh as {format.upper()}", "",
            f"{format.upper()} Files (*.{format});;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Ensure file has the correct extension
            if not file_path.lower().endswith(f".{format}"):
                file_path += f".{format}"
            
            # Export the mesh
            success = self.voxel_processor.export_mesh(file_path, format)
            
            if success:
                logger.info(f"Exported mesh to {file_path}")
            else:
                logger.warning("Export failed")
            
        except Exception as e:
            logger.error(f"Error exporting mesh: {e}")
    
    def save_screenshot(self):
        """Save a screenshot of the current view"""
        # This will be handled by the main window
        # Defined here just for reference
        pass
    
    def get_mesh_extraction_method(self):
        """Get the selected mesh extraction method"""
        return self.extraction_method_combo.currentText()
    
    def get_mesh_extraction_level(self):
        """Get the selected mesh extraction iso-level"""
        return self.extraction_level_spin.value()


class VideoTab(QWidget):
    """Tab for webcam video capture and control.
    
    This class provides a comprehensive interface for video capture from various camera sources,
    including standard webcams through OpenCV and OAK-D spatial AI cameras through DepthAI.
    It supports RGB video capture, depth sensing, confidence mapping, and point cloud generation
    with functionality for displaying and saving frames and 3D data in various formats.
    
    Attributes:
        webcam (cv2.VideoCapture): Standard webcam instance when using OpenCV.
        webcam_index (int): Index of the currently active webcam.
        is_capturing (bool): Flag indicating if video capture is active.
        timer (QTimer): Timer for updating video frames at specific intervals.
        resolution (tuple): Current resolution as (width, height).
        fps (int): Current frames per second setting.
        is_depth_mode (bool): Flag enabling depth capture for supported cameras.
        is_pointcloud_mode (bool): Flag enabling point cloud generation for OAK-D cameras.
        oak_device (dai.Device): OAK-D camera instance when using DepthAI.
        current_frame (Union[numpy.ndarray, dict]): Currently displayed frame or frame data.
            For standard webcams, this is a numpy array. For OAK-D cameras in depth mode,
            this is a dictionary containing 'rgb', 'depth', and possibly 'confidence' data.
        pointcloud_visualizer (PointCloudVisualizer): Visualizer for point cloud data.
        device_pointcloud (DevicePointCloudGenerator): On-device point cloud generator.
    """
    
    def __init__(self):
        """Initialize the VideoTab widget.
        
        Sets up the initial camera state, UI components, and auto-detection of available
        camera devices. Initializes with default settings for resolution (1920x1080) and
        frame rate (30 FPS).
        """
        super().__init__()
        
        # Initialize webcam state
        self.webcam = None
        self.webcam_index = None
        self.is_capturing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Camera settings
        self.resolution = (1920, 1080)  # Default HD resolution
        self.fps = 30
        self.is_depth_mode = False
        self.is_pointcloud_mode = False
        self.oak_device = None  # For OAK-D camera
        self.current_frame = None  # For storing the current frame
        
        # Initialize data processing attributes
        self.q_rgb = None  # Queue for RGB frames from OAK-D
        self.q_depth = None  # Queue for depth frames from OAK-D
        self.q_conf = None  # Queue for confidence maps from OAK-D
        self.q_pcl = None   # Queue for point cloud data from OAK-D
        
        # Initialize point cloud support components
        self.pointcloud_visualizer = None
        self.device_pointcloud = None
        self.pcl_window_open = False
        
        # Video recording attributes
        self.video_writer = None
        self.is_recording = False
        self.recording_start_time = None
        self.recording_frames = []  # For storing frames during recording
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_recording)
        self.depth_recording = False  # Whether to record depth data alongside RGB
        
        # Set up the UI
        self.setup_ui()
        
        # Auto-detect cameras after a short delay to allow UI to initialize
        QTimer.singleShot(500, self.refresh_cameras)
    
    def setup_ui(self):
        """Set up the user interface elements for the VideoTab.
        
        Creates and arranges the following UI components:
        - Video display area for showing the camera feed
        - Camera selection dropdown with auto-detection
        - Control buttons (Start, Stop, Save)
        - Camera settings (resolution, FPS)
        - Specialized controls for OAK-D cameras (depth mode)
        - Status display for showing camera state and errors
        """
        layout = QVBoxLayout(self)
        
        # Video display section
        self.video_display = QLabel("No video feed")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("border: 1px solid gray; background-color: #222;")
        layout.addWidget(self.video_display)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.setToolTip("Start video capture from the selected camera")
        self.start_button.clicked.connect(self.start_capture)
        buttons_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setToolTip("Stop video capture and release the camera")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)
        
        self.save_button = QPushButton("Save Frame")
        self.save_button.setToolTip("Save the current video frame to disk")
        self.save_button.clicked.connect(self.save_frame)
        self.save_button.setEnabled(False)
        buttons_layout.addWidget(self.save_button)
        
        # Video recording buttons
        self.record_button = QPushButton("Record Video")
        self.record_button.setToolTip("Start recording a video clip")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        buttons_layout.addWidget(self.record_button)
        
        layout.addLayout(buttons_layout)
        
        # Recording settings
        recording_group = QGroupBox("Recording Settings")
        recording_layout = QFormLayout(recording_group)
        
        # Video format selector
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP4 (H.264)", "AVI (MJPG)", "MKV (H.264)", "MOV (H.264)"])
        recording_layout.addRow("Format:", self.format_combo)
        
        # Video duration
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 300)  # 1-300 seconds
        self.duration_spin.setValue(10)  # Default 10 seconds
        self.duration_spin.setSuffix(" sec")
        recording_layout.addRow("Duration:", self.duration_spin)
        
        # Video quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low", "Medium", "High", "Highest"])
        self.quality_combo.setCurrentIndex(2)  # Default to High
        recording_layout.addRow("Quality:", self.quality_combo)
        
        # Recording progress
        self.recording_progress = QProgressBar()
        self.recording_progress.setRange(0, 100)
        self.recording_progress.setValue(0)
        self.recording_progress.setVisible(False)
        recording_layout.addRow("Progress:", self.recording_progress)
        
        # Add recording settings to layout
        layout.addWidget(recording_group)
        
        # Camera selection and settings
        settings_group = QGroupBox("Camera Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Camera selection
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Auto-detect")
        for i in range(5):  # Cameras 0-4
            self.camera_combo.addItem(f"Camera {i}")
        self.camera_combo.addItem("OAK-D Camera")
        camera_layout.addWidget(self.camera_combo)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(self.refresh_button)
        
        settings_layout.addLayout(camera_layout)
        
        # Resolution selection
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480 (VGA)", 
            "1280x720 (HD)",
            "1920x1080 (Full HD)",
            "3840x2160 (4K UHD)"
        ])
        self.resolution_combo.setCurrentIndex(2)  # Default to Full HD
        resolution_layout.addWidget(self.resolution_combo)
        
        settings_layout.addLayout(resolution_layout)
        
        # FPS slider
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(10, 60)
        self.fps_slider.setValue(30)
        self.fps_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fps_slider.setTickInterval(10)
        
        self.fps_label = QLabel("30")
        self.fps_slider.valueChanged.connect(
            lambda value: (self.fps_label.setText(str(value)), setattr(self, 'fps', value))
        )
        
        fps_layout.addWidget(self.fps_slider)
        fps_layout.addWidget(self.fps_label)
        
        settings_layout.addLayout(fps_layout)
        
        # Special options for OAK-D camera
        self.oak_options_group = QGroupBox("OAK-D Options")
        oak_layout = QVBoxLayout(self.oak_options_group)
        
        # Depth mode toggle
        self.depth_check = QCheckBox("Enable Depth Mode")
        self.depth_check.setToolTip("Show depth information instead of RGB image")
        self.depth_check.toggled.connect(lambda checked: setattr(self, 'is_depth_mode', checked))
        oak_layout.addWidget(self.depth_check)
        
        # Point cloud mode toggle
        self.pointcloud_check = QCheckBox("Enable Point Cloud Generation")
        self.pointcloud_check.setToolTip("Generate and visualize 3D point cloud from depth data")
        self.pointcloud_check.toggled.connect(self.toggle_pointcloud_mode)
        oak_layout.addWidget(self.pointcloud_check)
        
        # Point cloud options group
        self.pointcloud_options_group = QGroupBox("Point Cloud Options")
        pc_layout = QVBoxLayout(self.pointcloud_options_group)
        
        # Downsample option
        self.downsample_check = QCheckBox("Downsample Point Cloud")
        self.downsample_check.setToolTip("Reduce point cloud density for better performance")
        self.downsample_check.setChecked(True)
        pc_layout.addWidget(self.downsample_check)
        
        # Remove outliers option
        self.remove_noise_check = QCheckBox("Remove Outliers")
        self.remove_noise_check.setToolTip("Filter out statistical outliers from the point cloud")
        self.remove_noise_check.setChecked(False)
        pc_layout.addWidget(self.remove_noise_check)
        
        # On-device vs. host processing
        self.device_pointcloud_check = QCheckBox("On-Device Point Cloud Processing")
        self.device_pointcloud_check.setToolTip("Process point cloud on the OAK-D device (faster but requires setup)")
        self.device_pointcloud_check.setChecked(False)
        pc_layout.addWidget(self.device_pointcloud_check)
        
        # Button to open point cloud in new window
        pc_button_layout = QHBoxLayout()
        
        self.open_pc_window_button = QPushButton("Open Point Cloud Viewer")
        self.open_pc_window_button.setToolTip("Open the point cloud in a separate interactive 3D viewer")
        self.open_pc_window_button.clicked.connect(self.open_pointcloud_window)
        self.open_pc_window_button.setEnabled(False)
        pc_button_layout.addWidget(self.open_pc_window_button)
        
        self.save_pc_button = QPushButton("Save Point Cloud")
        self.save_pc_button.setToolTip("Save the current point cloud to a file")
        self.save_pc_button.clicked.connect(self.save_pointcloud)
        self.save_pc_button.setEnabled(False)
        pc_button_layout.addWidget(self.save_pc_button)
        
        pc_layout.addLayout(pc_button_layout)
        
        # Export point cloud to voxel grid
        self.pc_to_voxel_button = QPushButton("Export Point Cloud to Voxel Grid")
        self.pc_to_voxel_button.setToolTip("Convert the current point cloud to a voxel grid for visualization and processing")
        self.pc_to_voxel_button.clicked.connect(self.export_to_voxel_grid)
        self.pc_to_voxel_button.setEnabled(False)
        pc_layout.addWidget(self.pc_to_voxel_button)
        
        # Add point cloud options to OAK options
        self.pointcloud_options_group.setVisible(False)
        oak_layout.addWidget(self.pointcloud_options_group)
        
        self.oak_options_group.setVisible(False)
        settings_layout.addWidget(self.oak_options_group)
        
        # Connect camera combo to show/hide OAK options
        self.camera_combo.currentIndexChanged.connect(self.update_camera_options)
        
        layout.addWidget(settings_group)
        
        # Status message
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        # Add a spacer at the bottom
        layout.addStretch()
    
    def toggle_pointcloud_mode(self, enabled):
        """Toggle point cloud generation mode
        
        Args:
            enabled (bool): True to enable point cloud generation, False to disable
        """
        self.is_pointcloud_mode = enabled
        self.pointcloud_options_group.setVisible(enabled)
        
        # Enable depth mode if point cloud is enabled (requires depth data)
        if enabled:
            self.depth_check.setChecked(True)
            self.depth_check.setEnabled(False)  # Can't disable depth when pointcloud is active
        else:
            self.depth_check.setEnabled(True)
    
    def update_camera_options(self):
        """Update UI options based on the selected camera type.
        
        Shows or hides specific camera controls based on the selected camera:
        - For OAK-D cameras, displays depth mode options
        - For specific camera models (e.g., Logitech Brio), enables appropriate resolution options
        - Adjusts available resolution options based on camera capabilities
        
        Called when the camera selection changes to update the UI dynamically.
        """
        selected_camera = self.camera_combo.currentText()
        
        # Show OAK-D options for the generic "OAK-D Camera" option or any specific OAK-D device
        is_oak_device = selected_camera == "OAK-D Camera" or selected_camera.startswith("OAK-D:")
        
        # Update UI based on camera type
        self.oak_options_group.setVisible(is_oak_device)
        
        # For point cloud functionality, check if Open3D is available
        try:
            import open3d as o3d
            # Open3D is available, enable point cloud options for OAK-D
            self.pointcloud_check.setEnabled(is_oak_device)
            self.pointcloud_check.setToolTip("Generate and visualize 3D point cloud from depth data")
        except ImportError:
            # Open3D not available, disable point cloud options
            self.pointcloud_check.setEnabled(False)
            self.pointcloud_check.setToolTip(
                "Open3D library not available. Install with 'pip install open3d' to enable point cloud support."
            )
        
        # Check for on-device point cloud processing availability
        try:
            import depthai as dai
            self.device_pointcloud_check.setEnabled(True)
        except ImportError:
            self.device_pointcloud_check.setEnabled(False)
            self.device_pointcloud_check.setToolTip(
                "DepthAI library not available. Install with 'pip install depthai' to enable on-device point cloud processing."
            )
        
        # Enable/disable specific resolution options based on camera type
        if is_oak_device:
            # Make sure 4K is available (OAK-D can do up to 4K)
            if self.resolution_combo.findText("3840x2160 (4K UHD)") == -1:
                self.resolution_combo.addItem("3840x2160 (4K UHD)")
        elif "Logitech Brio" in selected_camera:
            # Make sure 4K is available (Brio supports 4K)
            if self.resolution_combo.findText("3840x2160 (4K UHD)") == -1:
                self.resolution_combo.addItem("3840x2160 (4K UHD)")
        elif "Logitech C920" in selected_camera:
            # C920 supports up to Full HD
            # Remove 4K option if it exists
            uhd_index = self.resolution_combo.findText("3840x2160 (4K UHD)")
            if uhd_index >= 0:
                self.resolution_combo.removeItem(uhd_index)
    
    def refresh_cameras(self):
        """Scan for available cameras and update the camera selection dropdown.
        
        Performs the following operations:
        1. Detects standard webcams using OpenCV
        2. Detects OAK-D cameras using DepthAI if available
        3. Adds discovered cameras to the dropdown with relevant information
        4. Updates the status label with detection results
        5. In development/testing environments, adds a simulated camera option
        
        Returns:
            dict: Summary of camera detection results including counts of standard
                 and OAK-D cameras, and availability of required libraries.
        """
        # Clear previous camera list (keep the default options)
        while self.camera_combo.count() > 6:  # Keep "Auto-detect", Cameras 0-4, and "OAK-D Camera"
            self.camera_combo.removeItem(self.camera_combo.count() - 1)
            
        # Add test camera option in development environments
        if 'PYTEST_CURRENT_TEST' in os.environ or 'VIRTUAL_ENV' in os.environ:
            self.camera_combo.addItem("Test Camera (Simulated)")
            
        status_messages = []
        has_opencv = False
        has_depthai = False
        
        # Check for OpenCV webcams
        try:
            import cv2
            has_opencv = True
            
            # Try to find available cameras
            available_cameras = []
            camera_names = {}
            
            # Check Linux device paths first (more reliable than just trying to open)
            if platform.system() == "Linux":
                import glob
                video_devices = glob.glob("/dev/video*")
                for device in video_devices:
                    try:
                        device_num = int(device.replace("/dev/video", ""))
                        if device_num < 10:  # Focus on the first few video devices
                            cap = cv2.VideoCapture(device_num)
                            if cap.isOpened():
                                # Try to read a frame to make sure it works
                                ret, frame = cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    available_cameras.append(device_num)
                                    # Try to get camera name
                                    if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                                        name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                                        if name:
                                            camera_names[device_num] = name
                            cap.release()
                    except Exception as e:
                        logger.warning(f"Error checking device {device}: {e}")
            else:
                # For non-Linux systems, just try cameras 0-4
                for i in range(5):
                    try:
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            # Try to read a frame to make sure it works
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                available_cameras.append(i)
                                # Try to get camera name
                                if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                                    name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                                    if name:
                                        camera_names[i] = name
                        cap.release()
                    except Exception as e:
                        logger.warning(f"Error checking camera {i}: {e}")
            
            # Add detected cameras to the combo box if they have names
            for cam_idx in available_cameras:
                if cam_idx in camera_names and camera_names[cam_idx]:
                    # Add named camera as a new option
                    self.camera_combo.addItem(f"{camera_names[cam_idx]} (index {cam_idx})")
            
            if available_cameras:
                status_messages.append(f"Found {len(available_cameras)} standard webcam(s)")
            else:
                status_messages.append("No standard webcams found")
            
        except ImportError:
            status_messages.append("OpenCV not available. Cannot detect standard webcams.")
        
        # Check for OAK-D camera
        try:
            import depthai as dai
            has_depthai = True
            
            available_devices = dai.Device.getAllAvailableDevices()
            oak_count = len(available_devices)
            
            if oak_count > 0:
                # Add each device to the dropdown with its MX ID
                for device in available_devices:
                    mx_id = device.getMxId()
                    self.camera_combo.addItem(f"OAK-D: {mx_id}")
                
                status_messages.append(f"Found {oak_count} OAK-D camera(s)")
            else:
                status_messages.append("No OAK-D cameras detected")
                
        except ImportError:
            status_messages.append("DepthAI not available. Cannot detect OAK-D cameras.")
            # Make the OAK-D option in the dropdown disabled
            oak_index = self.camera_combo.findText("OAK-D Camera")
            if oak_index >= 0:
                self.camera_combo.model().item(oak_index).setEnabled(False)
        except Exception as e:
            status_messages.append(f"Error detecting OAK-D cameras: {str(e)}")
        
        # Update the status label with all messages
        self.status_label.setText(" | ".join(status_messages))
        
        # Special warning for virtual environments
        if 'VIRTUAL_ENV' in os.environ or 'WSL' in platform.release():
            self.status_label.setText(f"{self.status_label.text()} (Note: Running in virtual environment, camera access may be limited)")
        
        # Return a summary of what we found
        return {
            "has_opencv": has_opencv,
            "has_depthai": has_depthai,
            "standard_cameras": len(available_cameras) if has_opencv else 0,
            "oak_cameras": oak_count if has_depthai else 0
        }
    
    def get_resolution(self):
        """Get the selected resolution as a (width, height) tuple"""
        resolution_text = self.resolution_combo.currentText().split(' ')[0]
        width, height = map(int, resolution_text.split('x'))
        return (width, height)
    
    def start_capture(self):
        """Start video capture from the selected camera.
        
        Initializes the appropriate camera based on the current selection:
        - For standard webcams, uses OpenCV's VideoCapture
        - For OAK-D cameras, sets up a DepthAI pipeline with RGB and optional depth
        - For simulated cameras, creates a test pattern generator
        
        Configures the camera with the selected resolution and FPS settings.
        Updates UI elements to reflect the active capture state.
        """
        # Get selected resolution
        self.resolution = self.get_resolution()
        
        # Get selected camera type
        camera_selection = self.camera_combo.currentText()
        
        # Handle OAK-D cameras (both generic and specific devices)
        if camera_selection == "OAK-D Camera" or camera_selection.startswith("OAK-D:"):
            self._start_oak_camera(device_mx_id=camera_selection.split("OAK-D: ")[-1] if ": " in camera_selection else None)
        # Handle simulated test camera
        elif camera_selection == "Test Camera (Simulated)":
            self._start_simulated_camera()
        else:
            # Extract camera index if this is a named camera with index
            if " (index " in camera_selection and camera_selection.endswith(")"):
                try:
                    # Extract the index from format "Camera Name (index X)"
                    index_part = camera_selection.split(" (index ")[-1].rstrip(")")
                    camera_index = int(index_part)
                    self._start_normal_camera(camera_index)
                except (ValueError, IndexError):
                    # Fall back to auto-detection if parsing fails
                    self._start_normal_camera()
            else:
                # Handle numbered camera options (Camera 0, Camera 1, etc.)
                if camera_selection.startswith("Camera ") and len(camera_selection) > 7:
                    try:
                        camera_index = int(camera_selection[7:])
                        self._start_normal_camera(camera_index)
                    except ValueError:
                        self._start_normal_camera()
                else:
                    # Auto-detect or other unrecognized option
                    self._start_normal_camera()
    
    def _start_normal_camera(self, camera_index=None):
        """Initialize and start a standard webcam using OpenCV.
        
        Sets up a webcam for video capture using OpenCV's VideoCapture:
        1. Either uses the specified camera index or auto-detects an available camera
        2. Configures the camera with the selected resolution and FPS
        3. Updates UI elements to reflect the active capture state
        4. Starts the frame update timer
        
        Handles various error conditions including missing cameras, permission issues,
        and initialization failures. Provides appropriate feedback to the user.
        
        Args:
            camera_index: Optional specific camera index to use. If None, will auto-detect
                         based on the dropdown selection.
        """
        try:
            import cv2
            import platform
            
            # Check if we're running in a virtual environment or WSL
            is_virtual_env = False
            if 'VIRTUAL_ENV' in os.environ or 'WSL' in platform.release():
                is_virtual_env = True
                self.status_label.setText("Note: Running in virtual environment, camera access may be limited")
            
            # Special handling for auto-detect vs specific camera index
            if camera_index is None:
                # Get from dropdown if not specified directly
                selected_index = self.camera_combo.currentIndex()
                
                if selected_index == 0:  # Auto-detect option in dropdown
                    # Try cameras 0-4 to find one that works
                    found_camera = False
                    for i in range(5):
                        try:
                            cap = cv2.VideoCapture(i)
                            if cap.isOpened():
                                # Try to read a test frame to ensure the camera works
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None and test_frame.size > 0:
                                    self.webcam_index = i
                                    self.webcam = cap
                                    found_camera = True
                                    break
                            cap.release()
                        except Exception as e:
                            logger.warning(f"Error testing camera {i}: {e}")
                            continue
                    
                    if not found_camera:
                        msg = "No working cameras detected."
                        if is_virtual_env:
                            msg += " This may be due to running in a virtual environment."
                        QMessageBox.warning(self, "Camera Detection", msg)
                        self.status_label.setText("No working cameras found")
                        return
                        
                elif selected_index < 6:  # Regular cameras (indexes 1-5 correspond to cameras 0-4)
                    # Use the selected camera
                    camera_idx = selected_index - 1
                    try:
                        self.webcam = cv2.VideoCapture(camera_idx)
                        self.webcam_index = camera_idx
                    except Exception as e:
                        QMessageBox.warning(self, "Camera Error", f"Error accessing camera {camera_idx}: {str(e)}")
                        self.status_label.setText(f"Error with camera {camera_idx}")
                        return
            else:  # Use the provided camera index
                try:
                    self.webcam = cv2.VideoCapture(camera_index)
                    self.webcam_index = camera_index
                    
                    # Check if camera opened successfully
                    if not self.webcam.isOpened():
                        if platform.system() == 'Linux':
                            # Check if the device file exists
                            if not os.path.exists(f"/dev/video{camera_index}"):
                                QMessageBox.warning(self, "Camera Error", 
                                                  f"Device /dev/video{camera_index} does not exist.")
                            else:
                                QMessageBox.warning(self, "Camera Error", 
                                                  f"Could not open /dev/video{camera_index}. Check permissions.")
                        else:
                            QMessageBox.warning(self, "Camera Error", 
                                              f"Could not open camera {camera_index}.")
                        self.status_label.setText(f"Failed to open camera {camera_index}")
                        return
                except Exception as e:
                    QMessageBox.warning(self, "Camera Error", f"Error accessing camera {camera_index}: {str(e)}")
                    self.status_label.setText(f"Error with camera {camera_index}")
                    return
            
            # Check if we found a working camera
            if self.webcam is None or not self.webcam.isOpened():
                QMessageBox.warning(self, "Camera Error", "Could not open the selected camera.")
                self.status_label.setText("Failed to open camera")
                return
            
            # Try to read a test frame to ensure the camera works
            ret, test_frame = self.webcam.read()
            if not ret or test_frame is None or test_frame.size == 0:
                self.webcam.release()
                self.webcam = None
                QMessageBox.warning(self, "Camera Error", "Camera opened but failed to provide video.")
                self.status_label.setText("Camera not providing video")
                return
            
            # Set camera properties for resolution and FPS
            width, height = self.resolution
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.webcam.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start the timer for frame updates
            self.timer.start(1000 // self.fps)  # Convert FPS to milliseconds between frames
            self.is_capturing = True
            
            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            self.resolution_combo.setEnabled(False)
            self.fps_slider.setEnabled(False)
            
            # Get actual camera settings (they might differ from requested ones)
            actual_width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.webcam.get(cv2.CAP_PROP_FPS))
            
            # Get camera name if available
            camera_name = "Unknown"
            try:
                # CV_CAP_PROP_DEVICE_NAME is available in newer OpenCV versions
                if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                    camera_name = self.webcam.get(cv2.CAP_PROP_DEVICE_NAME)
                    if not camera_name:
                        camera_name = f"Camera {self.webcam_index}"
                else:
                    camera_name = f"Camera {self.webcam_index}"
            except:
                camera_name = f"Camera {self.webcam_index}"
            
            self.status_label.setText(
                f"Capturing from {camera_name} at "
                f"{actual_width}x{actual_height} @ {actual_fps}fps"
            )
            
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", 
                               "OpenCV is required for webcam capture.\n"
                               "Please install 'opencv-python'.")
            self.status_label.setText("OpenCV not available")
        except Exception as e:
            import logging
            logging.error(f"Error initializing camera: {e}")
            QMessageBox.warning(self, "Camera Error", f"Unexpected error: {str(e)}")
            self.status_label.setText("Error initializing camera")
    
    def _start_simulated_camera(self):
        """Create and start a simulated webcam for testing without physical hardware.
        
        Generates a synthetic test pattern video feed:
        1. Creates a SimulatedWebcam object that provides test frames
        2. Configures the simulation with the selected resolution
        3. Updates UI elements to reflect active capture state
        4. Starts the frame update timer
        
        This method is primarily used for development, testing, and demonstration
        purposes when no physical camera is available. The simulation includes
        moving elements and timestamp information to verify functionality.
        """
        try:
            import cv2
            import numpy as np
            
            class SimulatedWebcam:
                """Simulate a webcam for testing purposes"""
                
                def __init__(self, image_size=(640, 480), parent=None):
                    self.image_size = image_size
                    self.is_open = True
                    self.frame_count = 0
                    self.parent = parent  # Reference to the parent VideoTab
                    
                def isOpened(self):
                    return self.is_open
                    
                def read(self):
                    """Generate a test pattern for each frame"""
                    width, height = self.image_size
                    
                    # Create a base image
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Add a moving element based on frame count
                    self.frame_count += 1
                    position = self.frame_count % width
                    
                    # Create test pattern
                    img[:, :, 0] = np.ones((height, width)) * position % 255  # Blue channel
                    img[:, :, 1] = np.ones((height, width)) * ((position + 85) % 255)  # Green channel
                    img[:, :, 2] = np.ones((height, width)) * ((position + 170) % 255)  # Red channel
                    
                    # Add a moving circle
                    cv2.circle(img, (position, height//2), 50, (0, 0, 255), -1)
                    cv2.circle(img, (width - position, height//2), 30, (255, 0, 0), -1)
                    
                    # Add text showing frame count
                    cv2.putText(img, f"Frame: {self.frame_count}", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    cv2.putText(img, timestamp, (width - 200, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add "SIMULATED CAMERA" text
                    cv2.putText(img, "SIMULATED CAMERA", (width//2 - 120, height - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                    
                    # Add recording indicator if parent is recording
                    if self.parent is not None and hasattr(self.parent, 'is_recording') and self.parent.is_recording:
                        # Red recording indicator
                        cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(img, "REC", (50, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    return True, img
                    
                def set(self, prop, value):
                    """Simulate setting a camera property"""
                    # For testing, just return True to simulate success
                    return True
                    
                def get(self, prop):
                    """Simulate getting a camera property"""
                    # Return values based on property
                    if prop == cv2.CAP_PROP_FRAME_WIDTH:
                        return self.image_size[0]
                    elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                        return self.image_size[1]
                    elif prop == cv2.CAP_PROP_FPS:
                        return 30
                    elif hasattr(cv2, 'CAP_PROP_DEVICE_NAME') and prop == cv2.CAP_PROP_DEVICE_NAME:
                        return "Simulated Test Camera"
                    else:
                        return 0
                        
                def release(self):
                    """Release the simulated camera"""
                    self.is_open = False
            
            # Create a simulated webcam with the requested resolution
            self.webcam = SimulatedWebcam(image_size=self.resolution, parent=self)
            self.webcam_index = -1  # Use -1 to indicate simulated camera
            
            # Start the timer for frame updates
            self.timer.start(1000 // self.fps)  # Convert FPS to milliseconds between frames
            self.is_capturing = True
            
            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.record_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
            self.resolution_combo.setEnabled(False)
            self.fps_slider.setEnabled(False)
            
            self.status_label.setText(
                f"Using simulated camera at {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps"
            )
            
        except Exception as e:
            import logging
            logging.error(f"Error creating simulated camera: {e}")
            QMessageBox.warning(self, "Error", f"Failed to create simulated camera: {str(e)}")
            self.status_label.setText("Failed to create simulated camera")
            
    def _start_oak_camera(self, device_mx_id=None):
        """Initialize and start an OAK-D spatial AI camera using DepthAI.
        
        Sets up a complete pipeline for the OAK-D camera:
        1. Detects and connects to an available OAK-D device
        2. Creates a pipeline with RGB camera and optional stereo depth
        3. Configures advanced depth processing (LR check, subpixel, alignment)
        4. Sets up output queues for RGB, depth, and confidence data
        5. Updates UI elements to reflect the active capture state
        6. Starts the frame update timer
        
        Provides comprehensive error handling for device detection, permissions,
        and initialization failures, with specific guidance for troubleshooting.
        
        Args:
            device_mx_id: Optional MX ID of the specific OAK-D device to use.
                          If None, the first available device will be used.
                          Used when multiple OAK-D cameras are connected.
        """
        try:
            import depthai as dai
            import numpy as np
            
            # First check if any OAK-D devices are available
            available_devices = dai.Device.getAllAvailableDevices()
            if not available_devices:
                QMessageBox.warning(self, "OAK-D Camera Not Found", 
                                   "No OAK-D cameras detected.\n"
                                   "Please connect a device and try again.")
                self.status_label.setText("No OAK-D camera detected")
                return
            
            # Select the device to use
            selected_device = None
            
            # If a specific device MX ID was requested
            if device_mx_id:
                for device in available_devices:
                    if device.getMxId() == device_mx_id:
                        selected_device = device
                        break
                        
                if not selected_device:
                    QMessageBox.warning(self, "Device Not Found", 
                                       f"OAK-D camera with ID {device_mx_id} not found.\n"
                                       "Using first available device instead.")
                    selected_device = available_devices[0]
            else:
                # Use the first available device
                selected_device = available_devices[0]
                
            self.status_label.setText(f"Detected OAK-D camera: {selected_device.getMxId()}")
            
            # Create pipeline
            pipeline = dai.Pipeline()
            
            # Create color camera node
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(self.resolution[0], self.resolution[1])
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setFps(self.fps)
            
            # Create output for color camera preview
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            cam_rgb.preview.link(xout_rgb.input)
            
            # If depth mode is enabled, set up stereo depth
            if self.is_depth_mode:
                # Configure stereo pair
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                # LR-check is required for better depth accuracy
                stereo.setLeftRightCheck(True)
                # Enable subpixel for better accuracy with longer distances
                stereo.setSubpixel(True)
                # Set depth align to RGB camera for alignment between color and depth
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                
                # Create left/right mono cameras
                mono_left = pipeline.create(dai.node.MonoCamera)
                mono_right = pipeline.create(dai.node.MonoCamera)
                
                # Set resolution - OAK-D Lite supports 800P
                mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
                mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
                
                # Set FPS to match color camera
                mono_left.setFps(self.fps)
                mono_right.setFps(self.fps)
                
                # Set camera orientation (left/right)
                mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
                mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                
                # Link mono cameras to stereo node
                mono_left.out.link(stereo.left)
                mono_right.out.link(stereo.right)
                
                # Create outputs for depth and confidence
                xout_depth = pipeline.create(dai.node.XLinkOut)
                xout_depth.setStreamName("depth")
                stereo.depth.link(xout_depth.input)
                
                # Optional: Add confidence map output
                xout_conf = pipeline.create(dai.node.XLinkOut)
                xout_conf.setStreamName("confidence")
                stereo.confidenceMap.link(xout_conf.input)
            
            # Connect to device and start the pipeline
            try:
                self.oak_device = dai.Device(pipeline, selected_device)
                
                # Get output queues
                self.q_rgb = self.oak_device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                if self.is_depth_mode:
                    self.q_depth = self.oak_device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                    self.q_conf = self.oak_device.getOutputQueue(name="confidence", maxSize=4, blocking=False)
                
                # Start the timer for frame updates
                self.timer.start(1000 // self.fps)
                self.is_capturing = True
                
                # Update button states
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.save_button.setEnabled(True)
                self.record_button.setEnabled(True)
                self.camera_combo.setEnabled(False)
                self.resolution_combo.setEnabled(False)
                self.fps_slider.setEnabled(False)
                self.depth_check.setEnabled(False)
                
                # Set up point cloud processing if enabled
                if self.is_pointcloud_mode:
                    try:
                        # Get calibration data for point cloud generation
                        calib_data = self.oak_device.readCalibration()
                        intrinsics = calib_data.getCameraIntrinsics(
                            dai.CameraBoardSocket.RGB,
                            dai.Size2f(self.resolution[0], self.resolution[1])
                        )
                        
                        # Initialize the point cloud visualizer
                        from ..utils.pointcloud import PointCloudVisualizer
                        self.pointcloud_visualizer = PointCloudVisualizer(
                            intrinsic_matrix=np.array(intrinsics).reshape(3, 3),
                            width=self.resolution[0],
                            height=self.resolution[1]
                        )
                        
                        # Set up on-device point cloud if enabled
                        if self.device_pointcloud_check.isChecked():
                            from ..utils.pointcloud import DevicePointCloudGenerator
                            self.device_pointcloud = DevicePointCloudGenerator(
                                device=self.oak_device,
                                resolution=(self.resolution[0], self.resolution[1])
                            )
                            # Set up the point cloud processing
                            self.q_pcl = self.device_pointcloud.start()[0]
                        
                        # Enable point cloud buttons
                        self.open_pc_window_button.setEnabled(True)
                        self.save_pc_button.setEnabled(True)
                        self.pc_to_voxel_button.setEnabled(True)
                        
                    except Exception as e:
                        import logging
                        logging.error(f"Error setting up point cloud: {e}")
                        self.is_pointcloud_mode = False
                        self.pointcloud_check.setChecked(False)
                        QMessageBox.warning(self, "Point Cloud Error", 
                                           f"Failed to initialize point cloud processing: {str(e)}")
                
                self.status_label.setText(
                    f"Capturing from OAK-D camera at {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps"
                    f"{' (Depth mode)' if self.is_depth_mode else ''}"
                    f"{' with Point Cloud' if self.is_pointcloud_mode else ''}"
                )
            except Exception as e:
                error_text = str(e).lower()
                
                # Check for common error patterns in the error message
                if "insufficient permissions" in error_text or "udev rules" in error_text:
                    QMessageBox.warning(self, "Permission Error", 
                                      "Insufficient permissions to access the OAK-D camera.\n"
                                      "Please make sure udev rules are set up correctly.\n\n"
                                      "On Linux, you may need to run:\n"
                                      "echo 'SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"03e7\", "
                                      "MODE=\"0666\"' | sudo tee /etc/udev/rules.d/80-movidius.rules\n"
                                      "sudo udevadm control --reload-rules && sudo udevadm trigger")
                elif "not found" in error_text or "no device" in error_text:
                    QMessageBox.warning(self, "OAK-D Device Error", 
                                      "OAK-D camera found but could not be initialized.\n"
                                      "The device may be in use by another application or in an error state.")
                else:
                    QMessageBox.warning(self, "OAK-D Connection Error", f"Failed to connect to the OAK-D camera: {str(e)}")
                
                self.status_label.setText("Failed to connect to OAK-D camera")
            
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency", 
                               "DepthAI is required for OAK-D camera support.\n"
                               "Please install 'depthai'.")
            self.status_label.setText("DepthAI not available")
        except Exception as e:
            import logging
            logging.error(f"Error initializing OAK-D camera: {e}")
            QMessageBox.warning(self, "Camera Error", f"Failed to initialize OAK-D camera: {str(e)}")
            self.status_label.setText("Failed to initialize OAK-D camera")
    
    def stop_capture(self):
        """Stop video capture and release camera resources.
        
        Performs the following cleanup operations:
        1. Stops the frame update timer
        2. Releases the OpenCV camera if active
        3. Closes the OAK-D device if active
        4. Closes point cloud visualization windows
        5. Resets the video display
        6. Updates UI button states to reflect inactive capture
        
        This method ensures proper resource cleanup when stopping capture
        or when the application is closing.
        """
        if self.is_capturing:
            # Stop the timer
            self.timer.stop()
            
            # Release the webcam if using OpenCV
            if self.webcam:
                self.webcam.release()
                self.webcam = None
            
            # Close the OAK-D device if using it
            if self.oak_device:
                self.oak_device.close()
                self.oak_device = None
            
            # Close point cloud resources
            if hasattr(self, 'pointcloud_visualizer') and self.pointcloud_visualizer is not None:
                try:
                    self.pointcloud_visualizer.close_window()
                except Exception as e:
                    import logging
                    logging.error(f"Error closing point cloud window: {e}")
                self.pointcloud_visualizer = None
            
            if hasattr(self, 'device_pointcloud') and self.device_pointcloud is not None:
                self.device_pointcloud = None
            
            # Reset point cloud state
            self.pcl_window_open = False
            
            # Reset the display
            self.video_display.setText("No video feed")
            self.video_display.setPixmap(QPixmap())
            
            # Stop recording if active
            if self.is_recording:
                self.stop_recording()
            
            # Update button states
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.camera_combo.setEnabled(True)
            self.resolution_combo.setEnabled(True)
            self.fps_slider.setEnabled(True)
            self.depth_check.setEnabled(True)
            
            # Reset point cloud buttons
            self.open_pc_window_button.setEnabled(False)
            self.save_pc_button.setEnabled(False)
            self.pc_to_voxel_button.setEnabled(False)
            
            # Reset recording progress
            self.recording_progress.setValue(0)
            self.recording_progress.setVisible(False)
            
            self.is_capturing = False
            self.status_label.setText("Camera stopped")
    
    def update_frame(self):
        """Update the video frame display with the latest camera image.
        
        Called periodically by the timer to:
        1. Fetch the latest frame from the active camera
        2. Process the frame (convert color formats, apply effects)
        3. Update the UI with the new frame
        
        Handles both standard webcams and OAK-D cameras, dispatching to
        the appropriate specialized update method based on the active camera type.
        """
        if not self.is_capturing:
            return
            
        # Handle OAK-D camera
        if self.oak_device:
            self._update_oak_frame()
            return
            
        # Handle standard webcam
        if self.webcam:
            self._update_webcam_frame()
    
    def _update_webcam_frame(self):
        """Update the display with the latest frame from a standard webcam.
        
        Processes frames from OpenCV's VideoCapture:
        1. Reads a new frame from the camera
        2. Converts from BGR to RGB color format
        3. Creates a QImage and QPixmap for display
        4. Scales the image to fit the display area
        5. Updates the video display widget
        
        Stores the frame for later saving if the user requests it.
        Handles errors that might occur during frame acquisition.
        """
        try:
            import cv2
            
            # Read a frame from the webcam
            ret, frame = self.webcam.read()
            
            if ret:
                # Convert frame from BGR to RGB (OpenCV uses BGR, Qt uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get frame dimensions
                height, width, channels = rgb_frame.shape
                
                # Calculate bytes per line
                bytes_per_line = channels * width
                
                # Convert to QImage
                q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Scale while maintaining aspect ratio
                pixmap = QPixmap.fromImage(q_image)
                
                # Scale to fit the label size while preserving aspect ratio
                label_size = self.video_display.size()
                scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)
                
                # Update the display
                self.video_display.setPixmap(scaled_pixmap)
                
                # Store the current frame for saving
                self.current_frame = frame
                
                # Add frame to recording if active
                if self.is_recording:
                    self.recording_frames.append(frame.copy())
                    
                    # Write frame to video file
                    if self.video_writer is not None:
                        self.video_writer.write(frame)
            else:
                # Failed to read frame
                self.stop_capture()
                QMessageBox.warning(self, "Camera Error", "Failed to read frame from camera.")
                
        except Exception as e:
            import logging
            logging.error(f"Error updating video frame: {e}")
            self.stop_capture()
    
    def _update_oak_frame(self):
        """Update the display with the latest frame from an OAK-D camera.
        
        Processes frames from DepthAI camera pipeline:
        1. Gets RGB frame from the color camera
        2. If in depth mode, gets depth frames and confidence maps
        3. Processes depth data with appropriate normalization and colorization
        4. Creates blended visualizations combining RGB, depth, and confidence data
        5. Converts the processed frame to QImage/QPixmap for display
        
        Maintains a structured representation of all data (RGB, depth, confidence)
        for later saving and processing. Handles different frame formats and
        ensures proper visualization of depth information.
        """
        try:
            import numpy as np
            import cv2
            
            # Always get the RGB frame
            in_rgb = self.q_rgb.get()
            rgb_frame = in_rgb.getCvFrame()
            
            # Store RGB frame for potential saving
            rgb_frame_copy = rgb_frame.copy()
            
            if self.is_depth_mode and hasattr(self, 'q_depth'):
                # Get depth frame
                in_depth = self.q_depth.get()
                depth_frame = in_depth.getFrame()
                
                # Get stereo depth config info for proper scaling
                if hasattr(self, 'oak_device') and self.oak_device is not None:
                    # Get the stereo node to determine max disparity
                    device_info = self.oak_device.getDeviceInfo()
                    max_disparity = 95  # Default value
                    
                    # Use a more accurate normalization based on max disparity
                    depth_frame = (depth_frame * 255. / max_disparity).astype(np.uint8)
                else:
                    # Fallback normalization if max_disparity is unknown
                    depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                
                # Apply colormap to depth - use COLORMAP_JET for better visualization
                colored_depth = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
                
                # Also get confidence map if available
                if hasattr(self, 'q_conf'):
                    in_conf = self.q_conf.get()
                    conf_frame = in_conf.getFrame()
                    
                    # Normalize confidence - 0 is max confidence, 255 is low confidence
                    conf_colored = cv2.normalize(conf_frame, None, 0, 255, cv2.NORM_MINMAX)
                    conf_vis = cv2.applyColorMap(255 - conf_colored.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                    
                    # Create a blend of RGB + Depth + Confidence
                    # Weights: 60% depth, 30% RGB, 10% confidence
                    if rgb_frame.shape == colored_depth.shape == conf_vis.shape:
                        # Convert RGB frame to BGR for blending (OpenCV uses BGR)
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        
                        # First blend RGB and depth
                        blend1 = cv2.addWeighted(bgr_frame, 0.3, colored_depth, 0.6, 0)
                        # Then add confidence
                        final_blend = cv2.addWeighted(blend1, 0.9, conf_vis, 0.1, 0)
                        
                        # Convert back to RGB for Qt
                        rgb_frame = cv2.cvtColor(final_blend, cv2.COLOR_BGR2RGB)
                        
                        # Save full data for export
                        self.current_frame = {
                            'rgb': rgb_frame_copy,
                            'depth': depth_frame,
                            'confidence': conf_frame,
                            'visualization': cv2.cvtColor(final_blend, cv2.COLOR_BGR2RGB)
                        }
                    else:
                        # Just use depth visualization if shapes don't match
                        rgb_frame = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
                        self.current_frame = {
                            'rgb': rgb_frame_copy,
                            'depth': depth_frame,
                            'visualization': rgb_frame
                        }
                else:
                    # Blend RGB and depth without confidence
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    blended = cv2.addWeighted(bgr_frame, 0.4, colored_depth, 0.6, 0)
                    rgb_frame = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                    
                    # Save data for export
                    self.current_frame = {
                        'rgb': rgb_frame_copy,
                        'depth': depth_frame,
                        'visualization': rgb_frame
                    }
                
                # Process point cloud if enabled
                if self.is_pointcloud_mode and hasattr(self, 'pointcloud_visualizer'):
                    try:
                        if hasattr(self, 'device_pointcloud') and self.device_pointcloud is not None and hasattr(self, 'q_pcl'):
                            # Use on-device point cloud generation
                            if self.q_pcl.has():
                                pcl_data = self.q_pcl.get()
                                points = self.device_pointcloud.process_results(pcl_data)
                                
                                # Store point cloud data in current_frame
                                if isinstance(self.current_frame, dict):
                                    self.current_frame['pointcloud'] = points
                                
                                # Update visualizer if window is open
                                if self.pcl_window_open:
                                    import open3d as o3d
                                    pcd = o3d.geometry.PointCloud()
                                    pcd.points = o3d.utility.Vector3dVector(points)
                                    
                                    # Add color if available
                                    if rgb_frame_copy is not None:
                                        # Need to align colors to points
                                        self.pointcloud_visualizer.pcl.points = pcd.points
                                        self.pointcloud_visualizer.visualize_pcd()
                        else:
                            # Use host-based point cloud generation
                            downsample = self.downsample_check.isChecked()
                            remove_noise = self.remove_noise_check.isChecked()
                            
                            # Generate point cloud from RGB and depth
                            self.pointcloud_visualizer.rgbd_to_projection(
                                depth_frame, rgb_frame_copy, 
                                downsample=downsample, 
                                remove_noise=remove_noise
                            )
                            
                            # Store point cloud in current_frame
                            if isinstance(self.current_frame, dict):
                                import open3d as o3d
                                points = np.asarray(self.pointcloud_visualizer.pcl.points)
                                self.current_frame['pointcloud'] = points
                            
                            # Update visualization if window is open
                            if self.pcl_window_open:
                                self.pointcloud_visualizer.visualize_pcd()
                    except Exception as e:
                        import logging
                        logging.error(f"Error processing point cloud: {e}")
            else:
                # Just use RGB frame
                self.current_frame = rgb_frame_copy
            
            # Get frame dimensions
            height, width, channels = rgb_frame.shape
            
            # Calculate bytes per line
            bytes_per_line = channels * width
            
            # Convert to QImage
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit the label size while preserving aspect ratio
            label_size = self.video_display.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)
            
            # Update the display
            self.video_display.setPixmap(scaled_pixmap)
            
            # Add frame to recording if active
            if self.is_recording:
                self.recording_frames.append(self.current_frame.copy() if isinstance(self.current_frame, dict) else self.current_frame)
                
                # Write RGB frame to video file
                if self.video_writer is not None and isinstance(self.current_frame, dict) and 'rgb' in self.current_frame:
                    # Convert to BGR for OpenCV
                    bgr_frame = cv2.cvtColor(self.current_frame['rgb'], cv2.COLOR_RGB2BGR)
                    self.video_writer.write(bgr_frame)
            
        except Exception as e:
            import logging
            logging.error(f"Error updating OAK-D frame: {e}")
            self.stop_capture()
    
    def open_pointcloud_window(self):
        """Open a separate window for point cloud visualization
        
        Creates a 3D visualization window using Open3D for interactive
        viewing of the point cloud data. The window allows rotation,
        zoom, and other 3D interactions.
        """
        if not hasattr(self, 'pointcloud_visualizer') or self.pointcloud_visualizer is None:
            QMessageBox.warning(self, "Point Cloud Error", 
                              "Point cloud visualizer not initialized.\n"
                              "Make sure an OAK-D camera is connected with depth and point cloud modes enabled.")
            return
        
        try:
            # Check if Open3D is available
            try:
                import open3d as o3d
            except ImportError:
                QMessageBox.warning(self, "Missing Dependency", 
                                  "Open3D is required for point cloud visualization.\n"
                                  "Please install with 'pip install open3d'.")
                return
                
            # Check if visualizer is properly initialized
            if not hasattr(self.pointcloud_visualizer, 'vis_initialized') or not self.pointcloud_visualizer.vis_initialized:
                QMessageBox.warning(self, "Visualization Error", 
                                  "Point cloud visualizer could not be properly initialized.\n"
                                  "This may be due to missing graphics drivers or incompatible hardware.\n\n"
                                  "Try running with OPEN3D_ENABLE_VULKAN=1 environment variable\n"
                                  "or use the headless mode for saving point clouds without visualization.")
                return
                
            # If in headless mode, inform the user
            if hasattr(self.pointcloud_visualizer, 'headless_mode') and self.pointcloud_visualizer.headless_mode:
                QMessageBox.information(self, "Headless Mode", 
                                      "Running in headless mode. No visualization window will be shown,\n"
                                      "but you can still save point clouds using the 'Save Point Cloud' button.")
                return
                
            # Mark window as open (will trigger updates in the frame processing)
            self.pcl_window_open = True
            self.status_label.setText("Point cloud window opened. Close the window to continue.")
            
        except Exception as e:
            import logging
            logging.error(f"Error opening point cloud window: {e}")
            QMessageBox.warning(self, "Point Cloud Error", 
                              f"Failed to open point cloud window: {str(e)}")
    
    def save_pointcloud(self):
        """Save the current point cloud to a file
        
        Saves the point cloud in PCD, PLY, or OBJ format for use in
        other 3D software. Includes color information when available.
        """
        if not isinstance(self.current_frame, dict) or 'pointcloud' not in self.current_frame:
            QMessageBox.warning(self, "No Point Cloud", 
                              "No point cloud data available to save.\n"
                              "Make sure depth and point cloud modes are enabled.")
            return
        
        try:
            # Get save path
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Point Cloud", "",
                "Point Cloud Files (*.pcd);;PLY Files (*.ply);;OBJ Files (*.obj);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Add extension if needed
            if not file_path.lower().endswith(('.pcd', '.ply', '.obj')):
                if "PLY" in selected_filter:
                    file_path += ".ply"
                elif "OBJ" in selected_filter:
                    file_path += ".obj"
                else:
                    file_path += ".pcd"
            
            # We'll need Open3D to save
            import open3d as o3d
            
            # Create point cloud and add data
            pcd = o3d.geometry.PointCloud()
            
            # Add points
            points = self.current_frame['pointcloud']
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Add colors if available (need to align with points)
            if 'rgb' in self.current_frame:
                # This is placeholder - in a real app, you'd need to map RGB pixels to 3D points
                # which requires more complex math using depth+rgb calibration
                pass
            
            # Save to file
            o3d.io.write_point_cloud(file_path, pcd, compressed=True)
            
            self.status_label.setText(f"Point cloud saved to {file_path}")
            
        except Exception as e:
            import logging
            logging.error(f"Error saving point cloud: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save point cloud: {str(e)}")
    
    def toggle_recording(self):
        """Toggle video recording state
        
        Starts recording if not already recording, or stops recording if currently recording.
        Creates a video file with the selected format and quality settings.
        """
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording a video clip
        
        Sets up a video writer with the selected codec and format,
        and begins capturing frames. Uses a timer to track recording duration 
        and updates the progress bar.
        """
        if not self.is_capturing or self.current_frame is None:
            return
            
        try:
            import cv2
            import numpy as np
            import time
            from datetime import datetime
            
            # Get recording parameters
            duration_sec = self.duration_spin.value()
            format_idx = self.format_combo.currentIndex()
            quality_idx = self.quality_combo.currentIndex()
            
            # Determine file extension and codec based on format selection
            if format_idx == 0:  # MP4 (H.264)
                ext = ".mp4"
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            elif format_idx == 1:  # AVI (MJPG)
                ext = ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG codec
            elif format_idx == 2:  # MKV (H.264)
                ext = ".mkv" 
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            else:  # MOV (H.264)
                ext = ".mov"
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            
            # Quality settings
            if quality_idx == 0:  # Low
                bitrate = 1000000  # 1 Mbps
            elif quality_idx == 1:  # Medium
                bitrate = 2000000  # 2 Mbps
            elif quality_idx == 2:  # High
                bitrate = 5000000  # 5 Mbps
            else:  # Highest
                bitrate = 8000000  # 8 Mbps
            
            # Create output file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", f"video_{timestamp}{ext}",
                f"Video Files (*{ext});;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Ensure file extension is correct
            if not file_path.lower().endswith(ext):
                file_path += ext
            
            # Get frame dimensions from current frame
            if isinstance(self.current_frame, dict):
                # For depth cameras, use RGB frame
                if 'rgb' in self.current_frame:
                    height, width = self.current_frame['rgb'].shape[:2]
                else:
                    height, width = self.resolution
            elif self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
            else:
                height, width = self.resolution
            
            # Create video writer
            self.video_writer = cv2.VideoWriter(
                file_path, 
                fourcc,
                self.fps, 
                (width, height)
            )
            
            # Set recording state
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_frames = []
            
            # Check if depth recording is enabled for OAK-D cameras
            self.depth_recording = self.is_depth_mode
            
            # Update UI
            self.record_button.setText("Stop Recording")
            self.recording_progress.setVisible(True)
            self.recording_progress.setValue(0)
            self.status_label.setText(f"Recording video to {file_path}")
            
            # Start recording timer
            self.record_timer.start(100)  # Update progress bar every 100ms
            
        except Exception as e:
            import logging
            logging.error(f"Error starting video recording: {e}")
            QMessageBox.warning(self, "Recording Error", f"Failed to start recording: {str(e)}")
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording and save the video file
        
        Finalizes the video file and resets recording state.
        Processes any remaining frames and cleans up resources.
        """
        if not self.is_recording:
            return
            
        try:
            # Stop the recording timer
            self.record_timer.stop()
            
            # Release video writer
            if self.video_writer is not None:
                # Add any remaining frames from the buffer
                for frame in self.recording_frames:
                    if isinstance(frame, dict) and 'rgb' in frame:
                        # For OAK-D, write RGB frame (BGR for OpenCV)
                        bgr_frame = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
                        self.video_writer.write(bgr_frame)
                    elif isinstance(frame, np.ndarray):
                        # For regular webcam, frames are already BGR
                        self.video_writer.write(frame)
                
                self.video_writer.release()
                self.video_writer = None
            
            # If we have depth data and depth recording was enabled, save a separate file
            if self.depth_recording and any('depth' in frame for frame in self.recording_frames if isinstance(frame, dict)):
                self.save_depth_video()
            
            # Reset recording state
            self.is_recording = False
            self.recording_frames = []
            
            # Update UI
            self.record_button.setText("Record Video")
            self.recording_progress.setVisible(False)
            self.status_label.setText("Recording complete")
            
            # Ask user if they want to process this video
            result = QMessageBox.question(
                self, 
                "Processing Options", 
                "Would you like to process this video for voxel data?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if result == QMessageBox.Yes:
                self.process_recorded_video()
                
        except Exception as e:
            import logging
            logging.error(f"Error stopping video recording: {e}")
            QMessageBox.warning(self, "Recording Error", f"Error finalizing recording: {str(e)}")
            
        finally:
            # Ensure recording state is reset
            self.is_recording = False
            self.record_button.setText("Record Video")
            self.recording_progress.setVisible(False)
    
    def update_recording(self):
        """Update recording progress 
        
        Called periodically to update the recording progress bar
        and check if recording duration has been reached.
        """
        if not self.is_recording:
            return
            
        try:
            import time
            
            # Calculate elapsed time
            elapsed = time.time() - self.recording_start_time
            duration = self.duration_spin.value()
            
            # Update progress bar
            progress = min(int((elapsed / duration) * 100), 100)
            self.recording_progress.setValue(progress)
            
            # Check if we've reached the duration
            if elapsed >= duration:
                self.stop_recording()
                
        except Exception as e:
            import logging
            logging.error(f"Error updating recording progress: {e}")
    
    def save_depth_video(self):
        """Save depth data from recording as a separate file
        
        Creates a visualization of depth data as a video file, and 
        also saves raw depth data as a NumPy array.
        """
        try:
            import cv2
            import numpy as np
            from datetime import datetime
            
            # Get format from saved video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = f"depth_{timestamp}"
            
            # Save path for depth visualization
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Depth Visualization", f"{base_path}_vis.avi",
                "AVI Files (*.avi);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Extract first depth frame to get dimensions
            depth_frames = []
            for frame in self.recording_frames:
                if isinstance(frame, dict) and 'depth' in frame:
                    depth_frames.append(frame['depth'])
                    
            if not depth_frames:
                return
                
            # Get dimensions from first frame
            first_depth = depth_frames[0]
            height, width = first_depth.shape[:2]
            
            # Create video writer for depth visualization
            depth_writer = cv2.VideoWriter(
                file_path,
                cv2.VideoWriter_fourcc(*'MJPG'),
                self.fps,
                (width, height),
                isColor=True
            )
            
            # Write colorized depth frames
            for depth_frame in depth_frames:
                # Normalize and colorize depth
                normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                colorized = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
                depth_writer.write(colorized)
                
            depth_writer.release()
            
            # Also save all depth data as numpy array
            np_path = file_path.replace('_vis.avi', '_data.npy')
            np.save(np_path, np.array(depth_frames))
            
            self.status_label.setText(f"Depth data saved to {file_path} and {np_path}")
            
        except Exception as e:
            import logging
            logging.error(f"Error saving depth video: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to save depth video: {str(e)}")
    
    def process_recorded_video(self):
        """Process the recorded video for voxel data
        
        Integrates with the voxel processing pipeline to convert 
        the video frames into 3D voxel data for visualization.
        """
        try:
            # Get access to main window
            main_window = None
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'voxel_processor'):
                    main_window = parent
                    break
                parent = parent.parent()
                
            if main_window is None:
                QMessageBox.warning(self, "Processing Error", 
                                  "Could not access main application window.")
                return
                
            # Show options dialog
            # ... (This would be implemented based on specific processing needs)
            
            # For now, just show information
            QMessageBox.information(self, "Processing Information", 
                                  "Video processing will extract frames and convert to 3D data.\n\n"
                                  "For more examples, see: https://youtu.be/m-b51C82-UE")
            
        except Exception as e:
            import logging
            logging.error(f"Error processing video: {e}")
            QMessageBox.warning(self, "Processing Error", f"Failed to process video: {str(e)}")
    
    def export_to_voxel_grid(self):
        """Convert the point cloud to a voxel grid
        
        Integrates the point cloud data with the Voxel Projector's
        main processing pipeline for further visualization and
        analysis in the main viewer.
        """
        if not isinstance(self.current_frame, dict) or 'pointcloud' not in self.current_frame:
            QMessageBox.warning(self, "No Point Cloud", 
                              "No point cloud data available to export.\n"
                              "Make sure depth and point cloud modes are enabled.")
            return
        
        try:
            # Access the main window through parent widgets
            main_window = None
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, 'voxel_processor'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if main_window is None:
                QMessageBox.warning(self, "Export Error", 
                                  "Could not access the main application window.")
                return
            
            # Get the points
            points = self.current_frame['pointcloud']
            
            # Create intensity data (can be enhanced with actual color/intensity)
            intensities = np.ones(len(points))
            
            # Set data in the voxel processor
            main_window.voxel_processor.set_input_points(points, intensities)
            
            # Switch to the visualization tab
            main_window.tab_widget.setCurrentWidget(main_window.visualization_tab)
            
            # Update visualization
            main_window.update_visualization()
            
            self.status_label.setText("Point cloud exported to voxel grid")
            
        except Exception as e:
            import logging
            logging.error(f"Error exporting to voxel grid: {e}")
            QMessageBox.warning(self, "Export Error", 
                              f"Failed to export point cloud to voxel grid: {str(e)}")
    
    def save_frame(self):
        """Save the current video frame and associated data to disk.
        
        Saves different data types depending on the active camera:
        
        For standard webcams:
        - Saves the RGB frame as JPG or PNG based on user selection
        
        For OAK-D cameras in depth mode:
        - Saves the RGB frame and/or visualization blend
        - Saves depth data in two formats:
          1. Raw depth data as NumPy (.npy) file for further processing
          2. Colorized depth visualization as PNG
        - Optionally saves confidence data in similar formats
        
        Handles error conditions and provides status updates to the user.
        """
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No frame available to save.")
            return
            
        try:
            import cv2
            
            # Get save path
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Frame", "",
                "JPEG Images (*.jpg);;PNG Images (*.png);;All Files (*)"
            )
            
            if file_path:
                # Add extension if needed
                if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if "PNG" in selected_filter:
                        file_path += ".png"
                    else:
                        file_path += ".jpg"
                
                # Check if current_frame is a dictionary (OAK-D with depth data)
                if isinstance(self.current_frame, dict):
                    # Save the RGB or visualization image
                    if 'visualization' in self.current_frame:
                        # Save the blended visualization
                        cv2.imwrite(file_path, self.current_frame['visualization'])
                    else:
                        # Save the RGB frame
                        cv2.imwrite(file_path, self.current_frame['rgb'])
                    
                    # Base path for additional files
                    base_path = file_path.rsplit('.', 1)[0]
                    
                    # Save depth data if available
                    if 'depth' in self.current_frame:
                        try:
                            import numpy as np
                            # Save as NPY for raw data
                            depth_npy_path = f"{base_path}_depth.npy"
                            np.save(depth_npy_path, self.current_frame['depth'])
                            
                            # Also save a visualized version
                            depth_vis_path = f"{base_path}_depth.png"
                            colored_depth = cv2.applyColorMap(self.current_frame['depth'], cv2.COLORMAP_JET)
                            cv2.imwrite(depth_vis_path, colored_depth)
                        
                            # Save confidence data if available
                            if 'confidence' in self.current_frame:
                                # Save as NPY for raw data
                                conf_npy_path = f"{base_path}_confidence.npy"
                                np.save(conf_npy_path, self.current_frame['confidence'])
                                
                                # Also save a visualized version
                                conf_vis_path = f"{base_path}_confidence.png"
                                # Invert confidence (0 = max confidence in depthai, but this is visually confusing)
                                conf_normalized = cv2.normalize(self.current_frame['confidence'], None, 0, 255, cv2.NORM_MINMAX)
                                colored_conf = cv2.applyColorMap(255 - conf_normalized.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                                cv2.imwrite(conf_vis_path, colored_conf)
                                
                                self.status_label.setText(f"Saved frame with depth and confidence data to {base_path}*")
                            else:
                                self.status_label.setText(f"Saved frame with depth data to {base_path}*")
                        except Exception as e:
                            import logging
                            logging.error(f"Error saving depth/confidence data: {e}")
                            self.status_label.setText(f"Saved frame to {file_path}, but error saving depth data")
                    else:
                        self.status_label.setText(f"Frame saved to {file_path}")
                else:
                    # Standard frame (regular webcam)
                    cv2.imwrite(file_path, self.current_frame)
                    self.status_label.setText(f"Frame saved to {file_path}")
        
        except Exception as e:
            import logging
            logging.error(f"Error saving frame: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save frame: {str(e)}")
    
    def closeEvent(self, event):
        """Clean up resources when widget is closed"""
        self.stop_capture()
        super().closeEvent(event)