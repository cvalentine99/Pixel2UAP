"""
Main application window for Voxel Projector v2
"""

import os
import sys
import logging
import platform
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTabWidget,
    QSplitter, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QProgressBar,
    QCheckBox, QApplication, QDockWidget, QTextEdit,
    QStatusBar, QToolBar
)
from PyQt6.QtCore import Qt, QSettings, QSize, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QAction, QFont

from ..voxel_engine.processor import VoxelProcessor
from ..voxel_engine.viewer import MatplotlibVoxelViewer, PyVistaVoxelViewer
from ..utils.errors import show_error_dialog, ProcessingError, InputError, RenderingError
from . import tabs

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window for the Voxel Projector application"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Initialize application state
        self.settings = QSettings("VoxelProjector", "v2")
        self.voxel_processor = VoxelProcessor()
        
        # Determine which visualization to use
        try:
            from .. import PYVISTA_AVAILABLE, PYVISTAQT_AVAILABLE
            
            if PYVISTAQT_AVAILABLE:
                logger.info("Using PyVista for visualization")
                self.visualization_engine = "pyvista"
                self.voxel_viewer = PyVistaVoxelViewer(use_qt=True)
            else:
                logger.info("Using Matplotlib for visualization")
                self.visualization_engine = "matplotlib"
                self.voxel_viewer = MatplotlibVoxelViewer()
        except ImportError:
            logger.warning("PyVista not available, using Matplotlib for visualization")
            self.visualization_engine = "matplotlib"
            self.voxel_viewer = MatplotlibVoxelViewer()
        
        # Setup UI
        self.setup_ui()
        
        # Restore window state
        self.restore_settings()
        
        # Initial status
        self.statusBar().showMessage("Ready")
    
    def setup_ui(self):
        """Set up the user interface"""
        # Set window properties
        self.setWindowTitle("Voxel Projector v2")
        self.setMinimumSize(1200, 800)
        
        # Create menubar
        self.create_menu()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for tabs and visualization
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        
        # Create tab widget for controls
        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(400)
        
        # Create visualization container
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        
        # Add widgets to splitter
        self.splitter.addWidget(self.tab_widget)
        self.splitter.addWidget(self.viz_container)
        
        # Set initial splitter sizes (30% tabs, 70% visualization)
        self.splitter.setSizes([300, 700])
        
        # Create tabs
        self.create_tabs()
        
        # Create visualization
        self.setup_visualization()
        
        # Create docks
        self.create_docks()
        
        # Create status bar
        self.statusBar().setMinimumHeight(20)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_menu(self):
        """Create the application menu"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Open action
        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        
        # Export voxel data actions
        export_vti_action = QAction("Export Voxel Data as VTI", self)
        export_vti_action.triggered.connect(lambda: self.export_voxel_data("vti"))
        export_menu.addAction(export_vti_action)
        
        export_nrrd_action = QAction("Export Voxel Data as NRRD", self)
        export_nrrd_action.triggered.connect(lambda: self.export_voxel_data("nrrd"))
        export_menu.addAction(export_nrrd_action)
        
        export_nii_action = QAction("Export Voxel Data as NIFTI", self)
        export_nii_action.triggered.connect(lambda: self.export_voxel_data("nii"))
        export_menu.addAction(export_nii_action)
        
        export_menu.addSeparator()
        
        # Export mesh actions
        export_stl_action = QAction("Export Mesh as STL", self)
        export_stl_action.triggered.connect(lambda: self.export_mesh("stl"))
        export_menu.addAction(export_stl_action)
        
        export_obj_action = QAction("Export Mesh as OBJ", self)
        export_obj_action.triggered.connect(lambda: self.export_mesh("obj"))
        export_menu.addAction(export_obj_action)
        
        export_ply_action = QAction("Export Mesh as PLY", self)
        export_ply_action.triggered.connect(lambda: self.export_mesh("ply"))
        export_menu.addAction(export_ply_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        # Toggle docks
        log_dock_action = QAction("Log Panel", self)
        log_dock_action.setCheckable(True)
        log_dock_action.setChecked(True)
        log_dock_action.triggered.connect(lambda checked: self.log_dock.setVisible(checked))
        view_menu.addAction(log_dock_action)
        
        view_menu.addSeparator()
        
        # Reset view action
        reset_view_action = QAction("Reset View", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)
        
        # Take screenshot action
        screenshot_action = QAction("Take Screenshot", self)
        screenshot_action.setShortcut("Ctrl+P")
        screenshot_action.triggered.connect(self.take_screenshot)
        view_menu.addAction(screenshot_action)
        
        # Tools menu
        tools_menu = self.menuBar().addMenu("&Tools")
        
        # Extract mesh action
        extract_mesh_action = QAction("Extract Surface Mesh", self)
        extract_mesh_action.triggered.connect(self.extract_mesh)
        tools_menu.addAction(extract_mesh_action)
        
        # Add measurement widget action
        if self.visualization_engine == "pyvista":
            tools_menu.addSeparator()
            
            add_clip_plane_action = QAction("Add Clipping Plane", self)
            add_clip_plane_action.triggered.connect(self.add_clip_plane)
            tools_menu.addAction(add_clip_plane_action)
            
            add_slice_widget_action = QAction("Add Slice Widget", self)
            add_slice_widget_action.triggered.connect(self.add_slice_widget)
            tools_menu.addAction(add_slice_widget_action)
            
            add_measurement_action = QAction("Add Measurement Tool", self)
            add_measurement_action.triggered.connect(self.add_measurement_widget)
            tools_menu.addAction(add_measurement_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the application toolbar"""
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        
        # Open file button
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        self.toolbar.addAction(open_action)
        
        self.toolbar.addSeparator()
        
        # Add processing actions
        process_action = QAction("Process", self)
        process_action.triggered.connect(self.run_processing)
        self.toolbar.addAction(process_action)
        
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_processor)
        self.toolbar.addAction(reset_action)
        
        self.toolbar.addSeparator()
        
        # Add visualization type selector
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItem("Points")
        self.viz_type_combo.addItem("Volume")
        self.viz_type_combo.addItem("Mesh")
        self.viz_type_combo.currentIndexChanged.connect(self.change_visualization_type)
        self.toolbar.addWidget(QLabel("Display: "))
        self.toolbar.addWidget(self.viz_type_combo)
        
        self.toolbar.addSeparator()
        
        # Add view controls
        if self.visualization_engine == "pyvista":
            reset_view_action = QAction("Reset View", self)
            reset_view_action.triggered.connect(self.reset_view)
            self.toolbar.addAction(reset_view_action)
            
            screenshot_action = QAction("Screenshot", self)
            screenshot_action.triggered.connect(self.take_screenshot)
            self.toolbar.addAction(screenshot_action)
    
    def create_tabs(self):
        """Create the tab widgets"""
        # Input tab
        self.input_tab = tabs.InputTab(self.voxel_processor)
        self.tab_widget.addTab(self.input_tab, "Input")
        
        # Processing tab
        self.processing_tab = tabs.ProcessingTab(self.voxel_processor)
        self.tab_widget.addTab(self.processing_tab, "Processing")
        
        # Visualization tab
        self.visualization_tab = tabs.VisualizationTab(
            self.voxel_viewer, 
            self.visualization_engine
        )
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        
        # Export tab
        self.export_tab = tabs.ExportTab(self.voxel_processor)
        self.tab_widget.addTab(self.export_tab, "Export")
        
        # Video tab
        self.video_tab = tabs.VideoTab()
        self.tab_widget.addTab(self.video_tab, "Webcam")
        
        # Pixel2UAP tab for UAP detection
        try:
            from .pixel_motion_tab import PixelMotionTab
            self.pixel_motion_tab = PixelMotionTab()
            self.tab_widget.addTab(self.pixel_motion_tab, "UAP Detection")
        except ImportError:
            logger.warning("UAP Detection tab not available")
    
    def setup_visualization(self):
        """Set up the visualization area"""
        # Clear layout
        while self.viz_layout.count():
            item = self.viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        
        # Add appropriate visualization widget
        if self.visualization_engine == "pyvista":
            logger.debug("Setting up PyVista visualization")
            # Get the Qt widget from PyVista
            viz_widget = self.voxel_viewer.get_widget()
            if viz_widget:
                self.viz_layout.addWidget(viz_widget)
            else:
                logger.error("PyVista widget not available")
                # Fallback to Matplotlib
                self.voxel_viewer = MatplotlibVoxelViewer()
                self.viz_layout.addWidget(self.voxel_viewer.create_canvas())
        else:
            logger.debug("Setting up Matplotlib visualization")
            # Use Matplotlib canvas
            self.viz_layout.addWidget(self.voxel_viewer.create_canvas())
    
    def create_docks(self):
        """Create dockable widgets"""
        # Log dock
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Monospace", 9))
        self.log_dock.setWidget(self.log_widget)
        
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        
        # Set up log handler to display logs in the dock
        self.log_handler = LogHandler(self.log_widget)
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
    
    def restore_settings(self):
        """Restore application settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # Restore splitter position
        splitter_sizes = self.settings.value("splitterSizes")
        if splitter_sizes:
            self.splitter.setSizes([int(s) for s in splitter_sizes])
        
        # Restore active tab
        active_tab = self.settings.value("activeTab", 0, type=int)
        self.tab_widget.setCurrentIndex(active_tab)
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("splitterSizes", self.splitter.sizes())
        self.settings.setValue("activeTab", self.tab_widget.currentIndex())
    
    def update_recent_files_menu(self):
        """Update the recent files menu"""
        self.recent_files_menu.clear()
        
        recent_files = self.settings.value("recentFiles", [])
        if not recent_files:
            no_recent = QAction("No recent files", self)
            no_recent.setEnabled(False)
            self.recent_files_menu.addAction(no_recent)
            return
        
        for file_path in recent_files:
            action = QAction(os.path.basename(file_path), self)
            action.setData(file_path)
            action.triggered.connect(
                lambda checked, path=file_path: self.load_input(path)
            )
            self.recent_files_menu.addAction(action)
        
        self.recent_files_menu.addSeparator()
        clear_action = QAction("Clear Recent Files", self)
        clear_action.triggered.connect(self.clear_recent_files)
        self.recent_files_menu.addAction(clear_action)
    
    def add_to_recent_files(self, file_path):
        """Add a file to the recent files list"""
        recent_files = self.settings.value("recentFiles", [])
        
        # Ensure it's a list
        if not isinstance(recent_files, list):
            recent_files = []
        
        # Remove if already in list
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to the beginning
        recent_files.insert(0, file_path)
        
        # Limit to 10 recent files
        recent_files = recent_files[:10]
        
        # Save and update
        self.settings.setValue("recentFiles", recent_files)
        self.update_recent_files_menu()
    
    def clear_recent_files(self):
        """Clear the recent files list"""
        self.settings.setValue("recentFiles", [])
        self.update_recent_files_menu()
    
    def open_file(self):
        """Open a file dialog to select input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "All Files (*);;Images (*.png *.jpg *.jpeg *.tif *.tiff);;DICOM (*.dcm);;NIFTI (*.nii *.nii.gz);;Mesh Files (*.stl *.obj *.ply);;Motion Data (*.json)"
        )
        
        if file_path:
            self.load_input(file_path)
    
    def load_input(self, file_path):
        """Load the specified input file"""
        try:
            # Update the input tab
            self.input_tab.set_input_path(file_path)
            
            # Add to recent files
            self.add_to_recent_files(file_path)
            
            # Update status
            self.statusBar().showMessage(f"Loaded file: {file_path}")
            
            # Switch to input tab
            self.tab_widget.setCurrentWidget(self.input_tab)
            
            return True
        except Exception as e:
            show_error_dialog(e, self)
            self.statusBar().showMessage("Error loading file")
            return False
    
    def run_processing(self):
        """Run the voxel processing pipeline"""
        # Get input path from input tab
        input_path = self.input_tab.get_input_path()
        if not input_path:
            QMessageBox.warning(self, "No Input", "Please select an input file or directory first.")
            return
        
        # Get processing method
        method = self.processing_tab.get_processing_method()
        
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.statusBar().showMessage("Processing...")
            
            # Run processing
            self.voxel_processor.process_input(input_path, method)
            
            # Update progress
            self.progress_bar.setValue(70)
            
            # Visualize the result
            self.update_visualization()
            
            # Complete
            self.progress_bar.setValue(100)
            self.statusBar().showMessage("Processing completed")
            
            # Hide progress bar after a delay
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
        except (ProcessingError, InputError) as e:
            self.progress_bar.setVisible(False)
            show_error_dialog(e, self)
            self.statusBar().showMessage("Processing failed")
    
    def reset_processor(self):
        """Reset the voxel processor and visualization"""
        self.voxel_processor.reset_voxel_grid()
        
        # Clear visualization
        if self.visualization_engine == "pyvista":
            self.voxel_viewer.plotter.clear()
            self.voxel_viewer.plotter.update()
        else:
            # Matplotlib
            if hasattr(self.voxel_viewer, 'fig'):
                self.voxel_viewer.fig.clear()
                self.voxel_viewer.canvas.draw()
        
        self.statusBar().showMessage("Reset completed")
    
    def update_visualization(self):
        """Update the visualization based on current data and settings"""
        try:
            # Get the visualization type
            viz_type = self.viz_type_combo.currentText()
            
            if viz_type == "Points":
                # Get point data
                points, intensities = self.voxel_processor.get_voxel_data()
                
                if points is None or len(points) == 0:
                    QMessageBox.warning(self, "No Data", "No points to visualize.")
                    return
                
                # Visualize points
                if self.visualization_engine == "pyvista":
                    self.voxel_viewer.plot_points(points, intensities)
                else:
                    self.voxel_viewer.plot_3d_voxels(points, intensities)
                
            elif viz_type == "Volume":
                if self.visualization_engine == "pyvista":
                    # Direct volume rendering with PyVista
                    self.voxel_viewer.plot_volume(
                        self.voxel_processor.voxel_grid,
                        self.voxel_processor.voxel_grid_extent
                    )
                else:
                    # Fallback to points for Matplotlib
                    points, intensities = self.voxel_processor.get_voxel_data()
                    if points is not None and len(points) > 0:
                        self.voxel_viewer.plot_3d_voxels(points, intensities)
                    else:
                        QMessageBox.warning(self, "No Data", "No volume data to visualize.")
                
            elif viz_type == "Mesh":
                # Extract mesh if not already done
                if not hasattr(self.voxel_processor, 'output_mesh') or self.voxel_processor.output_mesh is None:
                    self.extract_mesh()
                
                # Show mesh if available
                if hasattr(self.voxel_processor, 'output_mesh') and self.voxel_processor.output_mesh is not None:
                    if self.visualization_engine == "pyvista":
                        self.voxel_viewer.plot_mesh(self.voxel_processor.output_mesh)
                    else:
                        # Fallback to points for Matplotlib
                        points, intensities = self.voxel_processor.get_voxel_data()
                        if points is not None and len(points) > 0:
                            self.voxel_viewer.plot_3d_voxels(points, intensities)
                        else:
                            QMessageBox.warning(self, "No Data", "No mesh data to visualize.")
                else:
                    QMessageBox.warning(self, "No Mesh", "No mesh data available. Try extracting a mesh first.")
            
            # Set visualization tab parameters based on what we're showing
            self.visualization_tab.update_for_visualization_type(viz_type)
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            show_error_dialog(e, self)
    
    def change_visualization_type(self, index):
        """Handle change in visualization type"""
        self.update_visualization()
    
    def export_voxel_data(self, format):
        """Export the voxel data in the specified format"""
        # Check if we have data
        if self.voxel_processor.voxel_grid is None or np.all(self.voxel_processor.voxel_grid == 0):
            QMessageBox.warning(self, "No Data", "No voxel data to export.")
            return
        
        # Get export path
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
                self.statusBar().showMessage(f"Exported voxel data to {file_path}")
            else:
                self.statusBar().showMessage("Export failed")
            
        except Exception as e:
            logger.error(f"Error exporting voxel data: {e}")
            show_error_dialog(e, self)
    
    def export_mesh(self, format):
        """Export the mesh in the specified format"""
        # Check if we have a mesh
        if not hasattr(self.voxel_processor, 'output_mesh') or self.voxel_processor.output_mesh is None:
            # Try to extract a mesh first
            try:
                self.extract_mesh()
            except Exception as e:
                logger.error(f"Error extracting mesh for export: {e}")
                show_error_dialog(e, self)
                return
            
            # Check again
            if not hasattr(self.voxel_processor, 'output_mesh') or self.voxel_processor.output_mesh is None:
                QMessageBox.warning(self, "No Mesh", "No mesh data available for export.")
                return
        
        # Get export path
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
                self.statusBar().showMessage(f"Exported mesh to {file_path}")
            else:
                self.statusBar().showMessage("Export failed")
            
        except Exception as e:
            logger.error(f"Error exporting mesh: {e}")
            show_error_dialog(e, self)
    
    def extract_mesh(self):
        """Extract a surface mesh from the voxel data"""
        # Check if we have data
        if self.voxel_processor.voxel_grid is None or np.all(self.voxel_processor.voxel_grid == 0):
            QMessageBox.warning(self, "No Data", "No voxel data to extract mesh from.")
            return
        
        try:
            # Get extraction parameters from the export tab
            method = self.export_tab.get_mesh_extraction_method()
            level = self.export_tab.get_mesh_extraction_level()
            
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.statusBar().showMessage("Extracting mesh...")
            
            # Extract the mesh
            mesh = self.voxel_processor.extract_mesh(method, level)
            
            # Update progress
            self.progress_bar.setValue(90)
            
            if mesh is not None:
                # Switch to mesh visualization
                self.viz_type_combo.setCurrentText("Mesh")
                self.update_visualization()
                
                self.statusBar().showMessage("Mesh extraction completed")
            else:
                self.statusBar().showMessage("Mesh extraction failed")
            
            # Hide progress bar after a delay
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            logger.error(f"Error extracting mesh: {e}")
            show_error_dialog(e, self)
    
    def reset_view(self):
        """Reset the visualization view"""
        if self.visualization_engine == "pyvista":
            try:
                self.voxel_viewer.plotter.reset_camera()
                self.voxel_viewer.plotter.view_isometric()
                self.voxel_viewer.plotter.update()
            except Exception as e:
                logger.error(f"Error resetting view: {e}")
        else:
            # For matplotlib, just redraw
            self.update_visualization()
    
    def take_screenshot(self):
        """Take a screenshot of the current visualization"""
        if self.visualization_engine == "pyvista":
            try:
                # Get save path
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Screenshot", "",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                
                if file_path:
                    # Ensure file has the correct extension
                    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path += ".png"
                    
                    # Take screenshot
                    self.voxel_viewer.screenshot(file_path)
                    self.statusBar().showMessage(f"Screenshot saved to {file_path}")
            except Exception as e:
                logger.error(f"Error taking screenshot: {e}")
                show_error_dialog(e, self)
        else:
            # For matplotlib, save the figure
            try:
                # Get save path
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Screenshot", "",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                
                if file_path:
                    # Ensure file has the correct extension
                    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path += ".png"
                    
                    # Save figure
                    if self.voxel_viewer.save_figure(file_path):
                        self.statusBar().showMessage(f"Screenshot saved to {file_path}")
                    else:
                        self.statusBar().showMessage("Failed to save screenshot")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")
                show_error_dialog(e, self)
    
    def add_clip_plane(self):
        """Add a clipping plane to the visualization"""
        if self.visualization_engine == "pyvista":
            try:
                self.voxel_viewer.add_clip_plane_widget()
                self.statusBar().showMessage("Clip plane added")
            except Exception as e:
                logger.error(f"Error adding clip plane: {e}")
                show_error_dialog(e, self)
    
    def add_slice_widget(self):
        """Add a slice widget to the visualization"""
        if self.visualization_engine == "pyvista":
            try:
                self.voxel_viewer.add_slice_widget()
                self.statusBar().showMessage("Slice widget added")
            except Exception as e:
                logger.error(f"Error adding slice widget: {e}")
                show_error_dialog(e, self)
    
    def add_measurement_widget(self):
        """Add a measurement widget to the visualization"""
        if self.visualization_engine == "pyvista":
            try:
                self.voxel_viewer.add_measurement_widget()
                self.statusBar().showMessage("Measurement widget added")
            except Exception as e:
                logger.error(f"Error adding measurement widget: {e}")
                show_error_dialog(e, self)
    
    def show_about(self):
        """Show the about dialog"""
        from .. import __version__
        
        # Create message
        about_text = (
            f"<h2>Voxel Projector v{__version__}</h2>"
            "<p>A tool for projecting 2D images into 3D voxel representations.</p>"
            f"<p><b>Python:</b> {platform.python_version()}<br>"
            f"<b>OS:</b> {platform.system()} {platform.release()}</p>"
            "<p><b>Visualization:</b> "
        )
        
        if self.visualization_engine == "pyvista":
            import pyvista
            import vtk
            about_text += f"PyVista {pyvista.__version__} (VTK {vtk.vtkVersion().GetVTKVersion()})"
        else:
            import matplotlib
            about_text += f"Matplotlib {matplotlib.__version__}"
        
        about_text += "</p>"
        
        # Credits and licenses
        about_text += (
            "<p><small>This software uses the following open source libraries:</small></p>"
            "<ul>"
            "<li><small>NumPy</small></li>"
            "<li><small>PyQt6</small></li>"
            "<li><small>PyVista</small></li>"
            "<li><small>Matplotlib</small></li>"
            "<li><small>scikit-image</small></li>"
            "<li><small>PyDICOM</small></li>"
            "<li><small>OpenCV</small></li>"
            "</ul>"
            "<p><small>License: MIT</small></p>"
        )
        
        QMessageBox.about(self, "About Voxel Projector", about_text)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings
        self.save_settings()
        
        # Clean up video capture if active
        if hasattr(self, 'video_tab'):
            self.video_tab.stop_capture()
        
        # Clean up resources
        if self.visualization_engine == "pyvista":
            try:
                self.voxel_viewer.close()
            except:
                pass
        
        # Accept the event
        event.accept()


class LogHandler(logging.Handler):
    """Custom log handler that outputs to a QTextEdit"""
    
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Define log colors
        self.colors = {
            logging.DEBUG: 'gray',
            logging.INFO: 'black',
            logging.WARNING: 'orange',
            logging.ERROR: 'red',
            logging.CRITICAL: 'darkred'
        }
    
    def emit(self, record):
        """Format and emit a log record to the text widget"""
        color = self.colors.get(record.levelno, 'black')
        formatted = self.formatter.format(record)
        
        # HTML coloring
        html = f'<span style="color:{color};">{formatted}</span>'
        
        # Append to text edit
        self.text_edit.append(html)
        
        # Auto-scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())