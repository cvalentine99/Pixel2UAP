#!/usr/bin/env python3
"""
Main application entry point for Voxel Projector v2.
"""

import sys
import os
import platform
import argparse
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, Qt

from voxel_projector_v2.ui.main_window import MainWindow
from voxel_projector_v2.utils.logging_config import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Voxel Projector v2")
    
    parser.add_argument(
        "--input", 
        help="Input file or directory to load on startup"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--no-gpu", 
        action="store_true",
        help="Disable GPU acceleration"
    )
    
    parser.add_argument(
        "--software-render", 
        action="store_true",
        help="Force software rendering for PyVista"
    )
    
    parser.add_argument(
        "--version", 
        action="store_true",
        help="Show version information and exit"
    )
    
    return parser.parse_args()


def show_version():
    """Display version and system information"""
    from voxel_projector_v2 import __version__
    
    print(f"Voxel Projector v{__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # Show available optional features
    from voxel_projector_v2 import (
        PYVISTA_AVAILABLE, 
        DICOM_AVAILABLE, 
        NIFTI_AVAILABLE, 
        OPENCV_AVAILABLE,
        CUPY_AVAILABLE
    )
    
    print("\nOptional features:")
    print(f"- 3D Visualization (PyVista): {'Available' if PYVISTA_AVAILABLE else 'Not available'}")
    print(f"- DICOM Support: {'Available' if DICOM_AVAILABLE else 'Not available'}")
    print(f"- NIFTI Support: {'Available' if NIFTI_AVAILABLE else 'Not available'}")
    print(f"- Video Support (OpenCV): {'Available' if OPENCV_AVAILABLE else 'Not available'}")
    print(f"- GPU Acceleration (CuPy): {'Available' if CUPY_AVAILABLE else 'Not available'}")


def setup_environment(args):
    """Configure environment variables and settings"""
    # Set high-DPI scaling - these attributes may be different in PyQt6
    # Check if these attributes exist
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Configure PyVista rendering if needed
    if args.software_render:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
        os.environ["PYVISTA_USE_IPYVTK"] = "false"
        logging.info("Software rendering enabled for PyVista")
    
    # Disable GPU acceleration if requested
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("GPU acceleration disabled")


def main():
    """Main application entry point"""
    try:
        args = parse_arguments()
        
        # Show version and exit if requested
        if args.version:
            show_version()
            return 0
        
        # Set up logging with debug level for more detailed output
        setup_logging("DEBUG")
        
        # Configure environment
        setup_environment(args)
        
        # Add a memory usage logger to check for memory issues
        import os
        import psutil
        process = psutil.Process(os.getpid())
        logging.info(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Log Python information
        import platform
        logging.info(f"Python version: {platform.python_version()}")
        logging.info(f"Platform: {platform.platform()}")
        
        # Create Qt application with better error handling
        app = QApplication(sys.argv)
        app.setApplicationName("Voxel Projector")
        app.setOrganizationName("VoxelProjector")
        
        # Set application style
        app.setStyle("Fusion")
        
        # Log Qt version
        from PyQt6.QtCore import QT_VERSION_STR
        logging.info(f"Qt version: {QT_VERSION_STR}")
        
        try:
            # Create and show main window with additional error handling
            logging.info("Creating main window...")
            window = MainWindow()
            logging.info("Showing main window...")
            window.show()
            
            # Load input file if provided
            if args.input:
                logging.info(f"Loading input from {args.input}")
                window.load_input(args.input)
                
            # Log memory usage before starting event loop
            logging.info(f"Memory usage before event loop: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                
            # Start event loop
            logging.info("Starting Qt event loop")
            return app.exec()
            
        except Exception as window_error:
            # Log window creation errors
            import traceback
            logging.error(f"Error creating or showing window: {window_error}")
            logging.error(traceback.format_exc())
            
            # Show error message to user
            from PyQt6.QtWidgets import QMessageBox
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setWindowTitle("Application Error")
            error_box.setText("The application encountered an error during startup.")
            error_box.setDetailedText(str(window_error) + "\n\n" + traceback.format_exc())
            error_box.exec()
            return 1
            
    except Exception as e:
        # Handle any other unexpected errors
        import traceback
        print(f"Critical error in application startup: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())