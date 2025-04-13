"""
Custom error classes and error handling utilities
"""

import platform
import sys
import logging
import traceback
from PyQt6.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


def get_platform_info():
    """Get basic information about the current platform"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "python": platform.python_version(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
    }


class VoxelProjectorError(Exception):
    """Base exception for Voxel Projector"""
    def __init__(self, message, platform_info=None):
        self.platform_info = platform_info or get_platform_info()
        self.message = message
        super().__init__(f"{message} (Platform: {self.platform_info['system']} {self.platform_info['release']})")


class InputError(VoxelProjectorError):
    """Error related to input data or files"""
    pass


class ProcessingError(VoxelProjectorError):
    """Error during data processing"""
    pass


class RenderingError(VoxelProjectorError):
    """Error during rendering or visualization"""
    pass


class ExportError(VoxelProjectorError):
    """Error during data export"""
    pass


def show_error_dialog(error, parent=None, detailed=True):
    """
    Display an error dialog to the user
    
    Args:
        error: Exception object
        parent: Parent widget or None
        detailed: Whether to include detailed information
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Error")
    
    # Set main message
    if isinstance(error, VoxelProjectorError):
        msg_box.setText(error.message)
    else:
        msg_box.setText(str(error))
    
    # Set detailed text
    if detailed:
        if isinstance(error, VoxelProjectorError) and hasattr(error, 'platform_info'):
            platform_str = "\n".join([f"{k}: {v}" for k, v in error.platform_info.items()])
            detail_text = f"Platform Information:\n{platform_str}\n\nTraceback:\n"
        else:
            detail_text = "Traceback:\n"
        
        detail_text += "".join(traceback.format_exception(
            type(error), error, error.__traceback__))
        msg_box.setDetailedText(detail_text)
    
    # Log the error
    if isinstance(error, VoxelProjectorError):
        logger.error(f"{type(error).__name__}: {error.message}")
    else:
        logger.error(f"Unexpected error: {str(error)}")
    
    logger.debug(traceback.format_exc())
    
    # Show dialog
    msg_box.exec()


def global_exception_handler(exctype, value, tb):
    """
    Global exception handler for uncaught exceptions
    
    Args:
        exctype: Exception type
        value: Exception value
        tb: Traceback object
    """
    # Log the error
    logger.critical("Uncaught exception", exc_info=(exctype, value, tb))
    
    # Show error dialog
    error_msg = str(value)
    detailed_text = "".join(traceback.format_exception(exctype, value, tb))
    
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Unhandled Error")
    msg_box.setText("An unexpected error occurred:")
    msg_box.setInformativeText(error_msg)
    msg_box.setDetailedText(detailed_text)
    msg_box.exec()


def install_global_exception_handler():
    """Install the global exception handler"""
    sys.excepthook = global_exception_handler