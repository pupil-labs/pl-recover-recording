"""Top-level entry-point for the <project_name> package"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pupil_labs.recording.recover")
except PackageNotFoundError:
    # package is not installed
    pass

from .recover import cli, recover_recording

__all__ = ["__version__", "cli", "recover_recording"]
