"""LeRobot camera plugin: a UVC camera (OpenCV / DirectShow) whose exposure, gain and white balance live in the YAML.

lerobot's own ``type: opencv`` has no exposure fields, so the wrist UVC camera would record on auto-exposure
(D022 §2026-09-13). ``register_third_party_plugins()`` - called by lerobot-record and lerobot-teleoperate -
imports every installed distribution whose name starts with ``lerobot_camera_``; importing this package
registers ``type: opencv_uvc``.
"""

from .config_opencv_uvc import OpenCVUVCCameraConfig
from .opencv_uvc import OpenCVUVCCamera

__all__ = ["OpenCVUVCCamera", "OpenCVUVCCameraConfig"]
