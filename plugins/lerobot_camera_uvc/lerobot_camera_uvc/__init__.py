"""LeRobot camera plugin: a UVC camera (OpenCV / DirectShow) whose exposure, gain and white balance live in the YAML.

lerobot's own ``type: opencv`` has no exposure fields, so the wrist UVC camera would record on auto-exposure
(D022 §2026-09-13). ``register_third_party_plugins()`` - called by lerobot-record and lerobot-teleoperate -
imports every installed distribution whose name starts with ``lerobot_camera_``; importing this package
registers ``type: opencv_uvc`` and ``type: intelrealsense_pinned`` (a RealSense camera whose None means AUTO, set
explicitly on connect - lerobot's own `intelrealsense` leaves None unchanged).
"""

from .config_opencv_uvc import OpenCVUVCCameraConfig
from .config_realsense_pinned import RealSensePinnedCameraConfig
from .opencv_uvc import OpenCVUVCCamera
from .realsense_pinned import RealSensePinnedCamera

__all__ = ["OpenCVUVCCamera", "OpenCVUVCCameraConfig", "RealSensePinnedCamera", "RealSensePinnedCameraConfig"]
