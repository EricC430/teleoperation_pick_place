from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig


@CameraConfig.register_subclass("intelrealsense_pinned")
@dataclass
class RealSensePinnedCameraConfig(RealSenseCameraConfig):
    """``type: intelrealsense`` whose colour controls are set explicitly on every connect.

    lerobot's own RealSense camera leaves an omitted exposure / gain / white_balance UNCHANGED, and the D455 keeps
    those states across processes (measured 2026-09-13: auto white balance left off by one process was still off
    when lerobot opened the camera with white_balance=None - visibly green). Here None means **force AUTO**, the
    same semantics as ``type: opencv_uvc``.

    exposure and gain go together (both set = manual exposure, both None = auto-exposure): a lone value would
    leave the other at whatever the previous program set.
    """

    def __post_init__(self) -> None:
        post = getattr(super(), "__post_init__", None)
        if post is not None:
            post()
        if (self.exposure is None) != (self.gain is None):
            raise ValueError(
                "`intelrealsense_pinned` needs `exposure` and `gain` together (both set, or both null for AUTO); "
                f"got exposure={self.exposure}, gain={self.gain}."
            )
