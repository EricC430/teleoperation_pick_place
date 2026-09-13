from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig

# DirectShow expresses exposure as log2(seconds): -6 -> 2**-6 s = 15.6 ms.
EXPOSURE_MIN, EXPOSURE_MAX = -13, -1


@CameraConfig.register_subclass("opencv_uvc")
@dataclass
class OpenCVUVCCameraConfig(OpenCVCameraConfig):
    """``type: opencv`` plus fixed exposure / gain / white balance, re-applied on every connect.

    Unlike RealSense's "None = leave unchanged", None here means **force auto**: UVC controls persist inside
    the device across processes (measured 2026-09-13), so "unchanged" would silently inherit whatever the
    previous program set.

    Attributes:
        exposure: DirectShow exposure in log2 seconds (e.g. -6 = 15.6 ms). None = auto-exposure.
            Must be shorter than one frame period, or the camera drops frames (-4 at 30 fps gave 17 fps).
        gain: Sensor gain. Required when ``exposure`` is set (a stale gain would otherwise carry over);
            must be None when it is not (auto-exposure owns the gain).
        white_balance: Colour temperature in kelvin. None = auto white balance.
    """

    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend != Cv2Backends.DSHOW:
            raise ValueError(
                "`opencv_uvc` needs `backend: DSHOW` (exposure is in DirectShow's log2-seconds units), "
                f"got {self.backend.name}."
            )
        if self.exposure is None:
            if self.gain is not None:
                raise ValueError("`gain` is set without `exposure`: auto-exposure owns the gain. Set both or neither.")
            return
        if self.gain is None:
            raise ValueError("`exposure` is set but `gain` is not: set both, or a stale gain carries over.")
        if not EXPOSURE_MIN <= self.exposure <= EXPOSURE_MAX:
            raise ValueError(
                f"`exposure` must be in [{EXPOSURE_MIN}, {EXPOSURE_MAX}] (log2 seconds), got {self.exposure}."
            )
        if self.fps is not None and 2.0**self.exposure >= 1.0 / self.fps:
            raise ValueError(
                f"`exposure: {self.exposure}` = {2.0**self.exposure * 1e3:.1f} ms is not shorter than one frame at "
                f"{self.fps} fps ({1e3 / self.fps:.1f} ms): the camera would drop frames."
            )
