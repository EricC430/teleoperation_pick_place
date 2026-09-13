import logging

import cv2

from lerobot.cameras.opencv import OpenCVCamera

from .config_opencv_uvc import OpenCVUVCCameraConfig

logger = logging.getLogger(__name__)

# DirectShow's CAP_PROP_AUTO_EXPOSURE switch values (verified on the Innomaker U20CAM-720P, 2026-09-13).
_DSHOW_MANUAL_EXPOSURE = 0.25
_DSHOW_AUTO_EXPOSURE = 0.75


class OpenCVUVCCamera(OpenCVCamera):
    """OpenCVCamera that pins exposure / gain / white balance before its read thread starts."""

    def __init__(self, config: OpenCVUVCCameraConfig):
        super().__init__(config)
        self.config: OpenCVUVCCameraConfig = config

    def _configure_capture_settings(self) -> None:
        # connect() calls this after the capture opens and BEFORE _start_read_thread(): no cap.set() races the
        # reader, and the warmup that follows lets the new exposure settle.
        super()._configure_capture_settings()
        self._apply_uvc_controls()

    def _apply_uvc_controls(self) -> None:
        cfg = self.config
        if cfg.exposure is None:
            # Read-back of AUTO_EXPOSURE under DSHOW is not meaningful (-1 on this camera), so it is not verified.
            self._set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_AUTO_EXPOSURE, "auto-exposure on", verify=False)
        else:
            self._set(cv2.CAP_PROP_AUTO_EXPOSURE, _DSHOW_MANUAL_EXPOSURE, "auto-exposure off", verify=False)
            self._set(cv2.CAP_PROP_EXPOSURE, cfg.exposure, "exposure")
            self._set(cv2.CAP_PROP_GAIN, cfg.gain, "gain")
        if cfg.white_balance is None:
            self._set(cv2.CAP_PROP_AUTO_WB, 1, "auto white balance on")
        else:
            self._set(cv2.CAP_PROP_AUTO_WB, 0, "auto white balance off")
            self._set(cv2.CAP_PROP_WB_TEMPERATURE, cfg.white_balance, "white balance")
        logger.info(
            f"{self} UVC controls: exposure={cfg.exposure} gain={cfg.gain} "
            f"white_balance={cfg.white_balance} (None = auto)"
        )

    def _set(self, prop: int, value: float, name: str, verify: bool = True) -> None:
        ok = self.videocapture.set(prop, float(value))
        actual = self.videocapture.get(prop)
        if not ok or (verify and round(actual) != round(value)):
            raise RuntimeError(f"{self} failed to set {name}={value} (actual={actual}, success={ok}).")
