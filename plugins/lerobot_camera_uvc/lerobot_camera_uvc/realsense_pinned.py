import logging

from lerobot.cameras.realsense import RealSenseCamera

from .config_realsense_pinned import RealSensePinnedCameraConfig

logger = logging.getLogger(__name__)


class RealSensePinnedCamera(RealSenseCamera):
    """RealSenseCamera that switches AUTO back on for every colour control the config leaves as None."""

    def __init__(self, config: RealSensePinnedCameraConfig):
        super().__init__(config)
        self.config: RealSensePinnedCameraConfig = config

    def _configure_sensor_options(self) -> None:
        # _open_pipeline() calls this right after the pipeline starts and BEFORE the read thread starts.
        super()._configure_sensor_options()  # applies the values that are set (and switches their auto modes off)
        import pyrealsense2 as rs

        sensor = self._get_color_sensor()
        if self.exposure is None and self.gain is None and sensor.supports(rs.option.enable_auto_exposure):
            self._set_sensor_option(sensor, rs.option.enable_auto_exposure, 1, "auto-exposure")
        if self.white_balance is None and sensor.supports(rs.option.enable_auto_white_balance):
            self._set_sensor_option(sensor, rs.option.enable_auto_white_balance, 1, "auto white balance")
        logger.info(
            f"{self} colour controls: exposure={self.exposure} gain={self.gain} "
            f"white_balance={self.white_balance} (None = AUTO)"
        )
