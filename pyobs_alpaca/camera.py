from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from pyobs.images import Image
from pyobs.interfaces import (
    Binning,
    BinningCapabilities,
    BinningState,
    CoolingState,
    GainState,
    IAbortable,
    IBinning,
    ICooling,
    IGain,
    ITemperatures,
    IWindow,
    SensorReading,
    TemperaturesState,
    WindowCapabilities,
    WindowState,
)
from pyobs.modules.camera.basecamera import BaseCamera
from pyobs.utils import exceptions as exc
from pyobs.utils.enums import ExposureStatus

from .device import DEVICE_INIT_KWARGS, OBJECT_SHARED_KWARGS, AlpacaDevice

log = logging.getLogger(__name__)

# ASCOM CameraStates enum values
_CAMERA_STATE_ERROR = 5

# ASCOM SensorType enum values
_SENSOR_TYPE_RGGB = 2


@dataclass(frozen=True, slots=True)
class CameraMetadata:
    """Immutable static capabilities and sensor geometry queried on open()."""

    camera_x_size: int
    camera_y_size: int
    pixel_size_x_um: float
    pixel_size_y_um: float
    max_bin_x: int
    max_bin_y: int
    can_asymmetric_bin: bool
    sensor_type: int
    bayer_offset_x: int
    bayer_offset_y: int
    can_abort_exposure: bool
    can_set_ccd_temperature: bool
    can_get_cooler_power: bool
    gain_min: int | None
    gain_max: int | None
    offset_min: int | None
    offset_max: int | None
    description: str
    sensor_name: str


class AlpacaCamera(
    BaseCamera,
    IWindow,
    IBinning,
    ICooling,
    IGain,
    IAbortable,
):
    """A pyobs camera module for ASCOM Alpaca camera devices (ICameraV3)."""

    __module__ = "pyobs_alpaca"

    def __init__(
        self,
        cooling_setpoint: float | None = None,
        default_gain: float | None = None,
        default_offset: float | None = None,
        poll_interval: float = 0.2,
        readout_timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        if poll_interval <= 0 or not math.isfinite(poll_interval):
            raise ValueError("poll_interval must be a finite positive number.")
        if readout_timeout <= 0 or not math.isfinite(readout_timeout):
            raise ValueError("readout_timeout must be a finite positive number.")

        super().__init__(
            **{k: v for k, v in kwargs.items() if k not in DEVICE_INIT_KWARGS},
        )
        self.add_background_task(self._update_cooling, True)

        device_kwargs = {k: v for k, v in kwargs.items() if k in DEVICE_INIT_KWARGS | OBJECT_SHARED_KWARGS}
        device_kwargs.setdefault("device_type", "camera")
        self._device: AlpacaDevice = self.add_child_object(AlpacaDevice, **device_kwargs)

        self._default_cooling_setpoint = cooling_setpoint
        self._default_gain = default_gain
        self._default_offset = default_offset
        self._poll_interval = poll_interval
        self._readout_timeout = readout_timeout
        self._meta: CameraMetadata | None = None
        self._full_frame: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._window: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._binning: tuple[int, int] = (1, 1)
        self._gain: float | None = None
        self._offset: float | None = None
        self._cooling_enabled: bool = False
        self._cooling_setpoint: float | None = cooling_setpoint
        self._cooling_power: float | None = None
        self._temperatures: dict[str, float] = {}

    async def _try_get(self, attribute: str) -> Any:
        """Query an optional ASCOM property that may raise PropertyNotImplemented (1024)."""
        try:
            return await self._device.get(attribute)
        except ConnectionError:
            return None

    async def _query_metadata(self) -> CameraMetadata:
        """Query static sensor geometry and capabilities from the Alpaca camera."""
        camera_x_size = int(await self._device.get("CameraXSize"))
        camera_y_size = int(await self._device.get("CameraYSize"))
        if camera_x_size <= 0 or camera_y_size <= 0:
            raise ConnectionError(f"Invalid sensor dimensions from Alpaca camera: {camera_x_size}x{camera_y_size}")

        pixel_size_x_um = float(await self._device.get("PixelSizeX"))
        pixel_size_y_um = float(await self._device.get("PixelSizeY"))
        max_bin_x = int(await self._device.get("MaxBinX"))
        max_bin_y = int(await self._device.get("MaxBinY"))
        if max_bin_x < 1 or max_bin_y < 1:
            raise ConnectionError(f"Invalid maximum binning from Alpaca camera: {max_bin_x}x{max_bin_y}")

        can_asymmetric_bin = bool(await self._device.get("CanAsymmetricBin"))
        sensor_type = int(await self._device.get("SensorType"))
        bayer_offset_x = 0
        bayer_offset_y = 0
        if sensor_type != 0:
            raw_bx = await self._try_get("BayerOffsetX")
            raw_by = await self._try_get("BayerOffsetY")
            if raw_bx is not None:
                bayer_offset_x = int(raw_bx)
            if raw_by is not None:
                bayer_offset_y = int(raw_by)

        can_abort_exposure = bool(await self._device.get("CanAbortExposure"))
        can_set_ccd_temperature = bool(await self._device.get("CanSetCCDTemperature"))
        can_get_cooler_power = bool(await self._device.get("CanGetCoolerPower"))

        raw_gain_min = await self._try_get("GainMin")
        raw_gain_max = await self._try_get("GainMax")
        if raw_gain_min is not None and raw_gain_max is not None:
            gain_min: int | None = int(raw_gain_min)
            gain_max: int | None = int(raw_gain_max)
        else:
            raw_gains = await self._try_get("Gains")
            if isinstance(raw_gains, list) and len(raw_gains) > 0:
                gain_min = 0
                gain_max = len(raw_gains) - 1
            else:
                gain_min = None
                gain_max = None

        raw_offset_min = await self._try_get("OffsetMin")
        raw_offset_max = await self._try_get("OffsetMax")
        if raw_offset_min is not None and raw_offset_max is not None:
            offset_min: int | None = int(raw_offset_min)
            offset_max: int | None = int(raw_offset_max)
        else:
            raw_offsets = await self._try_get("Offsets")
            if isinstance(raw_offsets, list) and len(raw_offsets) > 0:
                offset_min = 0
                offset_max = len(raw_offsets) - 1
            else:
                offset_min = None
                offset_max = None

        description = str(await self._device.get("Description"))
        sensor_name = str(await self._device.get("SensorName"))

        return CameraMetadata(
            camera_x_size=camera_x_size,
            camera_y_size=camera_y_size,
            pixel_size_x_um=pixel_size_x_um,
            pixel_size_y_um=pixel_size_y_um,
            max_bin_x=max_bin_x,
            max_bin_y=max_bin_y,
            can_asymmetric_bin=can_asymmetric_bin,
            sensor_type=sensor_type,
            bayer_offset_x=bayer_offset_x,
            bayer_offset_y=bayer_offset_y,
            can_abort_exposure=can_abort_exposure,
            can_set_ccd_temperature=can_set_ccd_temperature,
            can_get_cooler_power=can_get_cooler_power,
            gain_min=gain_min,
            gain_max=gain_max,
            offset_min=offset_min,
            offset_max=offset_max,
            description=description,
            sensor_name=sensor_name,
        )

    @staticmethod
    def _build_binnings(meta: CameraMetadata) -> list[Binning]:
        if meta.can_asymmetric_bin:
            return [Binning(x=bx, y=by) for bx in range(1, meta.max_bin_x + 1) for by in range(1, meta.max_bin_y + 1)]
        max_sym = min(meta.max_bin_x, meta.max_bin_y)
        return [Binning(x=b, y=b) for b in range(1, max_sym + 1)]

    async def open(self) -> None:
        """Open connection to the Alpaca camera and query static metadata."""
        await super().open()

        connected = bool(await self._device.get("Connected"))
        if not connected:
            await self._device.put("Connected", Connected=True)

        meta = await self._query_metadata()
        self._meta = meta
        self._full_frame = (0, 0, meta.camera_x_size, meta.camera_y_size)
        self._window = self._full_frame
        self._binning = (1, 1)

        if meta.gain_min is not None:
            if self._default_gain is not None:
                await self.set_gain(self._default_gain)
            else:
                raw_gain = await self._try_get("Gain")
                self._gain = float(raw_gain) if raw_gain is not None else float(meta.gain_min)
        else:
            self._gain = None

        if meta.offset_min is not None:
            if self._default_offset is not None:
                await self.set_offset(self._default_offset)
            else:
                raw_offset = await self._try_get("Offset")
                self._offset = float(raw_offset) if raw_offset is not None else float(meta.offset_min)
        else:
            self._offset = None

        await self.comm.set_capabilities(
            IWindow,
            WindowCapabilities(
                full_frame_x=0,
                full_frame_y=0,
                full_frame_width=meta.camera_x_size,
                full_frame_height=meta.camera_y_size,
            ),
        )
        await self.comm.set_capabilities(
            IBinning,
            BinningCapabilities(binnings=self._build_binnings(meta)),
        )
        await self.comm.set_state(
            IWindow,
            WindowState(
                x=self._window[0],
                y=self._window[1],
                width=self._window[2],
                height=self._window[3],
            ),
        )
        await self.comm.set_state(IBinning, BinningState(x=self._binning[0], y=self._binning[1]))
        if self._gain is not None or self._offset is not None:
            await self.comm.set_state(
                IGain,
                GainState(
                    gain=self._gain if self._gain is not None else 0.0,
                    offset=self._offset if self._offset is not None else 0.0,
                ),
            )
        if self._default_cooling_setpoint is not None and meta.can_set_ccd_temperature:
            await self.set_cooling(True, self._default_cooling_setpoint)
        await self._poll_cooling()

    async def _poll_cooling(self) -> None:
        """Query temperatures and cooler status from the Alpaca camera and publish states."""
        if self._meta is None:
            return

        temps: dict[str, float] = {}
        raw_ccd_temp = await self._try_get("CCDTemperature")
        if raw_ccd_temp is not None:
            temps["CCD"] = float(raw_ccd_temp)

        raw_heatsink_temp = await self._try_get("HeatSinkTemperature")
        if raw_heatsink_temp is not None:
            temps["Heatsink"] = float(raw_heatsink_temp)

        self._temperatures = temps

        raw_cooler_on = await self._try_get("CoolerOn")
        self._cooling_enabled = bool(raw_cooler_on) if raw_cooler_on is not None else False

        if self._meta.can_set_ccd_temperature:
            raw_setpoint = await self._try_get("SetCCDTemperature")
            if raw_setpoint is not None:
                self._cooling_setpoint = float(raw_setpoint)

        if self._meta.can_get_cooler_power:
            raw_power = await self._try_get("CoolerPower")
            self._cooling_power = float(raw_power) if raw_power is not None else None
        else:
            self._cooling_power = None

        await self.comm.set_state(
            ITemperatures,
            TemperaturesState(readings=[SensorReading(name=name, value=val) for name, val in temps.items()]),
        )
        await self.comm.set_state(
            ICooling,
            CoolingState(
                setpoint=self._cooling_setpoint,
                power=int(round(self._cooling_power)) if self._cooling_power is not None else None,
                enabled=self._cooling_enabled,
            ),
        )

    async def _update_cooling(self) -> None:
        """Background task periodically polling camera temperatures and cooler status."""
        while True:
            if self._meta is not None:
                await self._poll_cooling()
            await asyncio.sleep(5.0)

    async def close(self) -> None:
        """Close the camera module and release the underlying HTTP session."""
        await self._device.close()
        await super().close()

    async def reset(self, **kwargs: Any) -> None:
        """Reset exposure time, image type, binning, window, gain, offset, and pipeline to their defaults."""
        await BaseCamera.reset(self, **kwargs)
        if self._meta is not None:
            await self.set_binning(1, 1)
            await self.set_window(*self._full_frame)
            if self._default_gain is not None and self._meta.gain_min is not None:
                await self.set_gain(self._default_gain)
            if self._default_offset is not None and self._meta.offset_min is not None:
                await self.set_offset(self._default_offset)

    async def full_reset(self, **kwargs: Any) -> None:
        """Reconnect the Alpaca camera hardware and reset settings to defaults."""
        if self._camera_status != ExposureStatus.IDLE or self._sequence_count_left > 0 or self._stack_running:
            raise exc.DeviceBusyError("Cannot reset camera while it is busy.")
        await self._device.put("Connected", Connected=False)
        await self._device.put("Connected", Connected=True)
        self._meta = await self._query_metadata()
        self._full_frame = (0, 0, self._meta.camera_x_size, self._meta.camera_y_size)
        await self.reset(**kwargs)
        if self._default_cooling_setpoint is not None and self._meta.can_set_ccd_temperature:
            await self.set_cooling(True, self._default_cooling_setpoint)

    async def get_full_frame(self, **kwargs: Any) -> tuple[int, int, int, int]:
        """Return full frame size of camera (left, top, width, height)."""
        return self._full_frame

    async def get_window(self, **kwargs: Any) -> tuple[int, int, int, int]:
        """Return the camera window (left, top, width, height) in unbinned pixels."""
        return self._window

    async def get_binning(self, **kwargs: Any) -> tuple[int, int]:
        """Return the camera binning (x, y)."""
        return self._binning

    async def set_window(self, left: int, top: int, width: int, height: int, **kwargs: Any) -> None:
        """Set the camera readout window in unbinned pixels."""
        if self._meta is None:
            raise ValueError("Camera is not initialized.")
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise ValueError("Window origin must be non-negative and dimensions must be positive.")
        if left + width > self._meta.camera_x_size or top + height > self._meta.camera_y_size:
            raise ValueError(
                f"Window ({left}, {top}, {width}, {height}) exceeds sensor dimensions "
                f"({self._meta.camera_x_size}x{self._meta.camera_y_size})."
            )
        log.info("Setting window to %dx%d at %d,%d...", width, height, left, top)
        self._window = (left, top, width, height)
        await self.comm.set_state(IWindow, WindowState(x=left, y=top, width=width, height=height))

    async def set_binning(self, x: int, y: int, **kwargs: Any) -> None:
        """Set the camera binning (x, y)."""
        if self._meta is None:
            raise ValueError("Camera is not initialized.")
        if x < 1 or y < 1 or x > self._meta.max_bin_x or y > self._meta.max_bin_y:
            raise ValueError(
                f"Binning {x}x{y} is out of supported range (1..{self._meta.max_bin_x} x 1..{self._meta.max_bin_y})."
            )
        if not self._meta.can_asymmetric_bin and x != y:
            raise ValueError(f"Asymmetric binning {x}x{y} is not supported by this camera.")
        log.info("Setting binning to %dx%d...", x, y)
        self._binning = (x, y)
        await self.comm.set_state(IBinning, BinningState(x=x, y=y))

    async def get_gain(self, **kwargs: Any) -> float:
        """Return the current camera gain."""
        if self._gain is None:
            raise ValueError("Camera does not support gain.")
        return self._gain

    async def set_gain(self, gain: float, **kwargs: Any) -> None:
        """Set the camera gain within [gain_min, gain_max]."""
        if self._meta is None or self._meta.gain_min is None or self._meta.gain_max is None:
            raise ValueError("Camera does not support setting gain.")
        if not math.isfinite(gain):
            raise ValueError("Gain must be finite.")
        gain_int = int(round(gain))
        if gain_int < self._meta.gain_min or gain_int > self._meta.gain_max:
            raise ValueError(
                f"Gain {gain_int} is outside supported range [{self._meta.gain_min}, {self._meta.gain_max}]."
            )
        try:
            await self._device.put("Gain", Gain=gain_int)
        except ConnectionError as e:
            raise ValueError(f"Could not set camera gain: {e}") from e
        self._gain = float(gain_int)
        await self.comm.set_state(
            IGain,
            GainState(
                gain=self._gain,
                offset=self._offset if self._offset is not None else 0.0,
            ),
        )

    async def get_offset(self, **kwargs: Any) -> float:
        """Return the current camera offset."""
        if self._offset is None:
            raise ValueError("Camera does not support offset.")
        return self._offset

    async def set_offset(self, offset: float, **kwargs: Any) -> None:
        """Set the camera offset within [offset_min, offset_max]."""
        if self._meta is None or self._meta.offset_min is None or self._meta.offset_max is None:
            raise ValueError("Camera does not support setting offset.")
        if not math.isfinite(offset):
            raise ValueError("Offset must be finite.")
        offset_int = int(round(offset))
        if offset_int < self._meta.offset_min or offset_int > self._meta.offset_max:
            raise ValueError(
                f"Offset {offset_int} is outside supported range [{self._meta.offset_min}, {self._meta.offset_max}]."
            )
        try:
            await self._device.put("Offset", Offset=offset_int)
        except ConnectionError as e:
            raise ValueError(f"Could not set camera offset: {e}") from e
        self._offset = float(offset_int)
        await self.comm.set_state(
            IGain,
            GainState(
                gain=self._gain if self._gain is not None else 0.0,
                offset=self._offset,
            ),
        )

    async def set_cooling(self, enabled: bool, setpoint: float, **kwargs: Any) -> None:
        """Enable/disable camera cooling and set the target CCD temperature in Celsius."""
        if self._meta is None or not self._meta.can_set_ccd_temperature:
            raise ValueError("Camera does not support controllable cooling.")
        if not math.isfinite(setpoint):
            raise ValueError("Cooling setpoint must be finite.")
        try:
            await self._device.put("CoolerOn", CoolerOn=bool(enabled))
            if enabled:
                await self._device.put("SetCCDTemperature", SetCCDTemperature=float(setpoint))
        except ConnectionError as e:
            raise ValueError(f"Could not set camera cooling: {e}") from e
        self._cooling_enabled = bool(enabled)
        self._cooling_setpoint = float(setpoint)
        await self.comm.set_state(
            ICooling,
            CoolingState(
                setpoint=self._cooling_setpoint,
                power=int(round(self._cooling_power)) if self._cooling_power is not None else None,
                enabled=self._cooling_enabled,
            ),
        )

    async def get_cooling_status(self, **kwargs: Any) -> tuple[bool, float | None, float | None]:
        """Return the current cooling status (enabled, setpoint, power)."""
        return (self._cooling_enabled, self._cooling_setpoint, self._cooling_power)

    async def get_temperatures(self, **kwargs: Any) -> dict[str, float]:
        """Return the most recently polled camera temperatures."""
        return dict(self._temperatures)

    async def _apply_roi_and_binning(self, window: tuple[int, int, int, int], binning: tuple[int, int]) -> None:
        """Push binning and binned ROI coordinates to the Alpaca camera before exposure."""
        bx, by = binning
        left, top, width, height = window
        start_x = left // bx
        start_y = top // by
        num_x = width // bx
        num_y = height // by
        if num_x <= 0 or num_y <= 0:
            raise exc.GrabImageError(f"Binned window dimensions ({num_x}x{num_y}) must be positive.")
        try:
            await self._device.put("BinX", BinX=bx)
            await self._device.put("BinY", BinY=by)
            await self._device.put("StartX", StartX=start_x)
            await self._device.put("StartY", StartY=start_y)
            await self._device.put("NumX", NumX=num_x)
            await self._device.put("NumY", NumY=num_y)
        except ConnectionError as e:
            raise exc.GrabImageError(f"Could not configure camera window/binning: {e}") from e

    async def _abort_exposure(self) -> None:
        """Send AbortExposure or StopExposure to the Alpaca camera."""
        if self._meta is None:
            return
        try:
            if self._meta.can_abort_exposure:
                await self._device.put("AbortExposure")
            else:
                await self._device.put("StopExposure")
        except ConnectionError as e:
            raise exc.GrabImageError(f"Failed to abort exposure on Alpaca camera: {e}") from e

    def _populate_header(
        self,
        hdr: fits.Header,
        exposure_time: float,
        date_obs: datetime,
        data: NDArray[Any],
        window: tuple[int, int, int, int],
        binning: tuple[int, int],
        gain: float | None,
        offset: float | None,
    ) -> None:
        """Populate standard FITS keywords on the captured image."""
        if self._meta is None:
            return

        hdr["NAXIS1"] = int(data.shape[1])
        hdr["NAXIS2"] = int(data.shape[0])
        hdr["DATE-OBS"] = (date_obs.strftime("%Y-%m-%dT%H:%M:%S.%f"), "Date and time of start of exposure")
        hdr["EXPTIME"] = (float(exposure_time), "Exposure time [s]")
        hdr["XBINNING"] = hdr["DET-BIN1"] = (binning[0], "Binning factor used on X axis")
        hdr["YBINNING"] = hdr["DET-BIN2"] = (binning[1], "Binning factor used on Y axis")
        hdr["XORGSUBF"] = (window[0], "Subframe origin on X axis")
        hdr["YORGSUBF"] = (window[1], "Subframe origin on Y axis")

        if self._meta.pixel_size_x_um > 0:
            hdr["DET-PIXL"] = (self._meta.pixel_size_x_um / 1000.0, "Size of detector pixels (square) [mm]")

        if self._meta.description:
            instrume = (
                self._meta.description.replace("\u00b5", "u")
                .replace("\u03bc", "u")
                .encode("ascii", errors="replace")
                .decode("ascii")
            )
            hdr["INSTRUME"] = (instrume, "Name of instrument")
        if self._meta.sensor_name:
            detector = (
                self._meta.sensor_name.replace("\u00b5", "u")
                .replace("\u03bc", "u")
                .encode("ascii", errors="replace")
                .decode("ascii")
            )
            hdr["DETECTOR"] = (detector, "Name of detector")

        if "CCD" in self._temperatures:
            hdr["DET-TEMP"] = (self._temperatures["CCD"], "Temperature of detector [C]")
        if self._cooling_power is not None:
            hdr["DET-CPWR"] = (self._cooling_power, "Cooler power [%]")
        if self._cooling_setpoint is not None:
            hdr["DET-TSET"] = (self._cooling_setpoint, "Cooler setpoint [C]")

        if gain is not None:
            hdr["GAIN"] = (gain, "Gain used for exposure")
        if offset is not None:
            hdr["OFFSET"] = (offset, "Offset used for exposure")

        if self._meta.sensor_type == _SENSOR_TYPE_RGGB:
            hdr["BAYERPAT"] = ("RGGB", "Bayer color pattern")
            hdr["XBAYROFF"] = (self._meta.bayer_offset_x, "Bayer X offset")
            hdr["YBAYROFF"] = (self._meta.bayer_offset_y, "Bayer Y offset")

        hdr["DATAMIN"] = float(np.min(data))
        hdr["DATAMAX"] = float(np.max(data))
        hdr["DATAMEAN"] = float(np.mean(data))
        self.set_biassec_trimsec(hdr, *self._full_frame)

    @staticmethod
    def _normalize_image_data(raw_data: NDArray[Any]) -> NDArray[Any]:
        """Normalizes raw Alpaca image array into a 16-bit unsigned integer array."""
        if raw_data.dtype == np.uint16:
            return raw_data
        return np.clip(raw_data, 0, 65535).astype(np.uint16)

    async def _expose(self, exposure_time: float, open_shutter: bool, abort_event: asyncio.Event) -> Image:
        """Execute an exposure on the Alpaca camera and fetch the frame via ImageBytes."""
        if self._meta is None:
            raise exc.GrabImageError("Camera is not initialized.")

        log.info("Starting %.3fs exposure with %s shutter...", exposure_time, "open" if open_shutter else "closed")
        window = self._window
        binning = self._binning
        gain = self._gain
        offset = self._offset
        await self._apply_roi_and_binning(window, binning)
        date_obs = datetime.now(UTC)

        try:
            await self._device.put("StartExposure", Duration=exposure_time, Light=open_shutter)
        except ConnectionError as e:
            raise exc.GrabImageError(f"Could not start exposure: {e}") from e

        deadline = time.monotonic() + exposure_time + self._readout_timeout

        # Sleep for the bulk of the exposure without spamming HTTP requests, waking early on abort
        bulk_wait = exposure_time - self._poll_interval * 2.0
        if bulk_wait > 0:
            try:
                await asyncio.wait_for(abort_event.wait(), timeout=bulk_wait)
            except TimeoutError:
                pass

        while True:
            if abort_event.is_set():
                await self._abort_exposure()
                await self._change_exposure_status(ExposureStatus.IDLE)
                raise exc.AbortedError("Exposure was aborted.")

            try:
                ready = bool(await self._device.get("ImageReady"))
            except ConnectionError as e:
                raise exc.GrabImageError(f"Failed to poll ImageReady: {e}") from e

            if ready:
                break

            try:
                cam_state = int(await self._device.get("CameraState"))
            except ConnectionError as e:
                raise exc.GrabImageError(f"Failed to poll CameraState: {e}") from e

            if cam_state == _CAMERA_STATE_ERROR:
                raise exc.GrabImageError("Camera reported CameraError state during exposure.")

            if time.monotonic() >= deadline:
                try:
                    await self._abort_exposure()
                except exc.GrabImageError:
                    pass
                raise exc.GrabImageError("Timed out waiting for camera ImageReady.")

            try:
                await asyncio.wait_for(abort_event.wait(), timeout=self._poll_interval)
            except TimeoutError:
                pass

        await self._change_exposure_status(ExposureStatus.READOUT)
        try:
            raw_data = await self._device.get_imagebytes("imagearray", timeout=self._readout_timeout)
        except ConnectionError as e:
            raise exc.GrabImageError(f"Could not download image array: {e}") from e

        data = self._normalize_image_data(raw_data)
        image = Image(data)
        self._populate_header(
            image.header,
            exposure_time,
            date_obs,
            data,
            window=window,
            binning=binning,
            gain=gain,
            offset=offset,
        )
        log.info("Exposure finished.")
        return image


__all__ = ["AlpacaCamera", "CameraMetadata"]
