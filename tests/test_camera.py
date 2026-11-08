from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from numpy.typing import NDArray
from pyobs.interfaces import Binning, GainState, IBinning, IGain, IWindow
from pyobs.utils import exceptions as exc
from pyobs.utils.enums import ExposureStatus, ImageType

from pyobs_alpaca.camera import AlpacaCamera


class FakeAlpacaCameraTransport:
    """In-memory fake for _AlpacaDevice backing AlpacaCamera unit tests."""

    def __init__(self, properties: dict[str, Any] | None = None) -> None:
        self.props: dict[str, Any] = {
            "Connected": True,
            "Description": "ZWO ASI290MM Mini",
            "SensorName": "IMX290",
            "CameraXSize": 1936,
            "CameraYSize": 1096,
            "PixelSizeX": 2.9,
            "PixelSizeY": 2.9,
            "MaxBinX": 4,
            "MaxBinY": 4,
            "CanAsymmetricBin": False,
            "SensorType": 0,
            "BayerOffsetX": 0,
            "BayerOffsetY": 0,
            "CanAbortExposure": True,
            "GainMin": 0,
            "GainMax": 600,
            "Gain": 120,
            "OffsetMin": 0,
            "OffsetMax": 100,
            "Offset": 10,
            "ImageReady": True,
            "CameraState": 0,
        }
        if properties:
            self.props.update(properties)
        self.gets: list[str] = []
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self.opened = False
        self.closed = False
        self.image_array: NDArray[Any] = np.arange(12, dtype=np.uint16).reshape((3, 4))
        self.image_ready_sequence: list[bool] = []
        self.fail_on_get: set[str] = set()
        self.fail_on_put: set[str] = set()
        self.fail_imagebytes = False

    async def open(self) -> None:
        self.opened = True

    async def get(self, attribute: str) -> Any:
        if not self.opened:
            raise ConnectionError("Not connected to ASCOM.")
        self.gets.append(attribute)
        if attribute in self.fail_on_get:
            raise ConnectionError(f"Alpaca error on {attribute}")
        if attribute == "ImageReady" and self.image_ready_sequence:
            return self.image_ready_sequence.pop(0)
        if attribute not in self.props or self.props[attribute] is None:
            raise ConnectionError(f"Property {attribute} not implemented (1024)")
        return self.props[attribute]

    async def put(self, attribute: str, **kwargs: Any) -> Any:
        if not self.opened:
            raise ConnectionError("Not connected to ASCOM.")
        self.puts.append((attribute, dict(kwargs)))
        if attribute in self.fail_on_put:
            raise ConnectionError(f"Alpaca PUT error on {attribute}")
        if attribute == "Connected" and "Connected" in kwargs:
            self.props["Connected"] = bool(kwargs["Connected"])
        elif attribute == "Gain" and "Gain" in kwargs:
            self.props["Gain"] = int(kwargs["Gain"])
        elif attribute == "Offset" and "Offset" in kwargs:
            self.props["Offset"] = int(kwargs["Offset"])
        return None

    async def get_imagebytes(self, attribute: str = "imagearray", timeout: float = 30.0) -> NDArray[Any]:
        if not self.opened:
            raise ConnectionError("Not connected to ASCOM.")
        if self.fail_imagebytes:
            raise ConnectionError("ImageBytes download failed")
        return self.image_array

    async def close(self) -> None:
        self.closed = True


def _make_camera(
    fake: FakeAlpacaCameraTransport,
    poll_interval: float = 0.01,
    readout_timeout: float = 0.5,
    default_gain: float | None = None,
    default_offset: float | None = None,
) -> AlpacaCamera:
    cam = AlpacaCamera(
        server="127.0.0.1",
        port=11111,
        device=0,
        poll_interval=poll_interval,
        readout_timeout=readout_timeout,
        default_gain=default_gain,
        default_offset=default_offset,
    )
    cam._device = fake  # type: ignore[assignment]
    cam._child_objects = [fake]  # type: ignore[list-item]
    cam.comm.set_capabilities = AsyncMock()  # type: ignore[method-assign]
    cam.comm.set_state = AsyncMock()  # type: ignore[method-assign]
    cam.comm.send_event = AsyncMock()  # type: ignore[method-assign]
    cam.comm.register_event = AsyncMock()  # type: ignore[method-assign]
    return cam


def test_init_validates_poll_and_timeout_bounds() -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        AlpacaCamera(server="127.0.0.1", port=11111, device=0, poll_interval=0.0)
    with pytest.raises(ValueError, match="poll_interval"):
        AlpacaCamera(server="127.0.0.1", port=11111, device=0, poll_interval=float("inf"))
    with pytest.raises(ValueError, match="readout_timeout"):
        AlpacaCamera(server="127.0.0.1", port=11111, device=0, readout_timeout=-1.0)
    with pytest.raises(ValueError, match="readout_timeout"):
        AlpacaCamera(server="127.0.0.1", port=11111, device=0, readout_timeout=float("nan"))


@pytest.mark.asyncio
async def test_open_monochrome_skips_bayer_offsets_and_publishes_capabilities() -> None:
    fake = FakeAlpacaCameraTransport()
    cam = _make_camera(fake)
    await cam.open()

    assert cam._meta is not None
    assert cam._meta.camera_x_size == 1936
    assert cam._meta.camera_y_size == 1096
    assert cam._meta.pixel_size_x_um == 2.9
    assert cam._meta.max_bin_x == 4
    assert cam._meta.max_bin_y == 4
    assert cam._meta.can_asymmetric_bin is False
    assert cam._meta.sensor_type == 0
    assert cam._meta.bayer_offset_x == 0
    assert cam._meta.bayer_offset_y == 0
    assert cam._meta.gain_min == 0
    assert cam._meta.gain_max == 600
    assert cam._meta.offset_min == 0
    assert cam._meta.offset_max == 100
    assert "BayerOffsetX" not in fake.gets
    assert "BayerOffsetY" not in fake.gets

    assert await cam.get_full_frame() == (0, 0, 1936, 1096)
    assert await cam.get_window() == (0, 0, 1936, 1096)
    assert await cam.get_binning() == (1, 1)
    assert await cam.get_gain() == 120.0
    assert await cam.get_offset() == 10.0

    cap_calls = {call.args[0]: call.args[1] for call in cam.comm.set_capabilities.call_args_list}  # type: ignore[attr-defined]
    assert cap_calls[IWindow].full_frame_width == 1936
    assert cap_calls[IWindow].full_frame_height == 1096
    assert cap_calls[IBinning].binnings == [Binning(1, 1), Binning(2, 2), Binning(3, 3), Binning(4, 4)]

    await cam.close()
    assert fake.closed


@pytest.mark.asyncio
async def test_open_connects_if_disconnected_and_queries_bayer_offsets_for_rggb() -> None:
    fake = FakeAlpacaCameraTransport(
        {
            "Connected": False,
            "SensorType": 2,
            "BayerOffsetX": 1,
            "BayerOffsetY": 0,
            "MaxBinX": 2,
            "MaxBinY": 2,
            "CanAsymmetricBin": True,
        }
    )
    cam = _make_camera(fake)
    await cam.open()

    assert ("Connected", {"Connected": True}) in fake.puts
    assert cam._meta is not None
    assert cam._meta.sensor_type == 2
    assert cam._meta.bayer_offset_x == 1
    assert cam._meta.bayer_offset_y == 0

    cap_calls = {call.args[0]: call.args[1] for call in cam.comm.set_capabilities.call_args_list}  # type: ignore[attr-defined]
    assert cap_calls[IBinning].binnings == [Binning(1, 1), Binning(1, 2), Binning(2, 1), Binning(2, 2)]


@pytest.mark.asyncio
async def test_open_rejects_invalid_sensor_dimensions_or_max_bin() -> None:
    fake_dim = FakeAlpacaCameraTransport({"CameraXSize": 0})
    cam_dim = _make_camera(fake_dim)
    with pytest.raises(ConnectionError, match="Invalid sensor dimensions"):
        await cam_dim.open()

    fake_bin = FakeAlpacaCameraTransport({"MaxBinX": 0})
    cam_bin = _make_camera(fake_bin)
    with pytest.raises(ConnectionError, match="Invalid maximum binning"):
        await cam_bin.open()


@pytest.mark.asyncio
async def test_window_and_binning_validation() -> None:
    fake = FakeAlpacaCameraTransport()
    cam = _make_camera(fake)

    with pytest.raises(ValueError, match="not initialized"):
        await cam.set_window(0, 0, 100, 100)
    with pytest.raises(ValueError, match="not initialized"):
        await cam.set_binning(1, 1)

    await cam.open()

    await cam.set_window(10, 20, 400, 300)
    assert await cam.get_window() == (10, 20, 400, 300)

    with pytest.raises(ValueError, match="non-negative"):
        await cam.set_window(-1, 0, 100, 100)
    with pytest.raises(ValueError, match="positive"):
        await cam.set_window(0, 0, 0, 100)
    with pytest.raises(ValueError, match="exceeds sensor dimensions"):
        await cam.set_window(1000, 0, 1000, 100)

    await cam.set_binning(2, 2)
    assert await cam.get_binning() == (2, 2)

    with pytest.raises(ValueError, match="out of supported range"):
        await cam.set_binning(0, 0)
    with pytest.raises(ValueError, match="out of supported range"):
        await cam.set_binning(5, 5)
    with pytest.raises(ValueError, match="Asymmetric binning"):
        await cam.set_binning(1, 2)


@pytest.mark.asyncio
async def test_gain_and_offset_supported_and_unsupported() -> None:
    fake = FakeAlpacaCameraTransport()
    cam = _make_camera(fake)
    await cam.open()

    await cam.set_gain(250.0)
    assert await cam.get_gain() == 250.0
    assert ("Gain", {"Gain": 250}) in fake.puts

    await cam.set_offset(35.0)
    assert await cam.get_offset() == 35.0
    assert ("Offset", {"Offset": 35}) in fake.puts

    state_calls = [call.args[1] for call in cam.comm.set_state.call_args_list if call.args[0] is IGain]  # type: ignore[attr-defined]
    assert isinstance(state_calls[-1], GainState)
    assert state_calls[-1].gain == 250.0
    assert state_calls[-1].offset == 35.0

    with pytest.raises(ValueError, match="outside supported range"):
        await cam.set_gain(-1.0)
    with pytest.raises(ValueError, match="outside supported range"):
        await cam.set_gain(601.0)
    with pytest.raises(ValueError, match="finite"):
        await cam.set_gain(float("nan"))

    with pytest.raises(ValueError, match="outside supported range"):
        await cam.set_offset(-1.0)
    with pytest.raises(ValueError, match="outside supported range"):
        await cam.set_offset(101.0)
    with pytest.raises(ValueError, match="finite"):
        await cam.set_offset(float("inf"))

    # Discrete list mode (Gains / Offsets arrays when GainMin/Max and OffsetMin/Max raise 1024)
    fake_discrete = FakeAlpacaCameraTransport(
        {
            "GainMin": None,
            "GainMax": None,
            "Gains": ["ISO 100", "ISO 200", "ISO 400", "ISO 800"],
            "Gain": 1,
            "OffsetMin": None,
            "OffsetMax": None,
            "Offsets": ["Low", "Medium", "High"],
            "Offset": 0,
        }
    )
    cam_discrete = _make_camera(fake_discrete)
    await cam_discrete.open()
    assert cam_discrete._meta is not None
    assert cam_discrete._meta.gain_min == 0
    assert cam_discrete._meta.gain_max == 3
    assert cam_discrete._meta.offset_min == 0
    assert cam_discrete._meta.offset_max == 2
    assert await cam_discrete.get_gain() == 1.0
    assert await cam_discrete.get_offset() == 0.0
    await cam_discrete.set_gain(3.0)
    await cam_discrete.set_offset(2.0)
    assert await cam_discrete.get_gain() == 3.0
    assert await cam_discrete.get_offset() == 2.0

    # Unsupported camera (GainMin/Max, Gains, OffsetMin/Max, and Offsets all raise 1024)
    fake_no_gain = FakeAlpacaCameraTransport(
        {"GainMin": None, "GainMax": None, "OffsetMin": None, "OffsetMax": None}
    )
    cam_no_gain = _make_camera(fake_no_gain)
    await cam_no_gain.open()
    assert cam_no_gain._meta is not None
    assert cam_no_gain._meta.gain_min is None
    assert cam_no_gain._meta.offset_min is None

    with pytest.raises(ValueError, match="does not support"):
        await cam_no_gain.get_gain()
    with pytest.raises(ValueError, match="does not support"):
        await cam_no_gain.set_gain(100.0)
    with pytest.raises(ValueError, match="does not support"):
        await cam_no_gain.get_offset()
    with pytest.raises(ValueError, match="does not support"):
        await cam_no_gain.set_offset(10.0)


@pytest.mark.asyncio
async def test_reset_and_full_reset() -> None:
    fake = FakeAlpacaCameraTransport()
    cam = _make_camera(fake, default_gain=100.0, default_offset=15.0)
    await cam.open()
    assert ("Gain", {"Gain": 100}) in fake.puts
    assert ("Offset", {"Offset": 15}) in fake.puts
    assert await cam.get_gain() == 100.0
    assert await cam.get_offset() == 15.0

    await cam.set_exposure_time(5.0)
    await cam.set_image_type(ImageType.DARK)
    await cam.set_binning(2, 2)
    await cam.set_window(10, 20, 200, 100)
    await cam.set_gain(300.0)
    await cam.set_offset(50.0)
    await cam.reset()
    assert cam._exposure_time == 0.0
    assert cam._image_type == ImageType.OBJECT
    assert await cam.get_binning() == (1, 1)
    assert await cam.get_window() == (0, 0, 1936, 1096)
    assert await cam.get_gain() == 100.0
    assert await cam.get_offset() == 15.0

    fake.puts.clear()
    await cam.full_reset()
    assert ("Connected", {"Connected": False}) in fake.puts
    assert ("Connected", {"Connected": True}) in fake.puts

    cam._camera_status = ExposureStatus.EXPOSING
    with pytest.raises(exc.DeviceBusyError):
        await cam.full_reset()


@pytest.mark.asyncio
async def test_expose_applies_binned_roi_and_populates_headers() -> None:
    fake = FakeAlpacaCameraTransport()
    fake.image_ready_sequence = [False, True]
    fake.image_array = np.array([[100, 200, 300, 400], [500, 600, 700, 800]], dtype=np.uint16)
    cam = _make_camera(fake)
    await cam.open()

    await cam.set_binning(2, 2)
    await cam.set_window(20, 40, 8, 4)

    abort_event = asyncio.Event()
    img = await cam._expose(0.02, open_shutter=True, abort_event=abort_event)

    assert ("BinX", {"BinX": 2}) in fake.puts
    assert ("BinY", {"BinY": 2}) in fake.puts
    assert ("StartX", {"StartX": 10}) in fake.puts
    assert ("StartY", {"StartY": 20}) in fake.puts
    assert ("NumX", {"NumX": 4}) in fake.puts
    assert ("NumY", {"NumY": 2}) in fake.puts
    assert ("StartExposure", {"Duration": 0.02, "Light": True}) in fake.puts

    assert img.data.shape == (2, 4)
    assert img.data.dtype == np.uint16
    assert img.header["NAXIS1"] == 4
    assert img.header["NAXIS2"] == 2
    assert img.header["XBINNING"] == 2
    assert img.header["YBINNING"] == 2
    assert img.header["XORGSUBF"] == 20
    assert img.header["YORGSUBF"] == 40
    assert img.header["EXPTIME"] == 0.02
    assert img.header["GAIN"] == 120.0
    assert img.header["OFFSET"] == 10.0
    assert img.header["INSTRUME"] == "ZWO ASI290MM Mini"
    assert img.header["DETECTOR"] == "IMX290"
    assert img.header["DET-PIXL"] == pytest.approx(0.0029)
    assert img.header["DATAMIN"] == 100.0
    assert img.header["DATAMAX"] == 800.0
    assert "BAYERPAT" not in img.header

    # Changing window, binning, gain, or offset mid-exposure must not corrupt the in-flight FITS header
    await cam.set_binning(1, 1)
    await cam.set_window(0, 0, 1936, 1096)
    await cam.set_gain(120.0)
    await cam.set_offset(10.0)
    fake.image_ready_sequence = [False, True]
    expose_task = asyncio.create_task(cam._expose(0.05, open_shutter=True, abort_event=asyncio.Event()))
    await asyncio.sleep(0)
    await cam.set_binning(2, 2)
    await cam.set_window(100, 100, 400, 300)
    await cam.set_gain(300.0)
    await cam.set_offset(50.0)
    inflight_img = await expose_task
    assert inflight_img.header["XBINNING"] == 1
    assert inflight_img.header["YBINNING"] == 1
    assert inflight_img.header["XORGSUBF"] == 0
    assert inflight_img.header["YORGSUBF"] == 0
    assert inflight_img.header["GAIN"] == 120.0
    assert inflight_img.header["OFFSET"] == 10.0


@pytest.mark.asyncio
async def test_expose_dark_frame_rggb_clips_int32_to_uint16() -> None:
    fake = FakeAlpacaCameraTransport(
        {
            "SensorType": 2,
            "BayerOffsetX": 1,
            "BayerOffsetY": 1,
            "Description": "ZWO ASI178MC (3072x2048, 2.40µm) [Go astrocam]",
        }
    )
    fake.image_array = np.array([[-5, 1000], [70000, 42]], dtype=np.int32)
    cam = _make_camera(fake)
    await cam.open()

    img = await cam._expose(0.0, open_shutter=False, abort_event=asyncio.Event())

    assert ("StartExposure", {"Duration": 0.0, "Light": False}) in fake.puts
    assert img.data.dtype == np.uint16
    np.testing.assert_array_equal(img.data, np.array([[0, 1000], [65535, 42]], dtype=np.uint16))
    assert img.header["BAYERPAT"] == "RGGB"
    assert img.header["XBAYROFF"] == 1
    assert img.header["YBAYROFF"] == 1
    assert img.header["INSTRUME"] == "ZWO ASI178MC (3072x2048, 2.40um) [Go astrocam]"

    # Also verify RGGB camera where BayerOffsetX/Y return PropertyNotImplemented (1024) defaults to (0, 0)
    fake_no_off = FakeAlpacaCameraTransport({"SensorType": 2, "BayerOffsetX": None, "BayerOffsetY": None})
    cam_no_off = _make_camera(fake_no_off)
    await cam_no_off.open()
    assert cam_no_off._meta is not None
    assert (cam_no_off._meta.bayer_offset_x, cam_no_off._meta.bayer_offset_y) == (0, 0)


@pytest.mark.asyncio
async def test_expose_aborts_with_abort_or_stop_exposure() -> None:
    fake = FakeAlpacaCameraTransport({"CanAbortExposure": True})
    cam = _make_camera(fake)
    await cam.open()

    abort_event = asyncio.Event()
    abort_event.set()

    with pytest.raises(exc.AbortedError):
        await cam._expose(0.01, open_shutter=True, abort_event=abort_event)
    assert ("AbortExposure", {}) in fake.puts

    fake_stop = FakeAlpacaCameraTransport({"CanAbortExposure": False})
    cam_stop = _make_camera(fake_stop)
    await cam_stop.open()
    with pytest.raises(exc.AbortedError):
        await cam_stop._expose(0.01, open_shutter=True, abort_event=abort_event)
    assert ("StopExposure", {}) in fake_stop.puts

    # Rejected abort must raise GrabImageError from both abort() and _expose() without publishing IDLE
    fake_fail = FakeAlpacaCameraTransport({"CanAbortExposure": True, "ImageReady": False})
    fake_fail.fail_on_put.add("AbortExposure")
    cam_fail = _make_camera(fake_fail)
    await cam_fail.open()
    cam_fail._camera_status = ExposureStatus.EXPOSING

    expose_task = asyncio.create_task(cam_fail._expose(0.5, open_shutter=True, abort_event=cam_fail.expose_abort))
    await asyncio.sleep(0)
    with pytest.raises(exc.GrabImageError, match="Failed to abort exposure"):
        await cam_fail.abort()
    with pytest.raises(exc.GrabImageError, match="Failed to abort exposure"):
        await expose_task
    assert cam_fail._camera_status == ExposureStatus.EXPOSING


@pytest.mark.asyncio
async def test_expose_error_paths() -> None:
    # 1. CameraError state (5)
    fake_err = FakeAlpacaCameraTransport({"ImageReady": False, "CameraState": 5})
    cam_err = _make_camera(fake_err)
    await cam_err.open()
    with pytest.raises(exc.GrabImageError, match="CameraError"):
        await cam_err._expose(0.01, open_shutter=True, abort_event=asyncio.Event())

    # 2. Timeout waiting for ImageReady
    fake_timeout = FakeAlpacaCameraTransport({"ImageReady": False, "CameraState": 2})
    cam_timeout = _make_camera(fake_timeout, poll_interval=0.01, readout_timeout=0.02)
    await cam_timeout.open()
    with pytest.raises(exc.GrabImageError, match="Timed out"):
        await cam_timeout._expose(0.01, open_shutter=True, abort_event=asyncio.Event())
    assert ("AbortExposure", {}) in fake_timeout.puts

    # 3. StartExposure transport failure
    fake_start = FakeAlpacaCameraTransport()
    fake_start.fail_on_put.add("StartExposure")
    cam_start = _make_camera(fake_start)
    await cam_start.open()
    with pytest.raises(exc.GrabImageError, match="Could not start exposure"):
        await cam_start._expose(0.01, open_shutter=True, abort_event=asyncio.Event())

    # 4. ImageBytes download failure
    fake_dl = FakeAlpacaCameraTransport()
    fake_dl.fail_imagebytes = True
    cam_dl = _make_camera(fake_dl)
    await cam_dl.open()
    with pytest.raises(exc.GrabImageError, match="Could not download image array"):
        await cam_dl._expose(0.01, open_shutter=True, abort_event=asyncio.Event())

    # 5. Binned window dimensions <= 0 (e.g. 1x1 window with 2x2 binning)
    fake_roi = FakeAlpacaCameraTransport()
    cam_roi = _make_camera(fake_roi)
    await cam_roi.open()
    await cam_roi.set_binning(2, 2)
    await cam_roi.set_window(0, 0, 1, 1)
    with pytest.raises(exc.GrabImageError, match="Binned window dimensions"):
        await cam_roi._expose(0.01, open_shutter=True, abort_event=asyncio.Event())

    # 6. PUT BinX failure
    fake_bin_err = FakeAlpacaCameraTransport()
    fake_bin_err.fail_on_put.add("BinX")
    cam_bin_err = _make_camera(fake_bin_err)
    await cam_bin_err.open()
    with pytest.raises(exc.GrabImageError, match="Could not configure camera window/binning"):
        await cam_bin_err._expose(0.01, open_shutter=True, abort_event=asyncio.Event())
