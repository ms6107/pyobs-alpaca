"""Unit tests for the pure, non-hardware logic in AlpacaDevice.

Hardware I/O (open/close, talking to the ASCOM server) is out of scope here; these cover
URL construction, version validation, the connection guard, and the connected-state
transition logic.
"""

import struct
from typing import Any

import numpy as np
import pytest

from pyobs_alpaca.device import _IMAGEBYTES_HEADER_FORMAT, AlpacaDevice, _decode_imagebytes

_ARGS = dict(server="myserver", port=11111, device_type="telescope", device=3)


def _pack_imagebytes(
    arr_xy: np.ndarray[Any, Any],
    tx_el_type: int = 8,
    image_el_type: int = 8,
    error_number: int = 0,
    error_msg: str = "",
) -> bytes:
    """Pack a column-major (X, Y[, C]) array or error string into an ASCOM ImageBytes buffer."""
    data_start = 44
    if error_number != 0:
        payload = error_msg.encode("utf-8") + b"\x00"
        rank, dim1, dim2, dim3 = 0, 0, 0, 0
    elif arr_xy.ndim == 2:
        payload = arr_xy.tobytes(order="C")
        rank, dim1, dim2, dim3 = 2, arr_xy.shape[0], arr_xy.shape[1], 0
    else:
        payload = arr_xy.tobytes(order="C")
        rank, dim1, dim2, dim3 = 3, arr_xy.shape[0], arr_xy.shape[1], arr_xy.shape[2]

    header = struct.pack(
        _IMAGEBYTES_HEADER_FORMAT,
        1,  # MetadataVersion
        error_number,
        1,  # ClientTransactionID
        42,  # ServerTransactionID
        data_start,
        image_el_type,
        tx_el_type,
        rank,
        dim1,
        dim2,
        dim3,
    )
    return header + payload


def test_build_alpaca_url() -> None:
    device = AlpacaDevice(**_ARGS)
    assert device._build_alpaca_url("RightAscension") == ("http://myserver:11111/api/v1/telescope/3/rightascension")


def test_only_v1_supported() -> None:
    with pytest.raises(ValueError):
        AlpacaDevice(**_ARGS, version="v2")


def test_starts_disconnected() -> None:
    device = AlpacaDevice(**_ARGS)
    assert device.connected is False


@pytest.mark.asyncio
async def test_get_requires_connection() -> None:
    device = AlpacaDevice(**_ARGS)
    with pytest.raises(ConnectionError):
        await device.get("RightAscension")


@pytest.mark.asyncio
async def test_put_requires_connection() -> None:
    device = AlpacaDevice(**_ARGS)
    with pytest.raises(ConnectionError):
        await device.put("Tracking", Tracking=False)


@pytest.mark.asyncio
async def test_get_imagebytes_requires_connection() -> None:
    device = AlpacaDevice(**_ARGS)
    with pytest.raises(ConnectionError):
        await device.get_imagebytes()


@pytest.mark.asyncio
async def test_check_connected_sets_flag_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    device = AlpacaDevice(**_ARGS)

    async def fake_get(name: str, timeout: float = 5.0) -> Any:
        return True

    monkeypatch.setattr(device, "_get", fake_get)
    await device._check_connected()
    assert device.connected is True


@pytest.mark.asyncio
async def test_check_connected_clears_flag_on_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    device = AlpacaDevice(**_ARGS)
    device._connected = True

    async def fake_get(name: str, timeout: float = 5.0) -> Any:
        raise ConnectionError

    monkeypatch.setattr(device, "_get", fake_get)
    await device._check_connected()
    assert device.connected is False


def test_next_params_increments_transaction_id() -> None:
    device = AlpacaDevice(**_ARGS)
    p1 = device._next_params()
    p2 = device._next_params()
    assert p1["ClientID"] == p2["ClientID"]
    assert int(p1["ClientID"]) > 0
    assert int(p1["ClientTransactionID"]) == 1
    assert int(p2["ClientTransactionID"]) == 2


@pytest.mark.asyncio
async def test_session_reused_and_closed() -> None:
    device = AlpacaDevice(**_ARGS)
    s1 = device._ensure_session()
    s2 = device._ensure_session()
    assert s1 is s2
    assert not s1.closed
    await device.close()
    assert s1.closed
    assert device._session is None


def test_decode_imagebytes_2d_uint16_transposes_to_row_major() -> None:
    # ASCOM sends shape (width=4, height=3) in column-major order
    arr_xy = np.arange(12, dtype="<u2").reshape((4, 3))
    raw = _pack_imagebytes(arr_xy, tx_el_type=8)
    decoded = _decode_imagebytes(raw)
    assert decoded.shape == (3, 4)
    assert decoded.dtype == np.uint16
    assert decoded.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(decoded, arr_xy.T)


def test_decode_imagebytes_2d_int32() -> None:
    arr_xy = np.arange(6, dtype="<i4").reshape((3, 2)) * 100
    raw = _pack_imagebytes(arr_xy, tx_el_type=2)
    decoded = _decode_imagebytes(raw)
    assert decoded.shape == (2, 3)
    assert decoded.dtype == np.int32
    assert decoded.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(decoded, arr_xy.T)


def test_decode_imagebytes_3d_color() -> None:
    arr_xyc = np.arange(24, dtype="<u2").reshape((4, 2, 3))
    raw = _pack_imagebytes(arr_xyc, tx_el_type=8)
    decoded = _decode_imagebytes(raw)
    assert decoded.shape == (2, 4, 3)
    assert decoded.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(decoded, np.transpose(arr_xyc, (1, 0, 2)))


def test_decode_imagebytes_error_number_raises() -> None:
    raw = _pack_imagebytes(np.zeros((2, 2), dtype="<u2"), error_number=1031, error_msg="Camera not connected")
    with pytest.raises(ConnectionError, match="Server error: Camera not connected"):
        _decode_imagebytes(raw)


def test_decode_imagebytes_truncated_raises() -> None:
    with pytest.raises(ConnectionError, match="Truncated ImageBytes response"):
        _decode_imagebytes(b"short")
