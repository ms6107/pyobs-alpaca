import asyncio
import logging
import os
import struct
from typing import Any, NamedTuple

import aiohttp
import numpy as np
from numpy.typing import NDArray
from pyobs.object import Object

log = logging.getLogger("pyobs")

_IMAGEBYTES_HEADER_FORMAT = "<iiIIiiiiiii"
_IMAGEBYTES_HEADER_SIZE = struct.calcsize(_IMAGEBYTES_HEADER_FORMAT)

_IMAGEBYTES_DTYPES: dict[int, np.dtype[Any]] = {
    1: np.dtype("<i2"),  # Int16
    2: np.dtype("<i4"),  # Int32
    3: np.dtype("<f8"),  # Double
    4: np.dtype("<f4"),  # Single
    5: np.dtype("<u8"),  # UInt64
    6: np.dtype("<u1"),  # Byte
    7: np.dtype("<i8"),  # Int64
    8: np.dtype("<u2"),  # UInt16
    9: np.dtype("<u4"),  # UInt32
}


class ServerPutResponse(NamedTuple):
    ClientTransactionID: int
    ErrorMessage: str
    ErrorNumber: int
    ServerTransactionID: int
    Value: Any = None


class ServerGetResponse(NamedTuple):
    ClientTransactionID: int
    ErrorMessage: str
    ErrorNumber: int
    ServerTransactionID: int
    Value: Any = None


DEVICE_INIT_KWARGS = {"server", "port", "device_type", "device", "version", "alive_parameter"}
"""Names of AlpacaDevice's own constructor params -- callers that also thread **kwargs through
their own cooperative super().__init__() chain must exclude these first, since nothing in that
chain (Module/mixins) recognizes them and they'd otherwise leak through to object.__init__()."""

OBJECT_SHARED_KWARGS = {"vfs", "comm", "timezone", "location", "observer"}
"""Names of pyobs.object.Object's own constructor params. AlpacaDevice forwards anything beyond
its own params (see DEVICE_INIT_KWARGS) straight to Object cooperatively, so callers building an
AlpacaDevice from a module's **kwargs must restrict to DEVICE_INIT_KWARGS | OBJECT_SHARED_KWARGS --
any module/mixin-specific kwarg (e.g. fits_namespaces, motion_status_interfaces) would otherwise
leak through to object.__init__() the same way."""


def _decode_imagebytes(raw: bytes) -> NDArray[Any]:
    """Decode an ASCOM Alpaca binary ImageBytes payload into a C-contiguous NumPy array.

    ASCOM transmits 2D arrays in column-major (X, Y) order; this function transposes the
    spatial axes so the returned array has standard FITS/pyobs row-major (height, width) shape.
    """
    if len(raw) < _IMAGEBYTES_HEADER_SIZE:
        raise ConnectionError(
            f"Truncated ImageBytes response: expected at least {_IMAGEBYTES_HEADER_SIZE} bytes, got {len(raw)}."
        )

    (
        _meta_version,
        error_number,
        _client_tx,
        _server_tx,
        data_start,
        _image_el_type,
        tx_el_type,
        rank,
        dim1,
        dim2,
        dim3,
    ) = struct.unpack_from(_IMAGEBYTES_HEADER_FORMAT, raw, 0)

    if data_start < _IMAGEBYTES_HEADER_SIZE or data_start > len(raw):
        raise ConnectionError(f"Invalid ImageBytes DataStart offset: {data_start} (payload size {len(raw)}).")

    if error_number != 0:
        err_msg = raw[data_start:].decode("utf-8", errors="replace").rstrip("\x00")
        raise ConnectionError(f"Server error: {err_msg}")

    dtype = _IMAGEBYTES_DTYPES.get(tx_el_type)
    if dtype is None:
        raise ConnectionError(f"Unsupported ImageBytes TransmissionElementType: {tx_el_type}.")

    if rank == 2:
        if dim1 <= 0 or dim2 <= 0:
            raise ConnectionError(f"Invalid ImageBytes 2D dimensions: ({dim1}, {dim2}).")
        count = dim1 * dim2
        expected_bytes = data_start + count * dtype.itemsize
        if len(raw) < expected_bytes:
            raise ConnectionError(f"Truncated ImageBytes 2D payload: expected {expected_bytes} bytes, got {len(raw)}.")
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=data_start).reshape((dim1, dim2))
        return np.ascontiguousarray(arr.T)

    if rank == 3:
        if dim1 <= 0 or dim2 <= 0 or dim3 <= 0:
            raise ConnectionError(f"Invalid ImageBytes 3D dimensions: ({dim1}, {dim2}, {dim3}).")
        count = dim1 * dim2 * dim3
        expected_bytes = data_start + count * dtype.itemsize
        if len(raw) < expected_bytes:
            raise ConnectionError(f"Truncated ImageBytes 3D payload: expected {expected_bytes} bytes, got {len(raw)}.")
        arr = np.frombuffer(raw, dtype=dtype, count=count, offset=data_start).reshape((dim1, dim2, dim3))
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))

    raise ConnectionError(f"Unsupported ImageBytes Rank: {rank}.")


class AlpacaDevice(Object):
    def __init__(
        self,
        server: str,
        port: int,
        device_type: str,
        device: int,
        version: str = "v1",
        alive_parameter: str = "Connected",
        **kwargs: Any,
    ):
        """Initializes a new ASCOM Alpaca device.

        Args:
            server: Name or IP of Alpaca remote server.
            port: Port of Alpaca remote server
            type: Type of device.
            device: Device number.
            version: Alpaca version.
            alive_parameter: Name of parameter to request in alive ping.
        """
        Object.__init__(self, **kwargs)

        # variables
        self._server = server
        self._port = port
        self._type = device_type
        self._device = device
        self._version = version
        self._alive_param = alive_parameter

        # Alpaca transaction tracking and persistent HTTP session
        self._client_id: int = (os.getpid() & 0x7FFFFFFF) or 1
        self._tx_id: int = 0
        self._session: aiohttp.ClientSession | None = None

        # do we have a connection to the ASCOM Remote server?
        self._connected = False

        # add thread
        self.add_background_task(self._check_connected_thread)

        # check version
        if version != "v1":
            raise ValueError("Only Alpaca v1 is supported.")

    @property
    def connected(self) -> bool:
        return self._connected

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Return an open aiohttp.ClientSession, creating one lazily if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _next_params(self) -> dict[str, str]:
        """Return ClientID and monotonically increasing ClientTransactionID strings."""
        self._tx_id = (self._tx_id % 0xFFFFFFFF) + 1
        return {
            "ClientID": str(self._client_id),
            "ClientTransactionID": str(self._tx_id),
        }

    async def open(self) -> None:
        """Open device."""
        await Object.open(self)
        self._ensure_session()

        # check connected
        await self._check_connected()
        if not self._connected:
            log.warning("Could not connect to ASCOM server.")

    async def close(self) -> None:
        """Close device and release HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        await Object.close(self)

    async def _check_connected_thread(self) -> None:
        """Periodically check, whether we're connected to ASCOM."""
        while True:
            await self._check_connected()
            await asyncio.sleep(5)

    async def _check_connected(self) -> None:
        """Check, whether we're connected to ASCOM"""

        # get new status
        try:
            await self._get(self._alive_param)
            connected = True
        except ConnectionError:
            connected = False

        # did it change?
        if connected != self._connected:
            if connected:
                log.info("Connected to ASCOM server.")
            else:
                log.warning("Lost connection to ASCOM server.")

        # store new status
        self._connected = connected

    def _build_alpaca_url(self, name: str) -> str:
        """Build URL for Alpaca server.

        Args:
            name: Name of Alpaca variable

        Returns:
            Full Alpaca URL
        """
        return f"http://{self._server}:{self._port}/api/{self._version}/{self._type}/{self._device}/{name.lower()}"

    async def _get(self, name: str, timeout: float = 5.0) -> Any:
        """Calls GET on Alpaca server, which returns value for variable with given name.

        Args:
            name: Name of variable.
            timeout: Time in sec for request.

        Returns:
            Value of variable.

        Raises:
            ConnectionError: If an error occurred.
        """

        # get url
        url = self._build_alpaca_url(name)
        session = self._ensure_session()

        # request it
        try:
            async with session.get(
                url,
                params=self._next_params(),
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"ALPACA server responded with error {response.status}: {await response.text()}."
                    )
                json = await response.json()
                resp = ServerGetResponse(**json)

        except TimeoutError:
            # raise a ConnectionError instead
            raise ConnectionError("Connection to ALPACA server timed out.")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection to ALPACA server failed: {e}")

        # check error
        if resp.ErrorNumber != 0:
            raise ConnectionError(f"Server error: {resp.ErrorMessage}")

        # return value
        return resp.Value

    async def get(self, name: str, timeout: float = 5.0) -> Any:
        """Calls GET on Alpaca server, which returns value for variable with given name.

        Args:
            name: Name of variable.
            timeout: Time in sec for request.

        Returns:
            Value of variable.

        Raises:
            ConnectionError: If an error occurred.
        """

        # only do it, if connected
        if not self._connected:
            raise ConnectionError("Not connected to ASCOM.")
        return await self._get(name, timeout=timeout)

    async def _get_imagebytes(self, name: str = "imagearray", timeout: float = 30.0) -> NDArray[Any]:
        """Calls GET on Alpaca server with Accept: application/imagebytes and decodes the image array.

        Falls back to standard JSON decoding if the server responds with application/json.

        Args:
            name: Name of variable (default: "imagearray").
            timeout: Time in sec for request.

        Returns:
            2D (height, width) or 3D (height, width, planes) C-contiguous NumPy array.

        Raises:
            ConnectionError: If an error occurred.
        """
        url = self._build_alpaca_url(name)
        session = self._ensure_session()

        try:
            async with session.get(
                url,
                params=self._next_params(),
                headers={"Accept": "application/imagebytes"},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"ALPACA server responded with error {response.status}: {await response.text()}."
                    )
                content_type = response.headers.get("Content-Type", "").lower()
                if "application/imagebytes" in content_type:
                    raw = await response.read()
                    return _decode_imagebytes(raw)

                json = await response.json()
                resp = ServerGetResponse(**json)
        except TimeoutError:
            raise ConnectionError("Connection to ALPACA server timed out.")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection to ALPACA server failed: {e}")

        if resp.ErrorNumber != 0:
            raise ConnectionError(f"Server error: {resp.ErrorMessage}")

        arr = np.asarray(resp.Value)
        if arr.ndim == 2:
            return np.ascontiguousarray(arr.T)
        if arr.ndim == 3:
            return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))
        raise ConnectionError(f"Unsupported JSON image array rank: {arr.ndim}.")

    async def get_imagebytes(self, name: str = "imagearray", timeout: float = 30.0) -> NDArray[Any]:
        """Calls GET on Alpaca server for an image array using binary ImageBytes transfer.

        Args:
            name: Name of variable (default: "imagearray").
            timeout: Time in sec for request.

        Returns:
            2D (height, width) or 3D (height, width, planes) C-contiguous NumPy array.

        Raises:
            ConnectionError: If not connected or an error occurred.
        """
        if not self._connected:
            raise ConnectionError("Not connected to ASCOM.")
        return await self._get_imagebytes(name, timeout=timeout)

    async def put(self, name: str, timeout: float = 5.0, **values: Any) -> None:
        """Calls PUT on Alpaca server with given variable, which might set a variable or call a method.

        Args:
            name: Name of variable.
            timeout: Time in sec for request.
            values: Values to set.

        Raises:
            ConnectionError: If an error occurred.
        """

        # only do it, if connected
        if not self._connected:
            raise ConnectionError("Not connected to ASCOM.")

        # get url
        url = self._build_alpaca_url(name)
        session = self._ensure_session()
        payload: dict[str, Any] = dict(self._next_params())
        for k, v in values.items():
            if isinstance(v, bool):
                payload[k] = "true" if v else "false"
            else:
                payload[k] = str(v)

        # request it
        try:
            async with session.put(url, data=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"ALPACA server responded with error {response.status}: {await response.text()}."
                    )
                json = await response.json()
                resp = ServerPutResponse(**json)

        except TimeoutError:
            # raise a ConnectionError instead
            raise ConnectionError("Connection to ALPACA server timed out.")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Connection to ALPACA server failed: {e}")

        # check error
        if resp.ErrorNumber != 0:
            raise ConnectionError(f"Server error: {resp.ErrorMessage}")


__all__ = ["AlpacaDevice"]
