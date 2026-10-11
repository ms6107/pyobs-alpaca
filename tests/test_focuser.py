from typing import Any

import pytest
from pyobs.utils import exceptions as exc
from pyobs.utils.enums import MotionStatus

from pyobs_alpaca.focuser import AlpacaFocuser

_FOCUSER_ARGS = dict(server="localhost", port=11111, device_type="focuser", device=0)


def test_focuser_default_move_timeout() -> None:
    focuser = AlpacaFocuser(**_FOCUSER_ARGS)
    assert focuser._move_timeout == 60.0


def test_focuser_custom_move_timeout() -> None:
    focuser = AlpacaFocuser(**_FOCUSER_ARGS, move_timeout=30.0)
    assert focuser._move_timeout == 30.0


@pytest.mark.asyncio
async def test_focuser_move_timeout_raises_and_halts() -> None:
    focuser = AlpacaFocuser(**_FOCUSER_ARGS, move_timeout=0.05)

    async def mock_get(prop: str) -> Any:
        if prop == "StepSize":
            return 0.02
        if prop == "Position":
            return 100  # Stalled at 100, target will be 200
        return 0

    halt_called = False

    async def mock_put(action: str, **kwargs: Any) -> None:
        nonlocal halt_called
        if action == "Halt":
            halt_called = True

    focuser._device.get = mock_get  # type: ignore[assignment]
    focuser._device.put = mock_put  # type: ignore[assignment]

    with pytest.raises(exc.MoveError, match="timed out waiting for position"):
        await focuser._set_focus(10.0)

    assert halt_called is True
    assert focuser.motion_status() == MotionStatus.ERROR
    assert not focuser._lock_motion.locked()


@pytest.mark.asyncio
async def test_focuser_move_success() -> None:
    focuser = AlpacaFocuser(**_FOCUSER_ARGS, move_timeout=1.0)

    target_foc = 200

    async def mock_get(prop: str) -> Any:
        if prop == "StepSize":
            return 0.02
        if prop == "Position":
            return target_foc
        return 0

    async def mock_put(action: str, **kwargs: Any) -> None:
        pass

    focuser._device.get = mock_get  # type: ignore[assignment]
    focuser._device.put = mock_put  # type: ignore[assignment]

    await focuser._set_focus(10.0)

    assert focuser.motion_status() == MotionStatus.POSITIONED
    assert not focuser._lock_motion.locked()


@pytest.mark.asyncio
async def test_focuser_move_abort() -> None:
    focuser = AlpacaFocuser(**_FOCUSER_ARGS, move_timeout=1.0)

    async def mock_get(prop: str) -> Any:
        if prop == "StepSize":
            return 0.02
        if prop == "Position":
            # Simulate abort requested while moving
            focuser._abort_motion.set()
            return 0
        return 0

    halt_called = False

    async def mock_put(action: str, **kwargs: Any) -> None:
        nonlocal halt_called
        if action == "Halt":
            halt_called = True

    focuser._device.get = mock_get  # type: ignore[assignment]
    focuser._device.put = mock_put  # type: ignore[assignment]

    with pytest.raises(InterruptedError, match="Setting focus aborted"):
        await focuser._set_focus(10.0)

    assert halt_called is True
    assert focuser.motion_status() == MotionStatus.POSITIONED
    assert not focuser._lock_motion.locked()
