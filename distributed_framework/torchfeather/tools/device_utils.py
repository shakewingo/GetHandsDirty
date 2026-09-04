import contextlib
from typing import Any, Protocol, cast

import torch
from torch._utils import _get_available_device_type, _get_device_module


class DeviceModule(Protocol):
    def set_device(self, device: torch.device | int) -> None: ...
    def current_device(self) -> int: ...
    def synchronize(self) -> None: ...
    def get_device_name(self, device: torch.device | int) -> str: ...
    def get_device_properties(self, device: torch.device | int) -> Any: ...
    def reset_peak_memory_stats(self) -> None: ...
    def empty_cache(self) -> None: ...
    def memory_stats(self, device: torch.device | int) -> dict[str, Any]: ...


class _CpuDeviceModule:
    """No-op stand-in for the accelerator memory API, so the trainer runs on a CPU-only box.

    `torch.cpu` implements set_device/current_device/synchronize but none of the memory
    introspection the metrics processor calls. Reporting zeros keeps the local dry run -- the
    gate before renting anything -- from needing a GPU.
    """

    def set_device(self, device): pass
    def current_device(self) -> int: return 0
    def synchronize(self) -> None: pass
    def get_device_name(self, device=None) -> str: return "cpu"
    def get_device_properties(self, device=None) -> Any:
        return type("Props", (), {"total_memory": 0, "name": "cpu"})()
    def reset_peak_memory_stats(self) -> None: pass
    def empty_cache(self) -> None: pass
    def memory_stats(self, device=None) -> dict[str, Any]: return {}


def get_device_info() -> tuple[str, DeviceModule]:
    device_type = _get_available_device_type()
    if device_type is None:
        # No accelerator. Claiming "cuda" here makes every downstream call fail obscurely.
        return "cpu", cast(DeviceModule, _CpuDeviceModule())
    device_module = cast(DeviceModule, _get_device_module(device_type))
    return device_type, device_module


device_type, device_module = get_device_info()


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
