"""Opt-in Windows physical-core placement for the two CPU worker pools."""

from __future__ import annotations

import ctypes
import os
import struct


def _parse_core_records(data: bytes) -> list[tuple[int, int]]:
    """Read 64-bit SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX core records."""
    cores = []
    seen = 0
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError("Truncated Windows CPU topology header")
        relationship, size = struct.unpack_from("<II", data, offset)
        if relationship != 0 or size < 48 or offset + size > len(data):
            raise ValueError("Invalid Windows physical-core topology record")
        efficiency = data[offset + 9]
        group_count = struct.unpack_from("<H", data, offset + 30)[0]
        mask, group = struct.unpack_from("<QH", data, offset + 32)
        if group_count != 1 or group != 0:
            raise ValueError("--cpu-core-split supports one Windows processor group")
        if not mask or mask & seen:
            raise ValueError("Invalid or overlapping Windows physical-core masks")
        cores.append((mask, efficiency))
        seen |= mask
        offset += size
    if not cores:
        raise ValueError("Windows returned no physical CPU cores")
    return cores


def _select_core_masks(
    cores: list[tuple[int, int]],
    allowed_mask: int,
    generation_threads: int,
    codec_threads: int,
) -> tuple[int, int]:
    if generation_threads <= 0 or codec_threads <= 0:
        raise ValueError("--cpu-core-split requires positive explicit worker counts")
    eligible = [(mask & allowed_mask, level) for mask, level in cores if mask & allowed_mask]
    if not eligible:
        raise ValueError("No physical CPU cores are available to this process")
    # Windows defines higher EfficiencyClass values as higher performance.
    fastest = max(level for _, level in eligible)
    bits = sorted(mask & -mask for mask, level in eligible if level == fastest)
    needed = generation_threads + codec_threads
    if len(bits) < needed:
        raise ValueError(
            f"--cpu-core-split needs {needed} physical cores, but only {len(bits)} "
            "are available in the highest performance class; reduce --cpu-threads "
            "and --cpu-codec-threads"
        )
    return sum(bits[:generation_threads]), sum(bits[generation_threads:needed])


def split_windows_cpu_cores(
    generation_threads: int, codec_threads: int
) -> tuple[int, int]:
    """Select one allowed logical processor per physical core, in disjoint pools."""
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError("--cpu-core-split requires 64-bit Windows")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    group_count = kernel32.GetActiveProcessorGroupCount
    group_count.argtypes = []
    group_count.restype = wintypes.WORD
    if group_count() != 1:
        raise ValueError("--cpu-core-split supports one Windows processor group")

    get_topology = kernel32.GetLogicalProcessorInformationEx
    get_topology.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    get_topology.restype = wintypes.BOOL
    length = wintypes.DWORD()
    ctypes.set_last_error(0)
    if get_topology(0, None, ctypes.byref(length)) or ctypes.get_last_error() != 122:
        raise OSError("Cannot query Windows physical-core topology")
    data = ctypes.create_string_buffer(length.value)
    if not get_topology(0, data, ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    cores = _parse_core_records(data.raw[:length.value])

    get_process = kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = wintypes.HANDLE
    get_affinity = kernel32.GetProcessAffinityMask
    get_affinity.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    get_affinity.restype = wintypes.BOOL
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not get_affinity(get_process(), ctypes.byref(allowed), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return _select_core_masks(cores, allowed.value, generation_threads, codec_threads)
