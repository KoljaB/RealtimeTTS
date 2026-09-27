import struct
import sys
from types import SimpleNamespace

import pytest

from RealtimeTTS import qwen_cpu_topology as topology
from RealtimeTTS import qwen_server as server


def core_record(mask, level=0, group=0, count=1):
    return struct.pack("<IIBB20xHQH6x", 0, 48, 1, level, count, mask, group)


@pytest.mark.parametrize("masks,allowed,expected", [
    ([3, 12, 48, 192], 255, (5, 80)),
    ([17, 34, 68, 136], 255, (3, 12)),
    ([3, 12, 48, 192], 170, (10, 160)),
    ([3, 12, 48, 192, 768], 1020, (20, 320)),
])
def test_disjoint_physical_cores_respect_smt_and_process_affinity(masks, allowed, expected):
    cores = topology._parse_core_records(b"".join(core_record(mask) for mask in masks))
    assert topology._select_core_masks(cores, allowed, 2, 2) == expected
    assert expected[0] & expected[1] == 0
    assert expected[0].bit_count() == expected[1].bit_count() == 2


def test_hybrid_layout_uses_only_fastest_available_class():
    cores = [(1, 0), (6, 1), (24, 1), (32, 0)]
    assert topology._select_core_masks(cores, 63, 1, 1) == (2, 8)
    with pytest.raises(ValueError, match="only 2"):
        topology._select_core_masks(cores, 63, 2, 1)


@pytest.mark.parametrize("data", [
    b"", b"short", core_record(0), core_record(3) + core_record(2),
    core_record(3, group=1), core_record(3, count=2), core_record(3)[:-1],
    struct.pack("<II", 0, 0), struct.pack("<II", 4, 48) + bytes(40),
])
def test_invalid_or_multigroup_topology_is_rejected(data):
    with pytest.raises(ValueError):
        topology._parse_core_records(data)


@pytest.mark.parametrize("allowed,generation,codec", [(0, 1, 1), (3, 0, 1), (3, 1, 0), (3, 2, 1)])
def test_unavailable_or_insufficient_cores_are_rejected(allowed, generation, codec):
    with pytest.raises(ValueError):
        topology._select_core_masks([(1, 0), (2, 0)], allowed, generation, codec)


def test_core_split_is_opt_in_and_windows_only(monkeypatch):
    assert server.build_argument_parser().parse_args([]).cpu_core_split is False
    monkeypatch.setattr(topology, "os", SimpleNamespace(name="posix"))
    with pytest.raises(ValueError, match="64-bit Windows"):
        topology.split_windows_cpu_cores(2, 2)


def test_cli_passes_discovered_masks_without_changing_priority(monkeypatch):
    calls = []
    def split(generation, codec):
        assert (generation, codec) == (4, 3)
        return 0x55, 0x1500
    class Constructed(Exception):
        pass
    def engine(**kwargs):
        calls.append(kwargs)
        raise Constructed
    monkeypatch.setattr(server, "split_windows_cpu_cores", split)
    monkeypatch.setattr(server, "QwenCpuEngine", engine)
    monkeypatch.setattr(server, "_set_windows_server_priority", lambda: pytest.fail("priority changed"))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())
    with pytest.raises(Constructed):
        server.main(["--device", "cpu", "--cpu-threads", "4",
                     "--cpu-codec-threads", "3", "--cpu-core-split"])
    assert calls[0]["cpu_affinity"] == 0x55
    assert calls[0]["cpu_codec_affinity"] == 0x1500
    assert calls[0]["cpu_threads"] == 4
    assert calls[0]["cpu_codec_threads"] == 3


@pytest.mark.parametrize("flags", [
    ["--device", "gpu"],
    ["--device", "cpu"],
    ["--device", "cpu", "--cpu-threads", "2"],
    ["--device", "cpu", "--cpu-codec-threads", "2"],
    ["--device", "cpu", "--cpu-threads", "2", "--cpu-codec-threads", "2", "--cpu-affinity", "3"],
    ["--device", "cpu", "--cpu-threads", "2", "--cpu-codec-threads", "2", "--cpu-codec-affinity", "12"],
])
def test_invalid_core_split_cli_fails_before_topology_or_model_loading(monkeypatch, flags):
    monkeypatch.setattr(server, "split_windows_cpu_cores", lambda *args: pytest.fail("queried topology"))
    monkeypatch.setattr(server, "QwenCpuEngine", lambda **kwargs: pytest.fail("loaded model"))
    with pytest.raises(SystemExit):
        server.main([*flags, "--cpu-core-split"])


def test_topology_error_is_an_actionable_cli_error(monkeypatch, capsys):
    def unavailable(*args):
        raise ValueError("Only one physical core is available")
    monkeypatch.setattr(server, "split_windows_cpu_cores", unavailable)
    with pytest.raises(SystemExit):
        server.main(["--device", "cpu", "--cpu-threads", "2",
                     "--cpu-codec-threads", "2", "--cpu-core-split"])
    assert "Only one physical core" in capsys.readouterr().err
