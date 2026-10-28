"""Run on an Ascend NPU: python -m pytest third_party/ascend/unittest/pytest_ut/test_msd_pack_full_tile.py -q."""

import numpy as np
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("triton")

# Load the example by path so tests also work from the unittest directory.
import importlib.util
from pathlib import Path
import sys

_example = Path(__file__).resolve().parents[2] / "tutorials/msd_pack/pack.py"
_spec = importlib.util.spec_from_file_location("msd_pack_full_tile_example", _example)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
pack_msd = _module.pack_msd


def reference(x):
    # Independent byte-level oracle; the kernel uses paired uint16 carriers.
    pairs = x.numpy().astype(np.uint16).reshape(x.shape[0], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    hi = (a >> 4) | (b & 240)
    lo = ((a & 15) | ((b & 15) << 4)) ^ 136
    return np.concatenate((hi, lo), axis=1).astype(np.uint8)


@pytest.mark.parametrize("rows", [1, 8, 49, 256])
@pytest.mark.parametrize("k,block", [(64, 16), (128, 16), (132, 16), (4092, 1024),
                                     (4096, 1024), (4100, 1024), (8192, 1024),
                                     (8196, 1024)])
def test_full_and_tail_tiles(rows, k, block):
    rng = np.random.default_rng(rows + k)
    cpu = torch.from_numpy(rng.integers(0, 256, (rows, k), dtype=np.uint8))
    x = cpu.npu()
    # Poison bytes around the output view detect writes past a short tail.
    backing = torch.full((rows * k + 64,), 85, dtype=torch.uint8, device=x.device)
    out = backing[32:-32].reshape(rows, k)
    assert pack_msd(x, out=out, block_size=block) is out
    np.testing.assert_array_equal(out.cpu().numpy(), reference(cpu))
    np.testing.assert_array_equal(backing[:32].cpu().numpy(), np.full(32, 85))
    np.testing.assert_array_equal(backing[-32:].cpu().numpy(), np.full(32, 85))


@pytest.mark.parametrize("k", [4096, 4100, 8196])
def test_mutable_graph(k):
    cpu = torch.zeros((8, k), dtype=torch.uint8)
    x = cpu.npu()
    out = torch.empty_like(x)
    for _ in range(3):
        pack_msd(x, out=out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        pack_msd(x, out=out)
    for byte in (0, 255, 128, 17):
        cpu.fill_(byte)
        x.copy_(cpu)
        out.fill_(85)
        graph.replay()
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), reference(cpu))


def test_all_uint16_codes():
    # All 65536 single-carrier codes, not every possible pair of carriers.
    cpu = torch.from_numpy(np.arange(65536, dtype=np.uint16).view(np.uint8).copy()).reshape(32, 4096)
    np.testing.assert_array_equal(pack_msd(cpu.npu()).cpu().numpy(), reference(cpu))


def test_nonzero_aligned_offsets():
    cpu = torch.from_numpy(np.arange(8 * 4100, dtype=np.uint16).astype(np.uint8)).reshape(8, 4100)
    x_storage = torch.empty((cpu.numel() + 128,), dtype=torch.uint8, device="npu")
    x = x_storage[64:-64].reshape(cpu.shape)
    x.copy_(cpu)
    y_storage = torch.full_like(x_storage, 85)
    out = y_storage[64:-64].reshape(cpu.shape)
    pack_msd(x, out=out)
    np.testing.assert_array_equal(out.cpu().numpy(), reference(cpu))
    np.testing.assert_array_equal(y_storage[:64].cpu().numpy(), np.full(64, 85))
    np.testing.assert_array_equal(y_storage[-64:].cpu().numpy(), np.full(64, 85))


def test_invalid_format_and_alias():
    x = torch.zeros((1, 64), dtype=torch.uint8, device="npu")
    with pytest.raises(ValueError, match="share storage"):
        pack_msd(x, out=x)
    with pytest.raises(ValueError, match="divisible by four"):
        pack_msd(x[:, :62])
    with pytest.raises(ValueError, match="power of two"):
        pack_msd(x, block_size=48)
    with pytest.raises(ValueError, match="two-byte-aligned"):
        pack_msd(torch.zeros(65, dtype=torch.uint8, device="npu")[1:].reshape(1, 64))


@pytest.mark.parametrize("shape", [(0, 4096), (8, 0)])
def test_empty(shape):
    x = torch.empty(shape, dtype=torch.uint8, device="npu")
    assert pack_msd(x).shape == x.shape
