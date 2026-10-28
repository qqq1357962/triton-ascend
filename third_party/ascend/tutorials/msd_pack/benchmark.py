"""Graph comparison against the original masked component; no model claim."""

import argparse
import random
import statistics

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from .pack import pack_msd


@triton.jit
def _masked_reference(X, Y, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + row * (K // 2) + 2 * offsets, offsets < K // 4, 0)
    b = tl.load(X + row * (K // 2) + 2 * offsets + 1, offsets < K // 4, 0)
    hi = ((a >> 4) & 15) | ((a >> 8) & 240) | ((b << 4) & 3840) | (b & 61440)
    lo = ((a & 15) | ((a >> 4) & 240) | ((b << 8) & 3840) | ((b << 4) & 61440)) ^ 34952
    tl.store(Y + row * (K // 2) + offsets, hi, offsets < K // 4)
    tl.store(Y + row * (K // 2) + K // 4 + offsets, lo, offsets < K // 4)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 256])
    parser.add_argument("--k", type=int, nargs="+", default=[4096, 8196])
    args = parser.parse_args()
    torch.npu.set_device(0)
    print(f"device={torch.npu.get_device_name(0)} torch={torch.__version__} triton={triton.__version__}")
    print("M,K,BLOCK,masked_us,specialized_us,speedup")
    for rows in args.rows:
        for k in args.k:
            if rows <= 0 or k <= 0 or k % 4:
                raise ValueError("positive M and positive K divisible by four are required")
            x = torch.randint(0, 256, (rows, k), dtype=torch.uint8, device="npu")
            out = torch.empty_like(x)
            x16, y16 = x.view(torch.uint16), out.view(torch.uint16)
            calls = {
                "masked": lambda: _masked_reference[(rows, triton.cdiv(k // 4, 1024))](
                    x16, y16, k, 1024, num_warps=4),
                "specialized": lambda: pack_msd(x, out=out),
            }
            calls["masked"]()
            expected = out.clone()
            calls["specialized"]()
            assert torch.equal(out, expected), "masked and specialized outputs differ"
            graphs = {}
            for name, call in calls.items():
                for _ in range(3):
                    call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(100):
                        call()
                graphs[name] = graph
            values = {name: [] for name in graphs}
            rng = random.Random(rows + k)
            for _ in range(7):
                order = list(graphs)
                rng.shuffle(order)
                for name in order + list(reversed(order)):
                    graphs[name].replay()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    values[name].append(start.elapsed_time(end) * 1000 / (100 * 10))
            baseline, optimized = (statistics.median(values[name]) for name in calls)
            print(f"{rows},{k},1024,{baseline:.4f},{optimized:.4f},{baseline / optimized:.3f}", flush=True)


if __name__ == "__main__":
    main()
