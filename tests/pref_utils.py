import contextlib
import time
from typing import Any, Dict, Optional

import psutil
import torch


@contextlib.contextmanager
def measure_gpu_memory(device: torch.device):
    """ピークGPUメモリ使用量を測るコンテキストマネージャ."""
    if device.type != "cuda":
        # CPUならダミー値を返す
        class Dummy:
            used = 0

        yield Dummy()
        return

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    class Result:
        used = 0

    yield Result

    torch.cuda.synchronize(device)
    Result.used = torch.cuda.max_memory_allocated(device)


@contextlib.contextmanager
def measure_cpu_memory():
    """ピークCPUメモリ使用量を測るコンテキストマネージャ."""
    process = psutil.Process()
    start_mem = process.memory_info().rss

    class Result:
        used = 0

    yield Result
    end_mem = process.memory_info().rss
    Result.used = end_mem - start_mem


def benchmark_fn(
    fn,
    *args,
    iters: int = 10,
    warmup: int = 3,
    device: Optional[torch.device] = None,
    measure_memory: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    関数 fn を複数回叩いて平均実行時間(秒), GPU/CPUメモリ使用量, CPU使用率を返す。
    Returns dict: {'time', 'gpu_mem', 'cpu_mem', 'cpu_percent'}
    """
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)

    # warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    gpu_mems = []
    cpu_mems = []
    cpu_percents = []
    process = psutil.Process()

    for _ in range(iters):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)

        cpu_percent_before = psutil.cpu_percent(interval=None)
        mem_before = process.memory_info().rss

        if measure_memory and device is not None:
            with measure_gpu_memory(device) as gpu_mem, measure_cpu_memory() as cpu_mem:
                start = time.perf_counter()
                _ = fn(*args, **kwargs)
                end = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(end - start)
                gpu_mems.append(getattr(gpu_mem, "used", 0))
                cpu_mems.append(getattr(cpu_mem, "used", 0))
        else:
            start = time.perf_counter()
            _ = fn(*args, **kwargs)
            end = time.perf_counter()
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(end - start)
            gpu_mems.append(0)
            cpu_mems.append(process.memory_info().rss - mem_before)

        cpu_percent_after = psutil.cpu_percent(interval=None)
        cpu_percents.append(cpu_percent_after)

    result = {
        "time": sum(times) / len(times) if times else 0,
        "gpu_mem": max(gpu_mems) if gpu_mems else 0,
        "cpu_mem": max(cpu_mems) if cpu_mems else 0,
        "cpu_percent": sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
    }
    return result
