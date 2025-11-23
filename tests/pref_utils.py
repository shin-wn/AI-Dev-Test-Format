import contextlib
import time
from typing import Any, Dict, Optional

import psutil  # process and system utilities
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

    torch.cuda.reset_peak_memory_stats(device)  # GPUのピークメモリ統計リセット
    torch.cuda.empty_cache()  # GPUキャッシュクリア
    torch.cuda.synchronize(device)  # GPUの処理を同期（待機）

    class Result:
        used = 0

    yield Result

    torch.cuda.synchronize(device)  # 再度GPUを同期
    Result.used = torch.cuda.max_memory_allocated(device)  # 実際のピークメモリ使用量を取得して格納


@contextlib.contextmanager
def measure_cpu_memory():
    """ピークCPUメモリ使用量を測るコンテキストマネージャ."""
    process = psutil.Process()  # 現在のプロセス情報を取得
    start_mem = process.memory_info().rss  # 開始時の物理メモリ使用量（RSS）を取得. rss: Resident Set Size.

    class Result:
        used = 0

    yield Result
    end_mem = process.memory_info().rss  # 終了時の物理メモリ使用量を取得
    Result.used = end_mem - start_mem  # 終了時の物理メモリ使用量との差分を計算して格納


def benchmark_fn(
    fn,
    *args,
    warmup: int = 3,
    device: Optional[torch.device] = None,
    measure_memory: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    関数 fn をiters回叩いて平均実行時間(秒), GPU/CPUメモリ使用量, GPUメモリ量, CPU使用率を返す。
    Returns dict: {'time', 'gpu_mem', 'gpu_mem_now', 'cpu_mem', 'cpu_percent'}

    使用例:
    推論:
    ```python
        # ダミーのモデルと入力
        model = torch.nn.Linear(100, 10).cuda()
        input_tensor = torch.randn(32, 100).cuda()

        def inference():
            with torch.no_grad():
                return model(input_tensor)

        # 推論のベンチマーク
        result = benchmark_fn(inference, device=torch.device("cuda"))
        print(result)
    ```
    学習:
    ```python
    import torch
    from tests.pref_utils import benchmark_fn

    model = torch.nn.Linear(100, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    input_tensor = torch.randn(32, 100).cuda()
    target = torch.randn(32, 10).cuda()
    loss_fn = torch.nn.MSELoss()

    def train_step():
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    # 学習ステップのベンチマーク
    result = benchmark_fn(train_step, device=torch.device("cuda"))
    print(result)
    ```
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

    # 現在のGPUメモリ量を取得
    gpu_mem_now = 0
    if device is not None and device.type == "cuda":
        gpu_mem_now = torch.cuda.memory_allocated(device)

    result = {
        "time": sum(times) / len(times) if times else 0,
        "gpu_mem": max(gpu_mems) if gpu_mems else 0,
        "gpu_mem_now": gpu_mem_now,
        "cpu_mem": max(cpu_mems) if cpu_mems else 0,
        "cpu_percent": sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
    }
    return result
