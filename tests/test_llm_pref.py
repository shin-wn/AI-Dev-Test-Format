import pytest
import torch
from src.optim.attention_baseline import BaselineTransformer
from src.optim.attention_flash import FlashTransformer

from .perf_utils import benchmark_fn, measure_gpu_memory

pytestmark = pytest.mark.perf  # このファイル全体を perf テストとしてマーク


@pytest.fixture(scope="session")
def baseline_model(device):
    model = BaselineTransformer(...).to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def flash_model(device):
    model = FlashTransformer(...).to(device)
    model.eval()
    return model


def _forward(model, batch, device):
    with torch.no_grad():
        return model(batch.to(device))


def test_flash_is_faster_and_uses_less_memory(baseline_model, flash_model, dummy_llm_batch, device):
    # 速度比較
    t_base = benchmark_fn(_forward, baseline_model, dummy_llm_batch, device=device)
    t_flash = benchmark_fn(_forward, flash_model, dummy_llm_batch, device=device)

    # GPUメモリ比較
    with measure_gpu_memory(device) as r_base:
        _ = _forward(baseline_model, dummy_llm_batch, device)
    mem_base = r_base.used

    with measure_gpu_memory(device) as r_flash:
        _ = _forward(flash_model, dummy_llm_batch, device)
    mem_flash = r_flash.used

    # テストとしての条件（閾値は好みで調整）
    # flash は 10%以上高速、かつ 5%以上メモリ削減されていること
    assert t_flash <= t_base * 0.90, f"Flash slower: {t_flash:.4f} vs {t_base:.4f}"
    assert mem_flash <= mem_base * 0.95, f"Flash uses more mem: {mem_flash} vs {mem_base}"
