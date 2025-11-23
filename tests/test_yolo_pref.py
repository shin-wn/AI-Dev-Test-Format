import pytest
import torch
from src.models.yolo_like import YoloLikeModel

from .perf_utils import benchmark_fn

pytestmark = pytest.mark.perf


@pytest.fixture(scope="session")
def yolo_model(device):
    model = YoloLikeModel(...).to(device)
    model.eval()
    return model


def _forward(model, batch, device):
    with torch.no_grad():
        return model(batch.to(device))


def test_yolo_throughput(yolo_model, dummy_yolo_batch, device):
    avg_time = benchmark_fn(_forward, yolo_model, dummy_yolo_batch, device=device)
    imgs_per_sec = dummy_yolo_batch.size(0) / avg_time

    # 最低限確保したいスループットを仕様として書いておく
    assert imgs_per_sec >= 50, f"Throughput too low: {imgs_per_sec:.1f} img/s"
