import pytest
import torch


@pytest.fixture(scope="session")  # セッション全体で共有
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)  # すべてのテストで自動的に有効化
def set_seed():
    import random

    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture  # テスト関数の引数に指定して使用
def dummy_llm_batch():
    # テキトウなトークンIDのバッチ（seq=128, batch=4）
    B, T, vocab = 4, 128, 32000
    return torch.randint(0, vocab, (B, T), dtype=torch.long)


@pytest.fixture  # テスト関数の引数に指定して使用
def dummy_yolo_batch():
    # バッチ4、3x640x640の画像
    return torch.randn(4, 3, 640, 640)
