import time

import pytest


@pytest.mark.perf
def test_agent_latency_under_threshold(agent):
    start = time.perf_counter()
    _ = agent.answer("簡単な質問")
    end = time.perf_counter()
    latency = end - start
    assert latency < 0.5  # 0.5秒以内とか、仕様として決める
