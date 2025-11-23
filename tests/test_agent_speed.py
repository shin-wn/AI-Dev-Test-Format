import time

import pytest


@pytest.mark.perf
def test_agent_latency(agent):
    start = time.perf_counter()
    _ = agent.answer("簡単な質問")
    end = time.perf_counter()
    latency = end - start
    print(f"Agent latency: {latency:.4f} seconds")
