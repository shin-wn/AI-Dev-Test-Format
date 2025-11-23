import pytest
from src.agents.my_agent import MyAgent


class FakeLLM:
    """テスト用に決め打ち応答を返すLLM."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "現在の天気" in prompt:
            # ツール呼び出しを指示したかのようなフォーマットを返す…など
            return "TOOL_CALL:weather"
        return "Hello"


@pytest.fixture
def fake_tools():
    class ToolBox:
        def weather(self, location: str):
            return "晴れ"

    return ToolBox()


@pytest.fixture
def agent(fake_tools):
    return MyAgent(llm_client=FakeLLM(), tools=fake_tools)


def test_agent_uses_weather_tool(agent):
    ans = agent.answer("東京の現在の天気を教えて")
    assert "晴れ" in ans
