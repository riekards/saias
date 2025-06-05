from types import SimpleNamespace
from src.agent import Agent
import requests

class DummyResponse:
    def __init__(self, text="hi"):
        self.status_code = 200
        self._text = text
    def json(self):
        return {"response": self._text}
    @property
    def text(self):
        return self._text


def test_agent_responds(monkeypatch):
    def fake_post(url, json):
        return DummyResponse("pong")
    monkeypatch.setattr(requests, "post", fake_post)
    agent = Agent(config_path="configs/default.yaml")
    resp = agent.respond("Test ping")
    assert resp == "pong"
