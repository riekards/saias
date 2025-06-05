import pytest
from src.agent import Agent

def test_agent_responds():
    agent = Agent(config_path="configs/default.yaml")
    resp = agent.respond("Test ping")
    assert isinstance(resp, str)
    assert len(resp) > 0
