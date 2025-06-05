import sys
from src import cli

called = {}

class DummyAgent:
    def __init__(self):
        pass
    def save_memory(self):
        pass
    def respond(self, text):
        return "reply"

class DummyTrainer:
    def __init__(self):
        called['created'] = True
    def fine_tune(self):
        called['fine_tune'] = True


def test_cli_train(monkeypatch):
    monkeypatch.setattr(cli, "Agent", DummyAgent)
    monkeypatch.setattr(cli, "Trainer", DummyTrainer)
    monkeypatch.setattr(sys, "argv", ["prog", "--train"])
    cli.main()
    assert called.get('fine_tune')
