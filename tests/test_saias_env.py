import numpy as np
from src.envs import saias_env


class DummyMemory:
    def __init__(self):
        self.buffer = []
        self.max_size = 10


class DummyAgent:
    def __init__(self, config_path=""):
        self.memory = DummyMemory()
        self.respond_called = False

    def respond(self, text):
        self.respond_called = True
        return "ok"


class DummyTrainer:
    def __init__(self, config_path=""):
        self.fine_tune_called = False

    def fine_tune(self):
        self.fine_tune_called = True


def test_saias_env_reset_and_step(monkeypatch):
    monkeypatch.setattr(saias_env, "Agent", DummyAgent)
    monkeypatch.setattr(saias_env, "Trainer", DummyTrainer)

    env = saias_env.SaiasEnv(max_steps=3, state_size=4)
    state, info = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (4,)
    assert info == {}

    # first action: no-op
    state, reward, done, _ = env.step(0)
    assert state.shape == (4,)
    assert reward == 0.0
    assert done is False

    # second action: fine_tune
    state, reward, done, _ = env.step(1)
    assert env.trainer.fine_tune_called
    assert reward == 1.0
    assert done is False

    # third action: respond
    state, reward, done, _ = env.step(2)
    assert env.agent.respond_called
    assert reward == -0.1
    assert done is True

