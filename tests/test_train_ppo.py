import os
import gym

import train_ppo


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=float)

    def reset(self, seed=None, options=None):
        return [0.0], {}

    def step(self, action):
        return [0.0], 0.0, True, {}


class DummyModel:
    def __init__(self, policy, env, **kwargs):
        DummyModel.init_called = True

    def learn(self, total_timesteps):
        DummyModel.learn_called = total_timesteps

    def save(self, path):
        DummyModel.save_path = path


def test_train_ppo_invoked(monkeypatch, tmp_path):
    monkeypatch.setattr(train_ppo, "SaiasEnv", lambda config_path="configs/default.yaml": DummyEnv())
    monkeypatch.setattr(train_ppo, "PPO", DummyModel)
    monkeypatch.chdir(tmp_path)

    train_ppo.main()

    assert getattr(DummyModel, "init_called", False)
    assert getattr(DummyModel, "learn_called", False)
    assert getattr(DummyModel, "save_path", None) is not None

