import gym
from gym import spaces
import numpy as np
from agent import Agent
from trainer import Trainer

class SaiasEnv(gym.Env):
    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()
        self.agent = Agent(config_path=config_path)
        self.trainer = Trainer(config_path=config_path)

        # Observation is a fixed-length float vector (e.g. 128 dims)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32
        )

        # Actions: 0=no-op, 1=fine-tune, 2=generate new memory
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.max_steps = 100  # or read from config

    def reset(self):
        self.current_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        reward = 0.0

        if action == 0:
            # no-op: return same observation
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        elif action == 1:
            # fine-tune
            self.trainer.fine_tune()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = +1.0

        elif action == 2:
            # generate new memory
            _ = self.agent.respond("hello?")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = -0.1

        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {}

        # New Gym/Stable-Baselines3 API requires (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"[SaiasEnv] step={self.current_step}")
