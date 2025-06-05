import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.agent import Agent
from src.trainer import Trainer

class SaiasEnv(gym.Env):
    """Simple Gymnasium environment for controlling the SaiAS agent."""

    def __init__(self, config_path: str = "configs/default.yaml", max_steps: int = 100, state_size: int = 128):
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
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

        # New Gym/Stable-Baselines3 API requires (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"[SaiasEnv] step={self.current_step}")
