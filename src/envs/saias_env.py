import gym
import numpy as np

from ..agent import Agent
from ..trainer import Trainer


class SaiasEnv(gym.Env):
    """Simple gym environment for controlling the SaiAS agent."""

    def __init__(self, config_path: str = "configs/default.yaml", max_steps: int = 100, state_size: int = 128):
        super().__init__()
        self.agent = Agent(config_path)
        self.trainer = Trainer(config_path)
        self.max_steps = max_steps
        self.state_size = state_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)
        self.current_step = 0
        self.state = self._get_state()

    def _get_state(self) -> np.ndarray:
        """Return a feature vector derived from the agent's memory."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        if self.agent.memory.buffer:
            vec[0] = len(self.agent.memory.buffer) / float(self.agent.memory.max_size)
        return vec

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.agent.memory.buffer.clear()
        self.state = np.zeros(self.state_size, dtype=np.float32)
        return self.state, {}

    def step(self, action: int):
        reward = 0.0
        if action == 0:
            # no-op
            pass
        elif action == 1:
            self.trainer.fine_tune()
            reward = 1.0
        elif action == 2:
            self.agent.respond("Generate a helpful response.")
            reward = -0.1
        else:
            raise ValueError(f"Invalid action {action}")

        self.state = self._get_state()
        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {}
        return self.state, reward, done, info

    def render(self):
        print(f"Step: {self.current_step}, State[0]: {self.state[0]:.3f}")
