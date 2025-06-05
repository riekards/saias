"""Gym environment wrapper around the SaiAS agent.

This minimal env exposes three discrete actions:

```
0 -> no-op
1 -> call Trainer.fine_tune()
2 -> call Agent.respond("hello?")
```

Observations encode a tiny bit of environment state: the current memory size
and the normalized step count (``current_step / max_steps``). The rest of the
vector is padded with zeros. This environment is intentionally simple and only
exists so that reinforcement-learning scaffolding (e.g. Stable-Baselines3) can
interact with the :class:`Agent` and :class:`Trainer` APIs in the examples and
tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.agent import Agent
from src.trainer import Trainer


class SaiasEnv(gym.Env):
    """Tiny environment driving :class:`Agent` and :class:`Trainer`."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        max_steps: int = 100,
        state_size: int = 128,
    ) -> None:
        super().__init__()

        self.agent = Agent(config_path=config_path)
        self.trainer = Trainer(config_path=config_path)
        # share the same memory buffer between agent and trainer so that
        # dialogue collected by the agent is available for fine‑tuning
        self.trainer.memory = self.agent.memory

        self.state_size = state_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.max_steps = max_steps
        self.current_step = 0
        self.state = np.zeros(self.state_size, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return the current observation vector."""
        obs = np.zeros(self.state_size, dtype=np.float32)
        obs[0] = len(self.trainer.memory.buffer)
        obs[1] = self.current_step / self.max_steps
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._get_obs()
        return self.state, {}

    def step(self, action: int):
        reward = 0.0

        if action == 1:
            # fine-tune the underlying model. Make sure at least one dialogue
            # pair exists in memory so the trainer has something to learn from.
            if len(self.trainer.memory.buffer) < 2:
                _ = self.agent.respond("hello?")
            self.trainer.fine_tune()
            reward = 1.0
        elif action == 2:
            # generate a dummy response and store the pair so that future
            # fine‑tunes have data available
            resp = self.agent.respond("hello?")
            if self.trainer.memory is not self.agent.memory:
                self.trainer.memory.add({"role": "user", "text": "hello?"})
                self.trainer.memory.add({"role": "assistant", "text": resp})
            reward = -0.1

        self.current_step += 1
        self.state = self._get_obs()
        terminated = self.current_step >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:  # pragma: no cover - simple I/O
        print(f"[SaiasEnv] step={self.current_step}")

