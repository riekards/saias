# src/trainer.py

import yaml
import torch
from memory import Memory
from ppo_model import PPOModel, prepare_dataset, ppo_update

class Trainer:
    """Trainer that performs PPO fine-tuning on the collected memory."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.memory = Memory(
            max_size=cfg["memory"]["max_size"],
            storage_path=cfg["memory"]["storage_path"]
        )
        train_cfg = cfg.get("trainer", {})
        self.batch_size = train_cfg.get("batch_size", 32)
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.num_epochs = train_cfg.get("num_epochs", 5)

    def fine_tune(self):
        """Run a simple PPO update over the memory buffer."""
        data = self.memory.buffer
        if not data:
            print("No memory to train on.")
            return

        obs, actions, rewards, vocab = prepare_dataset(data)
        if obs is None:
            print("No dialogue pairs to train on.")
            return

        model = PPOModel(obs_size=obs.size(1), action_size=len(vocab))
        optim = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))

        for _ in range(self.num_epochs):
            ppo_update(model, optim, obs, actions, rewards)

        print(f"PPO training complete on {len(actions)} pairs.")
