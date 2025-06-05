# src/trainer.py

import yaml
from memory import Memory

class Trainer:
    """
    Stub Trainer: Ollama’s Python client doesn’t currently include a training API.
    """

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
        """
        Placeholder: prints a message if there’s data, or “No memory” otherwise.
        """
        data = self.memory.buffer
        if not data:
            print("No memory to train on.")
            return
        print(f"Trainer stub: received {len(data)} entries, but no OllamaTrainer available.")
