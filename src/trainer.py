import yaml
from ollama import Ollama, OllamaTrainer
from memory import Memory

class Trainer:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model"]["name"]
        self.trainer = OllamaTrainer(model_name)
        self.memory = Memory(
            max_size=cfg["memory"]["max_size"],
            storage_path=cfg["memory"]["storage_path"],
        )
        train_cfg = cfg["trainer"]
        self.batch_size = train_cfg["batch_size"]
        self.learning_rate = train_cfg["learning_rate"]
        self.num_epochs = train_cfg["num_epochs"]

    def fine_tune(self):
        """
        Example: collect memory, train, and save updated model.
        """
        data = self.memory.buffer
        if not data:
            print("No memory to train on.")
            return
        inputs = [entry["text"] for entry in data if entry["role"] == "user"]
        targets = [entry["text"] for entry in data if entry["role"] == "assistant"]
        self.trainer.train(
            inputs=inputs,
            targets=targets,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.num_epochs,
        )
        self.trainer.save(f"{model_name}_fine_tuned.pt")

