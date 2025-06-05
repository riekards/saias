import os
import yaml
from ollama import Ollama
from memory import Memory

class Agent:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model"]["name"]
        self.device = cfg["model"]["device"]
        self.ollama = Ollama(model_name, device=self.device)
        mem_cfg = cfg["memory"]
        self.memory = Memory(max_size=mem_cfg["max_size"], storage_path=mem_cfg["storage_path"])

    def respond(self, user_input: str) -> str:
        self.memory.add({"role": "user", "text": user_input})
        response = self.ollama.generate(user_input)
        self.memory.add({"role": "assistant", "text": response})
        return response

    def save_memory(self):
        self.memory.save()

if __name__ == "__main__":
    agent = Agent()
    print(agent.respond("Hello, how are you?"))
