import os
import json

class Memory:
    def __init__(self, max_size: int = 10000, storage_path: str = "./memory_store"):
        self.max_size = max_size
        self.storage_path = storage_path
        self.buffer = []
        os.makedirs(self.storage_path, exist_ok=True)

    def add(self, entry: dict):
        """
        Entry format: {'role': 'user'|'assistant', 'text': str}
        """
        self.buffer.append(entry)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def save(self):
        """
        Dump entire buffer to a JSON file (timestamped).
        """
        fname = f"memory_{len(self.buffer)}.json"
        full_path = os.path.join(self.storage_path, fname)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.buffer, f, indent=2)

