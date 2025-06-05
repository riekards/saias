# src/agent.py

import yaml
import requests
from memory import Memory

class Agent:
    """
    Connects to Ollama’s local server on port 11435 (/api/generate),
    storing conversation history in a JSON memory buffer,
    and exposing a .respond(prompt) → str interface.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        # 1) Load YAML config
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 2) Read model settings (e.g., "mistral:latest")
        self.model_name = cfg["model"]["name"]
        self.device = cfg["model"]["device"]

        # 3) Point at Ollama’s /api/generate endpoint on port 11435
        self.ollama_url = "http://localhost:11435/api/generate"

        # 4) Initialize memory buffer
        mem_cfg = cfg["memory"]
        self.memory = Memory(
            max_size=mem_cfg["max_size"],
            storage_path=mem_cfg["storage_path"],
        )

    def respond(self, user_input: str) -> str:
        """
        - Save the user’s prompt in memory.
        - POST to Ollama’s /api/generate endpoint with stream=false.
        - Save the assistant’s reply in memory.
        - Return the generated text.
        """
        # 1) Store the user’s query
        self.memory.add({"role": "user", "text": user_input})

        # 2) Build the JSON payload. Setting "stream": false forces Ollama to
        #    return a single JSON object rather than NDJSON chunks.
        payload = {
            "model": self.model_name,
            "prompt": user_input,
            "stream": False
        }

        # 3) Send the HTTP POST
        resp = requests.post(self.ollama_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API returned {resp.status_code}: {resp.text}")

        # 4) Parse the JSON response. Ollama will return something like:
        #    {
        #      "model": "mistral:latest",
        #      "created_at": "2025-06-05T07:29:08.5243882Z",
        #      "response": "Hi there! How can I help?",
        #      "usage": { … }
        #    }
        data = resp.json()
        response_text = data.get("response", "").strip()

        # 5) Store the assistant’s reply
        self.memory.add({"role": "assistant", "text": response_text})
        return response_text

    def save_memory(self):
        """Dump the conversation buffer to a timestamped JSON file."""
        self.memory.save()


if __name__ == "__main__":
    # Quick sanity check: make sure the /api/generate endpoint replies
    agent = Agent()
    print("AI replied:", agent.respond("Hello, how are you?"))
