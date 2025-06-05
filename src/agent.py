import yaml
from ollama import generate
from memory import Memory

class Agent:
    """
    Loads a local Ollama model, stores conversations in a simple JSON memory buffer,
    and provides a .respond(prompt) → str interface.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        # 1) Load YAML config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # 2) Read model name + device (device is unused here but kept for future GPU support)
        self.model_name = cfg["model"]["name"]
        self.device = cfg["model"]["device"]

        # 3) Initialize memory buffer
        mem_cfg = cfg["memory"]
        self.memory = Memory(
            max_size=mem_cfg["max_size"],
            storage_path=mem_cfg["storage_path"],
        )

    def respond(self, user_input: str) -> str:
        """
        - Append the user prompt to memory.
        - Call ollama.generate(...) to get a completion.
        - Append the assistant’s reply to memory.
        - Return the text response.
        """

        # Store the user prompt
        self.memory.add({"role": "user", "text": user_input})

        # Call Ollama’s generate(...) endpoint.
        #   - `model=` must match the name you pulled with `ollama pull <model>`
        #   - By default, ollama.generate(...) hits http://localhost:11434/api/generate
        result = generate(
            model=self.model_name,
            prompt=user_input
        )

        # Extract the plain-text response
        # Note: result is a dict that looks like: { "response": "..." }
        response_text = result.get("response", "").strip()

        # Store the assistant reply
        self.memory.add({"role": "assistant", "text": response_text})
        return response_text

    def save_memory(self):
        """
        Dump the entire buffer (list of {role,text} entries) to a timestamped JSON file.
        """
        self.memory.save()


if __name__ == "__main__":
    # Quick sanity check: does Ollama load and reply?
    agent = Agent()
    reply = agent.respond("Hello, how are you?")
    print("AI replied:", reply)
