# Local Self-Learning AI Assistant

This project is a starting scaffold for a **local**, self-evolving, self-learning AI assistant that uses Ollama (no external APIs). It includes:

1. **configs/default.yaml** – default hyperparameters and paths.
2. **Dockerfile** – container definition for easy deployment.
3. **requirements.txt** – Python dependencies.
4. **src/** – core modules:
   - `agent.py` – loads the Ollama model, handles user queries.
   - `memory.py` – a simple replay-buffer/memory store.
   - `trainer.py` – routines to fine-tune the model locally.
   - `cli.py` – a command-line interface for interacting with the assistant.
5. **tests/test_agent.py** – a basic unit test for the Agent class.

## Quickstart

1. Clone the repo.
2. Build the Docker image:  
   ```bash
   docker build -t local-saias .
   ```
