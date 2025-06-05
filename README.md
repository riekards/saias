# Local Self-Learning AI Assistant

This project is a starting scaffold for a **local**, self-evolving, self-learning AI assistant that uses Ollama (no external APIs). It includes:

1. **configs/default.yaml** – default hyperparameters and paths.
2. **requirements.txt** – Python dependencies.
3. **src/** – core modules:
   - `agent.py` – loads the Ollama model, handles user queries.
   - `memory.py` – a simple replay-buffer/memory store.
   - `trainer.py` – routines to fine-tune the model locally using a simple PPO algorithm.
   - `cli.py` – a command-line interface for interacting with the assistant.
4. **tests/test_agent.py** – a basic unit test for the Agent class.

Additional modules:
   - `scheduler.py` – periodically triggers `Trainer.fine_tune()`.
   - `self_improve.py` – file watcher that launches fine-tuning when memory reaches a threshold.
   - `ppo_model.py` – minimal PyTorch implementation of PPO utilities.
   - `envs/` – contains `saias_env.py`, a simple Gym environment.

## Quickstart

1. Clone the repo.
2. Create a virtual environment (Windows):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   pip install -e .        # install the src package
   ```
   If you skip the editable install step, Python will not find the internal
   modules and running scripts such as `train_ppo.py` can raise
   `ModuleNotFoundError` errors.
4. Launch the CLI:
   ```bash
   python src/cli.py [--train | --self-improve]
   ```
5. Run the test suite:
   ```bash
   pytest
   ```
6. Train the reinforcement-learning environment:
   ```bash
   python train_ppo.py
   ```
