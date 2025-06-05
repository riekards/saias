# src/cli.py

import argparse
from agent import Agent
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Local AI Assistant CLI")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Fine-tune the model on memory buffer"
    )
    args = parser.parse_args()

    agent = Agent()
    trainer = Trainer()

    if args.train:
        print("Starting fine-tuning...")
        trainer.fine_tune()
        return

    print("Welcome to your Local AI Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            agent.save_memory()
            print("Goodbye!")
            break
        response = agent.respond(user_input)
        print("AI:", response)

if __name__ == "__main__":
    main()
