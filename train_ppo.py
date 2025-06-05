import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.saias_env import SaiasEnv


def main():
    # 1) Create vectorized environment
    def make_env():
        return SaiasEnv(config_path="configs/default.yaml")
    env = DummyVecEnv([make_env])

    # 2) Define a directory for saving checkpoints
    log_dir = "ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 3) Initialize PPO agent
    # Use MlpPolicy by default; adjust hyperparameters as needed
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log=log_dir,
        device="auto",
    )

    # 4) Train for 100,000 timesteps
    model.learn(total_timesteps=100_000)

    # 5) Save the final model
    model.save(os.path.join(log_dir, "saias_ppo_final"))

    # 6) Close environments
    env.close()


if __name__ == "__main__":
    main()
