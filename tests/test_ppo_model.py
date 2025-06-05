import torch
from src.ppo_model import PPOModel, ppo_update


def test_ppo_update_runs():
    model = PPOModel(obs_size=4, action_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    obs = torch.randn(5, 4)
    actions = torch.randint(0, 3, (5,))
    rewards = torch.ones(5)
    ppo_update(model, optimizer, obs, actions, rewards)
