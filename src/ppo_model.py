import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def tokenize(text: str):
    return text.lower().split()


def build_pairs(memory_buffer):
    pairs = []
    for i in range(len(memory_buffer) - 1):
        cur = memory_buffer[i]
        nxt = memory_buffer[i + 1]
        if cur.get("role") == "user" and nxt.get("role") == "assistant":
            pairs.append((cur.get("text", ""), nxt.get("text", "")))
    return pairs


def build_vocab(pairs):
    tokens = set()
    for user, assistant in pairs:
        tokens.update(tokenize(user))
        tokens.update(tokenize(assistant))
    vocab = {tok: idx for idx, tok in enumerate(sorted(tokens))}
    return vocab


def encode_text(text: str, vocab: dict):
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec


def prepare_dataset(memory_buffer):
    pairs = build_pairs(memory_buffer)
    if not pairs:
        return None, None, None, None
    vocab = build_vocab(pairs)
    obs = []
    actions = []
    for user, assistant in pairs:
        obs.append(encode_text(user, vocab))
        a_tok = tokenize(assistant)
        if not a_tok:
            continue
        actions.append(vocab.get(a_tok[0], 0))
    if not actions:
        return None, None, None, None
    obs = torch.stack(obs)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.ones(len(actions), dtype=torch.float32)
    return obs, actions, rewards, vocab


class PPOModel(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def ppo_update(model, optimizer, obs, actions, rewards, clip_eps=0.2):
    with torch.no_grad():
        old_log_probs, _, old_values = model.evaluate_actions(obs, actions)
    returns = compute_returns(rewards)
    for _ in range(1):
        log_probs, entropy, values = model.evaluate_actions(obs, actions)
        advantages = returns - values.detach()
        ratios = (log_probs - old_log_probs).exp()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
