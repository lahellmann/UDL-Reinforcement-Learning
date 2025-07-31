from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import random
import os

from environment import BulletChessEnv



class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels=15, n_actions=4096):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Flatten()

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions)
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value.squeeze(-1)
    
"""
MCTS (Monte Carlo Tree Search)
"""

class MCTSNode:
    def __init__(self, prior, parent=None):
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0

    def ucb_score(self, c_puct):
        return self.value() + c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
    


class MCTS:
    def __init__(self, model, env, n_simulations=800, c_puct=1.5, device=None):
        self.model = model
        self.env = env
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> np.ndarray:
        root = MCTSNode(prior=1.0)
        for _ in range(self.n_simulations):
            node = root
            scratch_env = deepcopy(self.env)

            # Selection
            while node.children:
                max_ucb = -float("inf")
                best_action = None
                for action, child in node.children.items():
                    ucb = child.ucb_score(self.c_puct)
                    if ucb > max_ucb:
                        max_ucb = ucb
                        best_action = action
                node = node.children[best_action]
                _, _, done, _ = scratch_env.step(best_action)
                if done:
                    break

            # Expansion
            obs = scratch_env.get_observation()
            obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            policy_logits, value = self.model(obs_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).detach().cpu().numpy()

            legal_actions = scratch_env.get_legal_actions()
            for action in legal_actions:
                if action not in node.children:
                    node.children[action] = MCTSNode(prior=policy[action], parent=node)

            # Backpropagation
            backup = node
            while backup:
                backup.visit_count += 1
                backup.value_sum += value.item()
                backup = backup.parent

        # Final action distribution π
        visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(4096)])
        if visit_counts.sum() == 0:
            # All visits 0 → fallback to uniform legal actions
            pi = np.zeros_like(visit_counts)
            for a in self.env.get_legal_actions():
                pi[a] = 1
            pi /= pi.sum()
        else:
            pi = visit_counts / visit_counts.sum()

        return pi


class MCTSAgent:
    """
    AlphaZero Agent with integrated MCTS and dual-headed policy-value network.
    No epsilon-greedy or target networks.
    """

    def __init__(self, cfg: dict, env: BulletChessEnv, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet().to(self.device)

        self.cfg = cfg
        self.env = env

        lr = cfg["agent"]["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
      
    def select_action_from_pi(self, pi, temperature=1.0):
        if temperature == 0:
            return int(np.argmax(pi))
        pi = np.asarray(pi) ** (1 / temperature)
        pi = pi / np.sum(pi)
        return int(np.random.choice(len(pi), p=pi))

    def run_mcts(self, env):
        """
        Runs MCTS on the current environment state and returns action probabilities.
        """

        mcts = MCTS(
            model=self.model,
            env=deepcopy(env),
            n_simulations = self.cfg["mcts"]["mcts_simulations"],
            c_puct = self.cfg["mcts"]["c_puct"],
            device=self.device
        )
        return mcts.run()


    def train_step(self, examples, optimizer, epochs=1):
        """
        Trains the model on a list of (state, policy, value) tuples.
        """
        self.model.train()
        for _ in range(epochs):
            states, policies, values = zip(*examples)

            states = torch.tensor(states, dtype=torch.float32, device=self.device)  # shape: (B, 8, 8, 15)
            states = states.permute(0, 3, 1, 2).contiguous()
            policies = torch.tensor(policies, dtype=torch.float32, device=self.device)
            values = torch.tensor(values, dtype=torch.float32, device=self.device)

            pred_policy, pred_value = self.model(states)

            # Policy loss: cross-entropy between MCTS probs and predicted policy
            policy_loss = -torch.sum(policies * torch.log(pred_policy + 1e-8), dim=1).mean()
            # Value loss: MSE between predicted and actual game outcome
            value_loss = F.mse_loss(pred_value.squeeze(), values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item()

    
    def save(self, path: str):
      """Save agent state to file"""
      os.makedirs(os.path.dirname(path), exist_ok=True)
      torch.save({
          "model": self.model.state_dict(),
          "optim": self.optimizer.state_dict()
      }, path)

    def load(self, path: str):
      """Load agent state from file"""
      ckpt = torch.load(path, map_location=self.device)
      self.model.load_state_dict(ckpt["model"])
      self.optimizer.load_state_dict(ckpt["optim"])

