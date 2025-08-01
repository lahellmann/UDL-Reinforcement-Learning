import os
import random
from collections import namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import src.utils as utils
from src.environment import BulletChessEnv

def load_ddqn_from_path(path, cfg, env):
    """
    Load a DDQN agent from a specified file path.

    Args:
        path (str): The file path to load the agent from.
        cfg (dict): The configuration dictionary containing agent parameters.

    Returns:
        DDQNAgent: The loaded DDQN agent.
    """
    agent = DDQNAgent( cfg, env)
    agent.load(path)
    return agent


class DuelingQNet(nn.Module):
    """
    Dueling DQN architecture for chess position evaluation

    Architecture:
    - Convolutional layers for spatial pattern recognition
    - Separate value and advantage streams
    - Batch normalization and dropout for regularization
    """

    def __init__(self, in_channels: int = 15, hidden: int = 512, n_actions: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.n_actions = n_actions

        # Convolutional feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        conv_out = 256 * 8 * 8

        # Shared feature layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Advantage stream (action preferences)
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, n_actions)
        )

        # Value stream (state value)
        self.val = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing dueling architecture"""
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        adv = self.adv(x)
        val = self.val(x)

        # Combine value and advantage streams
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


# Prioritized Experience Replay Components
NStepExp = namedtuple("NStepExp", ["state", "action", "reward", "next_state", "done", "mask"])

class NStepBuffer:
    """Buffer for n-step learning to improve temporal credit assignment"""

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf: List[NStepExp] = []

    def reset(self):
        self.buf.clear()

    def push(self, exp: NStepExp):
        self.buf.append(exp)

    def can_pop(self):
        return len(self.buf) >= self.n

    def pop(self) -> NStepExp:
        """Calculate n-step return and create transition"""
        R = 0.0
        next_state = None
        next_mask = None
        done = False

        # Calculate discounted n-step return
        for i, e in enumerate(self.buf[:self.n]):
            R += (self.gamma ** i) * e.reward
            if e.done:
                done = True
                next_state = e.next_state
                next_mask = e.mask
                break

        if not done:
            last = self.buf[self.n - 1]
            next_state = last.next_state
            next_mask = last.mask
            done = last.done

        first = self.buf[0]
        self.buf.pop(0)
        return NStepExp(first.state, first.action, R, next_state, done, next_mask)

    def flush_all(self):
        """Flush all remaining experiences at episode end"""
        outs = []
        while self.buf:
            outs.append(self.pop())
        return outs


class PERBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning

    Features:
    - Priority-based sampling using TD errors
    - Importance sampling weights to correct bias
    - Annealed beta parameter for bias correction
    """

    def __init__(self, capacity: int, n_actions: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 1_000_000):
        self.capacity = capacity
        self.n_actions = n_actions
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Importance sampling exponent
        self.beta_frames = beta_frames

        self.pos = 0
        self.size = 0

        # Pre-allocated arrays for efficiency
        self.states = np.zeros((capacity, 8, 8, 15), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, 8, 8, 15), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.bool_)

        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.frame = 1

    def __len__(self):
        return self.size

    def beta(self):
        """Annealed beta for importance sampling"""
        return utils.linear_anneal(self.beta_start, 1.0, self.frame, self.beta_frames)

    def push(self, s, a, r, ns, d, nm):
        """Add experience with maximum priority"""
        idx = self.pos
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = ns
        self.dones[idx] = d
        self.next_masks[idx] = nm
        self.priorities[idx] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch with importance weights"""
        if self.size == 0:
            raise ValueError("PERBuffer empty")

        # Calculate sampling probabilities
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(self.size, batch_size, p=probs)

        # Calculate importance sampling weights
        beta = self.beta()
        self.frame += 1
        weights = (self.size * probs[idxs]) ** (-beta)
        weights = weights / weights.max()

        # Return batch as tensors
        batch = (
            torch.from_numpy(self.states[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_states[idxs]),
            torch.from_numpy(self.dones[idxs]),
            torch.from_numpy(self.next_masks[idxs]),
            torch.from_numpy(weights.astype(np.float32)),
            torch.from_numpy(idxs.astype(np.int64)),
        )
        return batch

    def update_priorities(self, idxs: torch.Tensor, prios: torch.Tensor):
        """Update priorities based on TD errors"""
        prios = prios.detach().cpu().numpy()
        idxs = idxs.detach().cpu().numpy()
        np.maximum(prios, 1e-6, out=prios)  # Ensure minimum priority
        self.priorities[idxs] = prios
        self.max_priority = max(self.max_priority, prios.max())


class DDQNAgent:
    """
    Double Deep Q-Network agent with advanced features:
    - Dueling architecture for better value estimation
    - Prioritized experience replay for efficient learning
    - N-step learning for improved temporal credit assignment
    - Soft target updates for stability
    - Action masking for legal move enforcement
    """

    def __init__(self, cfg: dict, env: BulletChessEnv, device=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q = DuelingQNet(n_actions=cfg["model"]["n_actions"], dropout=cfg["agent"]["dropout"]).to(self.device)
        self.q_target = DuelingQNet(n_actions=cfg["model"]["n_actions"], dropout=cfg["agent"]["dropout"]).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Optimizer with weight decay for regularization
        self.optim = optim.Adam(self.q.parameters(), lr=cfg["agent"]["lr"], weight_decay=1e-5)
        self.gamma = cfg["agent"]["gamma"]

        # Exploration parameters
        self.eps = cfg["agent"]["epsilon_start"]
        self.eps_end = cfg["agent"]["epsilon_end"]
        self.eps_decay = cfg["agent"]["epsilon_decay"]

        # Training parameters
        self.batch_size = cfg["agent"]["batch_size"]
        self.target_update = cfg["agent"]["target_update"]
        self.steps = 0
        self.n_actions = cfg["model"]["n_actions"]
        self.tau = cfg["agent"]["tau"]
        self.use_soft_update = cfg["agent"]["use_soft_update"]
        self.noise_std = cfg["agent"]["noise_std"]

        # Experience replay components
        self.per_buffer = PERBuffer(
            capacity=cfg["replay"]["capacity"],
            n_actions=cfg["model"]["n_actions"],
            alpha=cfg["replay"]["per_alpha"],
            beta_start=cfg["replay"]["per_beta_start"],
            beta_frames=cfg["replay"]["per_beta_frames"]
        )

        self.n_step = cfg["agent"]["n_step"]
        self.nstep_buffer = NStepBuffer(n=cfg["agent"]["n_step"], gamma=cfg["agent"]["gamma"])

    def select_action(self, state: np.ndarray, legal_actions: List[int], eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy with legal move masking

        Args:
            state: Current board observation
            legal_actions: List of legal action indices
            eval_mode: If True, disable exploration and noise

        Returns:
            Selected action index
        """
        current_eps = 0.0 if eval_mode else self.eps

        # Epsilon-greedy exploration
        if random.random() < current_eps:
            return random.choice(legal_actions)

        # Prepare state tensor
        st = torch.tensor(state, dtype=torch.float32, device=self.device)\
                .unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        self.q.train(not eval_mode)

        with torch.no_grad():
            q = self.q(st).squeeze(0)

            # Add noise during training for exploration
            if not eval_mode and self.noise_std > 0:
                noise = torch.randn_like(q) * self.noise_std
                q = q + noise

        # Mask illegal actions
        mask = torch.full((self.n_actions,), -1e9, device=self.device)
        idx = torch.tensor(legal_actions, dtype=torch.long, device=self.device)
        mask[idx] = 0.0
        q_masked = q + mask

        return int(torch.argmax(q_masked).item())

    def _soft_update(self):
        """Soft update of target network parameters"""
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q.parameters()):
                tp.data.lerp_((p.data), self.tau)

    def store_transition(self, state, action, reward, next_state, done, next_legal_mask):
        """Store experience in n-step buffer and replay buffer"""
        exp = NStepExp(state, action, reward, next_state, done, next_legal_mask)
        self.nstep_buffer.push(exp)

        # Pop n-step experience when buffer is full
        if self.nstep_buffer.can_pop():
            n_exp = self.nstep_buffer.pop()
            self.per_buffer.push(n_exp.state, n_exp.action, n_exp.reward,
                                 n_exp.next_state, n_exp.done, n_exp.mask)

        # Flush remaining experiences at episode end
        if done:
            for n_exp in self.nstep_buffer.flush_all():
                self.per_buffer.push(n_exp.state, n_exp.action, n_exp.reward,
                                     n_exp.next_state, n_exp.done, n_exp.mask)

    def update(self):
        """
        Perform one gradient update step

        Returns:
            Training loss value
        """
        if len(self.per_buffer) < self.batch_size:
            return 0.0

        self.q.train()
        self.q_target.eval()

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, next_legal_masks, weights, idxs = \
            self.per_buffer.sample(self.batch_size)

        # Move to device and reshape
        states = states.to(self.device).permute(0, 3, 1, 2).contiguous()
        next_states = next_states.to(self.device).permute(0, 3, 1, 2).contiguous()
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device)
        next_legal_masks = next_legal_masks.to(self.device)
        weights = weights.to(self.device)
        idxs = idxs.to(self.device)

        # Current Q-values
        q_values = self.q(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            # Use online network to select actions
            next_q_online = self.q(next_states)
            next_q_online = next_q_online.masked_fill(~next_legal_masks, -1e9)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

            # Use target network to evaluate actions
            next_q_target = self.q_target(next_states)
            next_q_target = next_q_target.masked_fill(~next_legal_masks, -1e9)
            next_q_target_a = next_q_target.gather(1, next_actions).squeeze(1)

            # Calculate target values
            target = rewards + (1 - dones.float()) * (self.gamma ** self.n_step) * next_q_target_a

        # Compute loss with importance sampling weights
        td_errors = target - q_a
        loss = (weights * F.smooth_l1_loss(q_a, target, reduction='none')).mean()

        # Backward pass with gradient clipping
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()

        # Update priorities in replay buffer
        new_prios = td_errors.abs() + 1e-6
        self.per_buffer.update_priorities(idxs, new_prios)

        # Update target network
        self.steps += 1
        if self.use_soft_update:
            self._soft_update()
        elif self.steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        # Decay exploration
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return float(loss.item())

    def save(self, path: str):
        """Save agent state to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optim": self.optim.state_dict(),
            "eps": self.eps,
            "steps": self.steps
        }, path)

    def load(self, path: str):
        """Load agent state from file"""
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.eps = ckpt.get("eps", 0.1)
        self.steps = ckpt.get("steps", 0)