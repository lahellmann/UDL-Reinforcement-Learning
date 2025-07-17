### DQN ###

# src/bullet_chess_rl/agents/dqn/dqn_agent.py

import logging
import random
from collections import deque
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### DQN ###

# src/bullet_chess_rl/agents/dqn/dqn_agent.py

class ChessQNetwork(nn.Module):
    """Q-Network for bullet chess position evaluation."""

    def __init__(self, input_channels: int = 15, hidden_dim: int = 512):
        super().__init__()

        # Convolutional layers for board pattern recognition
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # First conv layer: 15 channels -> 64 features
            nn.BatchNorm2d(64),  # Normalize features for stable training
            nn.ReLU(),  # Activation function
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Second conv layer: 64 -> 128 features
            nn.BatchNorm2d(128),  # Normalize features
            nn.ReLU(),  # Activation function
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Third conv layer: 128 -> 256 features
            nn.BatchNorm2d(256),  # Normalize features
            nn.ReLU(),  # Activation function
        )

        # Residual blocks for deeper understanding
        self.res_blocks = nn.ModuleList([
            self._make_res_block(256) for _ in range(4)  # Create 4 residual blocks with 256 channels
        ])

        # Fully connected layers for Q-value estimation
        conv_output_size = 256 * 8 * 8  # Flattened conv output size (256 channels * 8x8 board)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),  # First FC layer
            nn.ReLU(),  # Activation
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),  # Second FC layer (half size)
            nn.ReLU(),  # Activation
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(hidden_dim // 2, 4096)  # Output layer: 64*64 possible moves
        )

        # Time pressure head for bullet chess specific features
        self.time_pressure_head = nn.Sequential(
            nn.Linear(conv_output_size, 256),  # Input from conv features
            nn.ReLU(),  # Activation
            nn.Linear(256, 64),  # Hidden layer
            nn.ReLU(),  # Activation
            nn.Linear(64, 1)  # Output: Time pressure urgency score
        )

    def _make_res_block(self, channels):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # First conv in residual block
            nn.BatchNorm2d(channels),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # Second conv in residual block
            nn.BatchNorm2d(channels),  # Batch normalization
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q-values and time pressure score.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            q_values: Q-values for all possible moves
            time_pressure_score: Urgency score for time management
        """
        # Convolutional feature extraction
        features = self.conv_layers(x)  # Extract board features through conv layers

        # Residual blocks
        for res_block in self.res_blocks:
            residual = features  # Store input for skip connection
            features = res_block(features)  # Apply residual block
            features = features + residual  # Add skip connection
            features = F.relu(features)  # Apply activation after residual connection

        # Flatten for fully connected layers
        flattened = features.reshape(features.size(0), -1)  # Flatten conv output to 1D

        # Q-values for moves
        q_values = self.fc_layers(flattened)  # Compute Q-values for all possible moves

        # Time pressure assessment
        time_pressure_score = self.time_pressure_head(flattened)  # Compute time pressure urgency

        return q_values, time_pressure_score


class BulletExperienceReplay:
    """Experience replay buffer optimized for bullet chess."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)  # Circular buffer for experiences
        self.time_pressure_weights = deque(maxlen=capacity)  # Weights for time pressure importance

    def push(self, state, action, reward, next_state, done, time_pressure_level):
        """Add experience with time pressure weighting."""
        experience = (state, action, reward, next_state, done)  # Bundle experience tuple
        self.buffer.append(experience)  # Add to buffer

        # Weight experiences based on time pressure for better learning
        weight = {
            "relaxed": 1.0,    # Normal weight for relaxed positions
            "moderate": 1.2,   # Slightly higher weight for moderate pressure
            "pressure": 1.5,   # Higher weight for pressure situations
            "scramble": 2.0    # Highest weight for time scramble
        }.get(time_pressure_level, 1.0)  # Default weight if level not found

        self.time_pressure_weights.append(weight)  # Store weight for this experience

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with time pressure weighting."""
        if len(self.buffer) < batch_size:
            return None  # Not enough experiences to sample

        # Sample with probability proportional to time pressure weights
        weights = np.array(self.time_pressure_weights)  # Convert weights to numpy array
        probabilities = weights / weights.sum()  # Normalize to probabilities

        # Sample indices based on time pressure importance
        indices = np.random.choice(len(self.buffer), batch_size,
                                 replace=False, p=probabilities)

        batch = [self.buffer[i] for i in indices]  # Get experiences at sampled indices
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))  # Unpack and stack

        # Convert to PyTorch tensors
        return (
            torch.FloatTensor(states),    # States tensor
            torch.LongTensor(actions),    # Actions tensor (integers)
            torch.FloatTensor(rewards),   # Rewards tensor
            torch.FloatTensor(next_states),  # Next states tensor
            torch.BoolTensor(dones)       # Done flags tensor
        )

    def __len__(self):
        return len(self.buffer)  # Return number of stored experiences


class BulletChessDQNAgent:
    """DQN Agent optimized for bullet chess time pressure."""

    def __init__(self,
                 learning_rate: float = 1e-4,      # Learning rate for optimizer
                 epsilon: float = 1.0,             # Initial exploration rate
                 epsilon_decay: float = 0.995,     # Exploration decay factor
                 epsilon_min: float = 0.05,        # Minimum exploration rate
                 gamma: float = 0.99,              # Discount factor for future rewards
                 batch_size: int = 64,             # Batch size for training
                 target_update_freq: int = 1000,   # Frequency to update target network
                 device: str = None):              # Device to run on (CPU/GPU)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect device

        # Networks
        self.q_network = ChessQNetwork().to(self.device)        # Main Q-network
        self.target_network = ChessQNetwork().to(self.device)   # Target Q-network for stability
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)  # Optimizer

        # Training parameters
        self.epsilon = epsilon                    # Current exploration rate
        self.epsilon_decay = epsilon_decay        # Rate of exploration decay
        self.epsilon_min = epsilon_min           # Minimum exploration rate
        self.gamma = gamma                       # Discount factor
        self.batch_size = batch_size             # Training batch size
        self.target_update_freq = target_update_freq  # Target network update frequency

        # Experience replay
        self.memory = BulletExperienceReplay()   # Experience replay buffer
        self.steps = 0                          # Training steps counter
        self.losses = []                        # Training losses history

        # Time management
        self.time_allocation_strategy = "adaptive"  # Strategy for time allocation

        # Copy weights to target network
        self.update_target_network()            # Initialize target network with same weights

        logging.info(f"DQN Agent initialized on {self.device}")

    def select_action(self, state: np.ndarray, legal_actions: List[int],
                     time_pressure_level: str = "moderate") -> int:
        """
        Select action using epsilon-greedy with time pressure adaptation.

        Args:
            state: Current board state
            legal_actions: List of legal move indices
            time_pressure_level: Current time pressure level

        Returns:
            Selected action index
        """
        # Adapt epsilon based on time pressure
        adapted_epsilon = self._adapt_epsilon(time_pressure_level)

        # Epsilon-greedy exploration
        if random.random() < adapted_epsilon:
            return random.choice(legal_actions)  # Random action for exploration

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        # Get Q-values from network (no gradients for inference)
        with torch.no_grad():
            q_values, time_pressure_score = self.q_network(state_tensor)
            q_values = q_values.squeeze(0)  # Remove batch dimension

        # Mask illegal moves
        masked_q_values = torch.full((4096,), float('-inf')).to(self.device)  # Initialize with -inf
        masked_q_values[legal_actions] = q_values[legal_actions]  # Set legal moves to actual Q-values

        # Time pressure adjustment
        if time_pressure_level in ["pressure", "scramble"]:
            # In time pressure, prefer quicker tactical shots
            urgency_bonus = time_pressure_score.item() * 0.1  # Small bonus based on urgency
            masked_q_values[legal_actions] += urgency_bonus   # Add bonus to legal moves

        return masked_q_values.argmax().item()  # Return action with highest Q-value

    def _adapt_epsilon(self, time_pressure_level: str) -> float:
        """Adapt exploration rate based on time pressure."""
        adaptation_factors = {
            "relaxed": 1.0,      # Normal exploration when time is abundant
            "moderate": 0.8,     # Slightly less exploration with moderate pressure
            "pressure": 0.5,     # Reduced exploration under pressure
            "scramble": 0.2      # Minimal exploration in time scramble - trust training
        }

        factor = adaptation_factors.get(time_pressure_level, 1.0)  # Get factor or default
        return max(self.epsilon_min, self.epsilon * factor)  # Apply factor but respect minimum

    def store_experience(self, state, action, reward, next_state, done, time_pressure_level):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done, time_pressure_level)

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough experiences to train

        # Sample batch
        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return 0.0  # Sampling failed

        states, actions, rewards, next_states, dones = batch_data

        # Convert to correct format (batch, channels, height, width)
        states = states.permute(0, 3, 1, 2).to(self.device)      # Rearrange dimensions
        next_states = next_states.permute(0, 3, 1, 2).to(self.device)  # Rearrange dimensions
        actions = actions.to(self.device)        # Move to device
        rewards = rewards.to(self.device)        # Move to device
        dones = dones.to(self.device)           # Move to device

        # Current Q values
        current_q_values, _ = self.q_network(states)                        # Get Q-values from main network
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))  # Select Q-values for taken actions

        # Next Q values from target network (Double DQN)
        with torch.no_grad():  # No gradients for target network
            next_q_values, _ = self.target_network(next_states)     # Get Q-values from target network
            max_next_q_values = next_q_values.max(1)[0]            # Get maximum Q-values
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)  # Compute targets

        # Compute loss with Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()                                          # Clear gradients
        loss.backward()                                                     # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)   # Clip gradients for stability
        self.optimizer.step()                                               # Update weights

        # Update target network
        self.steps += 1                                     # Increment step counter
        if self.steps % self.target_update_freq == 0:      # Check if it's time to update target
            self.update_target_network()                    # Copy weights to target network

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decay exploration rate

        loss_value = loss.item()    # Get scalar loss value
        self.losses.append(loss_value)  # Store for tracking

        return loss_value

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def allocate_time(self, position_complexity: float, remaining_time: float) -> float:
        """
        Smart time allocation based on position complexity and remaining time.

        Args:
            position_complexity: Estimated complexity of current position (0-1)
            remaining_time: Time remaining in seconds

        Returns:
            Allocated thinking time in seconds
        """
        if self.time_allocation_strategy == "adaptive":
            # Adaptive strategy: more time for complex positions, less when low on time
            base_time = min(remaining_time * 0.1, 5.0)  # Max 10% of remaining time, capped at 5s
            complexity_factor = 0.5 + position_complexity  # Scale from 0.5x to 1.5x based on complexity

            if remaining_time < 10:
                # Emergency time management - very quick moves
                return min(0.5, remaining_time * 0.05)  # Max 0.5s or 5% of remaining time
            elif remaining_time < 30:
                # Conservative time management - save time for endgame
                return base_time * 0.7 * complexity_factor
            else:
                # Normal time management - can afford to think
                return base_time * complexity_factor

        else:  # "fixed" strategy
            return min(1.0, remaining_time * 0.05)  # Fixed 1s or 5% of remaining time

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "steps": self.steps,                                              # Total training steps
            "epsilon": self.epsilon,                                          # Current exploration rate
            "memory_size": len(self.memory),                                 # Size of replay buffer
            "avg_loss_last_100": np.mean(self.losses[-100:]) if self.losses else 0.0,  # Recent average loss
            "target_updates": self.steps // self.target_update_freq          # Number of target network updates
        }

    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),         # Main network weights
            'target_network_state_dict': self.target_network.state_dict(), # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(),         # Optimizer state
            'epsilon': self.epsilon,                                      # Current exploration rate
            'steps': self.steps,                                         # Training step count
            'losses': self.losses,                                       # Loss history
            'hyperparameters': {                                         # Training hyperparameters
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)  # Load checkpoint

        # Restore network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        self.epsilon = checkpoint['epsilon']           # Restore exploration rate
        self.steps = checkpoint['steps']               # Restore step count
        self.losses = checkpoint.get('losses', [])     # Restore loss history

        logging.info(f"Model loaded from {filepath}")
        logging.info(f"Loaded state: steps={self.steps}, epsilon={self.epsilon:.3f}")

    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.q_network.eval()      # Disable dropout and batch norm updates
        self.target_network.eval() # Disable dropout and batch norm updates

    def set_train_mode(self):
        """Set networks to training mode."""
        self.q_network.train()      # Enable dropout and batch norm updates
        self.target_network.train() # Enable dropout and batch norm updates
    """DQN Agent optimized for bullet chess time pressure."""

    def __init__(self,
                 learning_rate: float = 1e-4,      # Learning rate for optimizer
                 epsilon: float = 1.0,             # Initial exploration rate
                 epsilon_decay: float = 0.995,     # Exploration decay factor
                 epsilon_min: float = 0.05,        # Minimum exploration rate
                 gamma: float = 0.99,              # Discount factor for future rewards
                 batch_size: int = 64,             # Batch size for training
                 target_update_freq: int = 1000,   # Frequency to update target network
                 device: str = None):              # Device to run on (CPU/GPU)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect device

        # Networks
        self.q_network = ChessQNetwork().to(self.device)        # Main Q-network
        self.target_network = ChessQNetwork().to(self.device)   # Target Q-network for stability
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)  # Optimizer

        # Training parameters
        self.epsilon = epsilon                    # Current exploration rate
        self.epsilon_decay = epsilon_decay        # Rate of exploration decay
        self.epsilon_min = epsilon_min           # Minimum exploration rate
        self.gamma = gamma                       # Discount factor
        self.batch_size = batch_size             # Training batch size
        self.target_update_freq = target_update_freq  # Target network update frequency

        # Experience replay
        self.memory = BulletExperienceReplay()   # Experience replay buffer
        self.steps = 0                          # Training steps counter
        self.losses = []                        # Training losses history

        # Time management
        self.time_allocation_strategy = "adaptive"  # Strategy for time allocation

        # Copy weights to target network
        self.update_target_network()            # Initialize target network with same weights

        logging.info(f"DQN Agent initialized on {self.device}")

    def select_action(self, state: np.ndarray, legal_actions: List[int],
                     time_pressure_level: str = "moderate") -> int:
        """
        Select action using epsilon-greedy with time pressure adaptation.

        Args:
            state: Current board state
            legal_actions: List of legal move indices
            time_pressure_level: Current time pressure level

        Returns:
            Selected action index
        """
        # Adapt epsilon based on time pressure
        adapted_epsilon = self._adapt_epsilon(time_pressure_level)

        # Epsilon-greedy exploration
        if random.random() < adapted_epsilon:
            return random.choice(legal_actions)  # Random action for exploration

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        # Get Q-values from network (no gradients for inference)
        with torch.no_grad():
            q_values, time_pressure_score = self.q_network(state_tensor)
            q_values = q_values.squeeze(0)  # Remove batch dimension

        # Mask illegal moves
        masked_q_values = torch.full((4096,), float('-inf')).to(self.device)  # Initialize with -inf
        masked_q_values[legal_actions] = q_values[legal_actions]  # Set legal moves to actual Q-values

        # Time pressure adjustment
        if time_pressure_level in ["pressure", "scramble"]:
            # In time pressure, prefer quicker tactical shots
            urgency_bonus = time_pressure_score.item() * 0.1  # Small bonus based on urgency
            masked_q_values[legal_actions] += urgency_bonus   # Add bonus to legal moves

        return masked_q_values.argmax().item()  # Return action with highest Q-value

    def _adapt_epsilon(self, time_pressure_level: str) -> float:
        """Adapt exploration rate based on time pressure."""
        adaptation_factors = {
            "relaxed": 1.0,      # Normal exploration when time is abundant
            "moderate": 0.8,     # Slightly less exploration with moderate pressure
            "pressure": 0.5,     # Reduced exploration under pressure
            "scramble": 0.2      # Minimal exploration in time scramble - trust training
        }

        factor = adaptation_factors.get(time_pressure_level, 1.0)  # Get factor or default
        return max(self.epsilon_min, self.epsilon * factor)  # Apply factor but respect minimum

    def store_experience(self, state, action, reward, next_state, done, time_pressure_level):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done, time_pressure_level)

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough experiences to train

        # Sample batch
        batch_data = self.memory.sample(self.batch_size)
        if batch_data is None:
            return 0.0  # Sampling failed

        states, actions, rewards, next_states, dones = batch_data

        # Convert to correct format (batch, channels, height, width)
        states = states.permute(0, 3, 1, 2).to(self.device)      # Rearrange dimensions
        next_states = next_states.permute(0, 3, 1, 2).to(self.device)  # Rearrange dimensions
        actions = actions.to(self.device)        # Move to device
        rewards = rewards.to(self.device)        # Move to device
        dones = dones.to(self.device)           # Move to device

        # Current Q values
        current_q_values, _ = self.q_network(states)                        # Get Q-values from main network
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))  # Select Q-values for taken actions

        # Next Q values from target network (Double DQN)
        with torch.no_grad():  # No gradients for target network
            next_q_values, _ = self.target_network(next_states)     # Get Q-values from target network
            max_next_q_values = next_q_values.max(1)[0]            # Get maximum Q-values
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)  # Compute targets

        # Compute loss with Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()                                          # Clear gradients
        loss.backward()                                                     # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)   # Clip gradients for stability
        self.optimizer.step()                                               # Update weights

        # Update target network
        self.steps += 1                                     # Increment step counter
        if self.steps % self.target_update_freq == 0:      # Check if it's time to update target
            self.update_target_network()                    # Copy weights to target network

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Decay exploration rate

        loss_value = loss.item()    # Get scalar loss value
        self.losses.append(loss_value)  # Store for tracking

        return loss_value

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def allocate_time(self, position_complexity: float, remaining_time: float) -> float:
        """
        Smart time allocation based on position complexity and remaining time.

        Args:
            position_complexity: Estimated complexity of current position (0-1)
            remaining_time: Time remaining in seconds

        Returns:
            Allocated thinking time in seconds
        """
        if self.time_allocation_strategy == "adaptive":
            # Adaptive strategy: more time for complex positions, less when low on time
            base_time = min(remaining_time * 0.1, 5.0)  # Max 10% of remaining time, capped at 5s
            complexity_factor = 0.5 + position_complexity  # Scale from 0.5x to 1.5x based on complexity

            if remaining_time < 10:
                # Emergency time management - very quick moves
                return min(0.5, remaining_time * 0.05)  # Max 0.5s or 5% of remaining time
            elif remaining_time < 30:
                # Conservative time management - save time for endgame
                return base_time * 0.7 * complexity_factor
            else:
                # Normal time management - can afford to think
                return base_time * complexity_factor

        else:  # "fixed" strategy
            return min(1.0, remaining_time * 0.05)  # Fixed 1s or 5% of remaining time

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "steps": self.steps,                                              # Total training steps
            "epsilon": self.epsilon,                                          # Current exploration rate
            "memory_size": len(self.memory),                                 # Size of replay buffer
            "avg_loss_last_100": np.mean(self.losses[-100:]) if self.losses else 0.0,  # Recent average loss
            "target_updates": self.steps // self.target_update_freq          # Number of target network updates
        }

    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),         # Main network weights
            'target_network_state_dict': self.target_network.state_dict(), # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(),         # Optimizer state
            'epsilon': self.epsilon,                                      # Current exploration rate
            'steps': self.steps,                                         # Training step count
            'losses': self.losses,                                       # Loss history
            'hyperparameters': {                                         # Training hyperparameters
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)  # Load checkpoint

        # Restore network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        self.epsilon = checkpoint['epsilon']           # Restore exploration rate
        self.steps = checkpoint['steps']               # Restore step count
        self.losses = checkpoint.get('losses', [])     # Restore loss history

        logging.info(f"Model loaded from {filepath}")
        logging.info(f"Loaded state: steps={self.steps}, epsilon={self.epsilon:.3f}")

    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.q_network.eval()      # Disable dropout and batch norm updates
        self.target_network.eval() # Disable dropout and batch norm updates

    def set_train_mode(self):
        """Set networks to training mode."""
        self.q_network.train()      # Enable dropout and batch norm updates
        self.target_network.train() # Enable dropout and batch norm updates