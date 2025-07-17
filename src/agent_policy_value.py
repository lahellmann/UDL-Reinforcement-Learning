### Policy Methods ###
import logging
import math
from typing import List, Tuple, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import random
import chess
from utils import get_time_pressure_level, generate_all_possible_uci_moves


### Policy Methods ###

class ChessPolicyValueNetwork(nn.Module):
    """Combined policy and value network for AlphaZero-style training."""

    def __init__(self, input_channels: int = 15, hidden_dim: int = 256):
        super().__init__()

        # Shared convolutional backbone
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # Initial conv layer: 15 -> 64 channels
            nn.BatchNorm2d(64),  # Batch normalization for stable training
            nn.ReLU(),  # ReLU activation function
        )

        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList([
            self._make_res_block(64) for _ in range(6)  # Create 6 residual blocks with 64 channels each
        ])

        # Policy head - predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # 1x1 conv to reduce channels: 64 -> 32
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Flatten(),  # Flatten 2D feature maps to 1D
            nn.Linear(32 * 8 * 8, 4096),  # Fully connected: flattened features -> 4096 possible moves
        )

        # Value head - predicts position evaluation
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),  # 1x1 conv to single channel: 64 -> 1
            nn.BatchNorm2d(1),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Flatten(),  # Flatten 2D to 1D
            nn.Linear(8 * 8, hidden_dim),  # Fully connected: 64 -> hidden_dim
            nn.ReLU(),  # Activation
            nn.Linear(hidden_dim, 1),  # Output layer: hidden_dim -> 1 value
            nn.Tanh()  # Tanh activation to output value in [-1, 1] range
        )

        # Time management head for bullet chess
        self.time_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),  # 1x1 conv to single channel: 64 -> 1
            nn.BatchNorm2d(1),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Flatten(),  # Flatten 2D to 1D
            nn.Linear(8 * 8, 64),  # Fully connected: 64 -> 64
            nn.ReLU(),  # Activation
            nn.Linear(64, 1),  # Output layer: 64 -> 1 urgency score
            nn.Sigmoid()  # Sigmoid activation for time urgency output in [0, 1] range
        )

    def _make_res_block(self, channels):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # First conv in residual block
            nn.BatchNorm2d(channels),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # Second conv in residual block
            nn.BatchNorm2d(channels),  # Batch normalization (no activation here for residual connection)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits, value, and time urgency.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            policy_logits: Raw policy scores for all moves
            value: Position evaluation (-1 to 1)
            time_urgency: Time management urgency (0 to 1)
        """
        # Shared feature extraction
        features = self.conv_backbone(x)  # Extract initial features through conv backbone

        # Residual blocks
        for res_block in self.res_blocks:
            residual = features  # Store input for skip connection
            features = res_block(features)  # Apply residual block transformations
            features = features + residual  # Add skip connection (residual connection)
            features = F.relu(features)  # Apply activation after residual addition

        # Heads - compute outputs from shared features
        policy_logits = self.policy_head(features)  # Compute policy logits for all moves
        value = self.value_head(features)  # Compute position value evaluation
        time_urgency = self.time_head(features)  # Compute time urgency score

        return policy_logits, value, time_urgency


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state  # Game state at this node
        self.parent = parent  # Parent node in the tree
        self.action = action  # Action that led to this node from parent
        self.prior = prior  # Prior probability from neural network policy

        self.visit_count = 0  # Number of times this node has been visited
        self.value_sum = 0.0  # Sum of all values backed up through this node
        self.children = {}  # Dictionary mapping actions to child nodes
        self.is_expanded = False  # Whether this node has been expanded with children

    def is_leaf(self) -> bool:
        return not self.is_expanded  # Node is leaf if it hasn't been expanded yet

    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0  # No visits yet, return neutral value
        return self.value_sum / self.visit_count  # Return average value

    def ucb_score(self, c_puct: float = 1.0) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have infinite score (will be selected first)

        # UCB1 + prior term (PUCT algorithm)
        exploitation = self.value()  # Exploitation term: average value
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)  # Exploration term

        return exploitation + exploration  # Combined UCB score

    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def expand(self, policy_probs: np.ndarray, legal_actions: List[int], game_env):
        """Expand node with children for legal actions."""
        self.is_expanded = True  # Mark node as expanded

        for action in legal_actions:
            prior = policy_probs[action]  # Get prior probability for this action

            # Create new state by making the move
            new_env = game_env.copy()  # Copy current environment
            new_env.step(action)  # Apply the action

            # Create child node
            self.children[action] = MCTSNode(
                state=new_env,  # New game state after action
                parent=self,  # This node is the parent
                action=action,  # Action that led to child
                prior=prior  # Prior probability from network
            )

    def backup(self, value: float):
        """Backup value through the tree."""
        self.visit_count += 1  # Increment visit count
        self.value_sum += value  # Add value to sum

        if self.parent:
            self.parent.backup(-value)  # Backup negated value to parent (zero-sum game)


class AdaptiveMCTS:
    """Monte Carlo Tree Search adapted for bullet chess time pressure."""

    def __init__(self, neural_net, c_puct: float = 1.0):
        self.neural_net = neural_net  # Neural network for position evaluation
        self.c_puct = c_puct  # Exploration constant for UCB
        self.device = next(neural_net.parameters()).device  # Get device from network

    def search(self, game_env, num_simulations: int, time_budget: float = None) -> np.ndarray:
        """
        Run MCTS and return action probabilities.

        Args:
            game_env: Current game environment
            num_simulations: Number of MCTS simulations
            time_budget: Time budget for search (None for unlimited)

        Returns:
            action_probs: Probability distribution over actions
        """
        root = MCTSNode(game_env)  # Create root node with current state
        legal_actions = game_env.get_legal_actions()  # Get legal actions from current position

        # Adaptive simulation count based on time pressure
        if time_budget and time_budget < 1.0:
            num_simulations = max(10, num_simulations // 4)  # Very low time: reduce simulations to 1/4
        elif time_budget and time_budget < 5.0:
            num_simulations = max(50, num_simulations // 2)  # Low time: reduce simulations to 1/2

        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root, legal_actions, game_env)  # Perform one MCTS simulation

        # Return visit count distribution as action probabilities
        action_probs = np.zeros(4096)  # Initialize probability array for all possible moves
        total_visits = sum(child.visit_count for child in root.children.values())  # Total visits to all children

        if total_visits > 0:
            for action, child in root.children.items():
                action_probs[action] = child.visit_count / total_visits  # Probability proportional to visit count

        return action_probs

    def _simulate(self, root: MCTSNode, legal_actions: List[int], game_env):
        """Run one MCTS simulation."""
        node = root  # Start from root
        path = [node]  # Track path for backup

        # Selection: traverse tree until leaf
        while not node.is_leaf() and not node.state.is_game_over():
            node = node.select_child(self.c_puct)  # Select child with highest UCB score
            path.append(node)  # Add to path

        # Evaluation: get value and policy from neural network
        if not node.state.is_game_over():
            value, policy_probs = self._evaluate(node.state)  # Evaluate position with neural network

            # Expansion: add children for legal moves
            current_legal_actions = node.state.get_legal_actions()  # Get legal actions from current state
            if current_legal_actions:
                node.expand(policy_probs, current_legal_actions, node.state)  # Expand node with children
        else:
            # Terminal node - game is over
            result = node.state.get_result()  # Get game result
            if result == "1-0":
                value = 1.0  # White wins
            elif result == "0-1":
                value = -1.0  # Black wins
            else:
                value = 0.0  # Draw

        # Backup: propagate value up the tree
        for node in reversed(path):
            node.backup(value)  # Backup value to each node in path
            value = -value  # Flip value for opponent (zero-sum game)

    def _evaluate(self, game_env) -> Tuple[float, np.ndarray]:
        """Evaluate position using neural network."""
        state = game_env.get_observation()  # Get state representation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)  # Convert to tensor

        with torch.no_grad():  # No gradients needed for evaluation
            policy_logits, value, _ = self.neural_net(state_tensor)  # Forward pass through network
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]  # Convert logits to probabilities
            value = value.cpu().item()  # Extract scalar value

        return value, policy_probs


class BulletChessAlphaZeroAgent:
    """AlphaZero agent adapted for bullet chess."""
    
    def __init__(self,
                 learning_rate: float = 1e-3,  # Learning rate for optimizer
                 device: str = None):  # Device to run on (CPU/GPU)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-detect device

        # Neural network
        self.neural_net = ChessPolicyValueNetwork().to(self.device)  # Policy-value network
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate)  # Adam optimizer

        # MCTS
        self.mcts = AdaptiveMCTS(self.neural_net)  # Monte Carlo Tree Search with neural network

        # Training data
        self.training_data = []  # Store training examples
        self.losses = []  # Track training losses
        
        self.all_possible_uci_moves = generate_all_possible_uci_moves()
        self.action_index_to_uci = {i: uci for i, uci in enumerate(self.all_possible_uci_moves)}
        self.uci_to_action_index = {uci: i for i, uci in enumerate(self.all_possible_uci_moves)}


        logging.info(f"AlphaZero Agent initialized on {self.device}")

    def select_action(self, game_env, time_remaining: Union[float, str],
                     temperature: float = 1.0,  return_probs=False) -> int:
        """
        Select action using MCTS with time-aware simulation budget.

        Args:
            game_env: Current game environment
            time_remaining: Time remaining in seconds
            temperature: Temperature for action selection (higher = more random)

        Returns:
            Selected action index
        """
        # Adaptive simulation count based on time remaining
        # If input is a string label, convert it to time-based category
        if isinstance(time_remaining, str):
            time_pressure = time_remaining
        else:
            time_pressure = get_time_pressure_level(time_remaining)

        # Decide number of simulations based on time pressure
        if time_pressure == "relaxed":
            num_simulations = 200
        elif time_pressure == "moderate":
            num_simulations = 100
        elif time_pressure == "pressure":
            num_simulations = 50
        else:  # scramble
            num_simulations = 20

        # Get action probabilities from MCTS
        action_probs = self.mcts.search(game_env, num_simulations, time_remaining)

        # Apply temperature
        if temperature > 0:
            action_probs = action_probs ** (1 / temperature)  # Apply temperature scaling
            action_probs = action_probs / action_probs.sum()  # Renormalize probabilities

        # Sample action (or take best if temperature is very low)
        legal_actions = game_env.get_legal_actions()  # Get legal actions
        legal_probs = action_probs[legal_actions]  # Extract probabilities for legal actions

        if temperature < 0.1 or time_remaining < 5:
            # Deterministic selection under time pressure or low temperature
            best_idx = np.argmax(legal_probs)  # Get index of best action
            return legal_actions[best_idx]  # Return best legal action
        else:
            # Stochastic selection
            legal_probs = legal_probs / legal_probs.sum()  # Renormalize legal probabilities
            return np.random.choice(legal_actions, p=legal_probs)  # Sample according to probabilities

    def store_training_data(self, state, action_probs, value):
        """Store training data for later network updates."""
        self.training_data.append((state, action_probs, value))  # Add training example

    def train_network(self, epochs: int = 10, batch_size: int = 32):
        """Train the neural network on collected self-play data."""
        if len(self.training_data) < batch_size:
            return  # Not enough data to train

        self.neural_net.train()  # Set network to training mode

        for epoch in range(epochs):
            # Shuffle training data
            random.shuffle(self.training_data)  # Randomize order of training examples

            epoch_losses = []  # Track losses for this epoch

            # Process data in batches
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i + batch_size]  # Get batch

                states, target_policies, target_values = zip(*batch)  # Unpack batch

                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)  # States
                target_policies_tensor = torch.FloatTensor(np.array(target_policies)).to(self.device)  # Target policies
                target_values_tensor = torch.FloatTensor(np.array(target_values)).to(self.device)  # Target values

                # Forward pass
                policy_logits, values, _ = self.neural_net(states_tensor)  # Get network predictions

                # Compute losses
                policy_loss = F.cross_entropy(policy_logits, target_policies_tensor)  # Policy loss
                value_loss = F.mse_loss(values.squeeze(), target_values_tensor)  # Value loss
                total_loss = policy_loss + value_loss  # Combined loss

                # Backward pass
                self.optimizer.zero_grad()  # Clear gradients
                total_loss.backward()  # Compute gradients
                torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0)  # Clip gradients for stability
                self.optimizer.step()  # Update weights

                epoch_losses.append(total_loss.item())  # Record loss

            avg_loss = np.mean(epoch_losses)  # Average loss for epoch
            self.losses.append(avg_loss)  # Store average loss

            if epoch % 5 == 0:
                logging.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    def clear_training_data(self):
        """Clear stored training data."""
        self.training_data = []  # Reset training data list

    def allocate_time(self, position_complexity: float, remaining_time: float) -> float:
        """
        Allocate thinking time based on position complexity and time remaining.

        Args:
            position_complexity: Estimated complexity (0-1)
            remaining_time: Time remaining in seconds

        Returns:
            Allocated time in seconds
        """
        # AlphaZero tends to need more time for complex positions
        base_allocation = remaining_time * 0.05  # Base: 5% of remaining time
        complexity_multiplier = 1.0 + position_complexity  # Scale from 1x to 2x based on complexity

        if remaining_time < 10:
            # Emergency mode - minimal time allocation
            return min(1.0, remaining_time * 0.1)  # Max 1s or 10% of remaining time
        elif remaining_time < 30:
            # Conservative mode - save time for later
            return min(base_allocation * complexity_multiplier * 0.8, 3.0)  # Reduced allocation, max 3s
        else:
            # Normal mode - can afford to think
            return min(base_allocation * complexity_multiplier, 5.0)  # Normal allocation, max 5s

    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "training_data_size": len(self.training_data),  # Number of training examples
            "avg_loss_last_10": np.mean(self.losses[-10:]) if self.losses else 0.0,  # Recent average loss
            "total_training_updates": len(self.losses)  # Total number of training updates
        }

    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.neural_net.state_dict(),  # Neural network weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'training_data': self.training_data,  # Training examples
            'losses': self.losses,  # Loss history
        }, filepath)
        logging.info(f"AlphaZero model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)  # Load checkpoint

        # Restore network and optimizer states
        self.neural_net.load_state_dict(checkpoint['model_state_dict'])  # Load network weights
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        self.training_data = checkpoint.get('training_data', [])  # Load training data
        self.losses = checkpoint.get('losses', [])  # Load loss history

        logging.info(f"AlphaZero model loaded from {filepath}")
        logging.info(f"Loaded {len(self.training_data)} training samples")

    def set_eval_mode(self):
        """Set network to evaluation mode."""
        self.neural_net.eval()  # Disable dropout and batch norm updates

    def set_train_mode(self):
        """Set network to training mode."""
        self.neural_net.train()  # Enable dropout and batch norm updates
    """AlphaZero agent adapted for bullet chess."""
