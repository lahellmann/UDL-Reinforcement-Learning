import chess
import random

from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch

# Utility functions for reproducibility and learning rate scheduling
def seed_everything(seed: int = 42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def linear_anneal(start, end, cur, total):
    """Linear annealing schedule for hyperparameters"""
    if total <= 0:
        total = 1  # Avoid division by zero
    t = min(1.0, cur / float(total))
    return start + (end - start) * t

def get_time_pressure_level(time_remaining: float) -> str:
    """Convert numeric time to pressure level."""
    if time_remaining > 30:
        return "relaxed"
    elif time_remaining > 10:
        return "moderate"
    elif time_remaining > 5:
        return "pressure"
    else:
        return "scramble"

def generate_all_possible_uci_moves():
        board = chess.Board()
        moves = set()
        squares = list(chess.SQUARES)
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_square in squares:
            for to_square in squares:
                # Normal move
                move = chess.Move(from_square, to_square)
                moves.add(move.uci())
                # Promotion moves (only if move is from 7th rank for white or 2nd rank for black)
                if chess.square_rank(from_square) == 6:  # White 7th rank
                    for promo in promotion_pieces:
                        promo_move = chess.Move(from_square, to_square, promotion=promo)
                        moves.add(promo_move.uci())
                if chess.square_rank(from_square) == 1:  # Black 2nd rank
                    for promo in promotion_pieces:
                        promo_move = chess.Move(from_square, to_square, promotion=promo)
                        moves.add(promo_move.uci())
        return sorted(moves)

def get_truly_fixed_cfg():
    """
    Configuration dictionary for training hyperparameters

    Contains all hyperparameters for:
    - Environment setup (time controls, thinking simulation)
    - Agent architecture and learning parameters
    - Experience replay configuration
    - Training schedule and evaluation
    - Opponent curriculum
    - File paths for models and logs
    """
    return {
        "seed": 42,
        "env": {
            "time_limit": 60,      # Bullet chess time limit in seconds
            "increment": 0.0,      # Time increment per move
            "simulate_think": True, # Simulate realistic thinking time
            "think_lo": 0.80,      # Minimum thinking time
            "think_hi": 1.60       # Maximum thinking time
        },
        "agent": {
            "lr": 1e-4,            # Learning rate
            "gamma": 0.99,         # Discount factor
            "epsilon_start": 1.0,  # Initial exploration rate
            "epsilon_end": 0.05,   # Final exploration rate
            "epsilon_decay": 0.9996, # Exploration decay rate
            "batch_size": 64,      # Mini-batch size
            "target_update": 2000, # Hard target update frequency
            "tau": 0.002,          # Soft target update rate
            "use_soft_update": True, # Use soft vs hard target updates
            "n_step": 3,           # N-step learning horizon
            "dropout": 0.1,        # Network dropout rate
            "noise_std": 0.01      # Action noise for exploration
        },
        "replay": {
            "capacity": 100000,    # Replay buffer size
            "per_alpha": 0.6,      # PER priority exponent
            "per_beta_start": 0.4, # PER importance sampling start
            "per_beta_frames": 1_000_000 # PER beta annealing frames
        },
        "training": {
            "episodes": 1000,      # Total training episodes
            "max_moves": 120,      # Maximum moves per game
            "move_limit_penalty": -3.0, # Penalty for reaching move limit
            "warmup": 50,         # Episodes before learning starts
            "log_freq": 10,        # Logging frequency
            "eval_freq": 100,      # Evaluation frequency
            "eval_games": 30,      # Games per evaluation
            "save_freq": 100,       # Model saving frequency
            'overall_time': 60 * 60,  # Total training time in seconds
        },
        "opponent": {
            "strength_start": 0.0, # Initial opponent strength
            "strength_end": 0.3    # Final opponent strength
        },
        "paths": {
            "models": "models", # Model save directory
            "ddqn_logs": "logs_ddqn",      # Log save directory
            "mcts_logs": "logs_polval"  # Log directory
        },
        "model": {
            "n_actions": 4096,                       # number of possible actions (copied from DDQN)
            "board_shape": (8, 8, 15)
        },
        "mcts": {
            "mcts_simulations": 10,
            "c_puct": 1.5
        }
    }

