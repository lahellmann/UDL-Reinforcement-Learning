"""

"""
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import numpy as np

import chess

@dataclass
class GameState:
    """State container for chess game including timing information"""
    board: chess.Board
    white_time: float
    black_time: float
    move_count: int
    last_move_time: float
    game_start_time: float

class BulletChessEnv:
    """
    Chess environment with time controls for reinforcement learning.
    Features:
    - Bullet time controls (60 seconds default)
    - Move simulation with thinking time
    - Material evaluation and positional rewards
    - Comprehensive game state observation
    """

    def __init__(self,
                 time_limit: int = 60,
                 increment: float = 0.0,
                 simulate_think: bool = True,
                 think_lo: float = 0.80,
                 think_hi: float = 1.60):
        self.time_limit = time_limit
        self.increment = increment
        self.simulate_think = simulate_think
        self.think_lo = think_lo
        self.think_hi = think_hi


        # Standard chess piece values for reward calculation
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.reset()

    def reset(self) -> np.ndarray:
        """Initialize new game and return initial observation"""
        self.state = GameState(
            board=chess.Board(),
            white_time=self.time_limit,
            black_time=self.time_limit,
            move_count=0,
            last_move_time=time.time(),
            game_start_time=time.time()
        )
        return self.get_observation()

    def step(self, action: Union[int, str], think_time: Optional[float] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one move in the environment

        Args:
            action: Either integer action index or UCI move string
            think_time: Optional override for thinking time

        Returns:
            observation, reward, done, info
        """
        if self.is_game_over():
            return self.get_observation(), 0.0, True, {"reason": "game_already_over"}

        old_board = self.state.board.copy()

        # Calculate thinking time
        if think_time is None and self.simulate_think:
            actual_think = float(np.random.uniform(self.think_lo, self.think_hi))
        else:
            actual_think = float(think_time or 0.0)

        # Deduct time from current player's clock
        if self.state.board.turn:
            self.state.white_time -= actual_think
            remaining = self.state.white_time
        else:
            self.state.black_time -= actual_think
            remaining = self.state.black_time

        # Check for time flag
        if remaining <= 0:
            reward = -20.0 if self.state.board.turn else 20.0
            return self.get_observation(), reward, True, {"reason": "time_flag"}

        # Convert action to chess move
        if isinstance(action, int):
            move = self._action_to_move(action)
        else:
            try:
                move = chess.Move.from_uci(action)
            except Exception:
                return self.get_observation(), -10.0, True, {"reason": "invalid_uci"}

        # Validate move legality
        if move not in self.state.board.legal_moves:
            return self.get_observation(), -10.0, True, {"reason": "illegal_move"}
            return self.get_observation(), -10.0, True, {"reason": "illegal_move"}

        # Execute move
        self.state.board.push(move)
        self.state.move_count += 1
        self.state.last_move_time += actual_think

        # Add increment after move
        if self.state.board.turn:
            self.state.black_time += self.increment
        else:
            self.state.white_time += self.increment

        # Calculate reward and check for game termination
        reward, done, info = self._calc_reward(old_board, move, actual_think)
        return self.get_observation(), reward, done, info

    def is_game_over(self) -> bool:
        """Check if game has ended by any condition"""
        return (
            self.state.board.is_game_over()
            or self.state.white_time <= 0
            or self.state.black_time <= 0
        )

    def get_result(self) -> Optional[str]:
        """Get game result in standard notation (1-0, 0-1, 1/2-1/2)"""
        if not self.is_game_over():
            return None
        if self.state.white_time <= 0:
            return "0-1"
        elif self.state.black_time <= 0:
            return "1-0"
        else:
            res = self.state.board.result()
            return res if res != "*" else None

    def get_observation(self) -> np.ndarray:
        """
        Create neural network input from game state

        Returns:
            8x8x15 observation tensor:
            - Channels 0-5: White pieces (P,R,N,B,Q,K)
            - Channels 6-11: Black pieces (P,R,N,B,Q,K)
            - Channel 12: Current player to move
            - Channel 13: Current player's remaining time (normalized)
            - Channel 14: Opponent's remaining time (normalized)
        """
        b = self.state.board
        obs = np.zeros((8, 8, 15), dtype=np.float32)

        # Map chess piece types to channel indices
        piece_map = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }

        # Encode piece positions
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if p:
                r, c = divmod(sq, 8)
                ch = piece_map[p.piece_type] + (6 if p.color == chess.BLACK else 0)
                obs[r, c, ch] = 1.0

        # Encode game metadata
        obs[:, :, 12] = 1.0 if b.turn else 0.0  # Current player
        current_time = self.state.white_time if b.turn else self.state.black_time
        opp_time = self.state.black_time if b.turn else self.state.white_time
        obs[:, :, 13] = current_time / self.time_limit  # Normalized time
        obs[:, :, 14] = opp_time / self.time_limit

        return obs

    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices for current position"""
        return [self._move_to_action(m) for m in self.state.board.legal_moves]

    def _move_to_action(self, move: chess.Move) -> int:
        """Convert chess move to action index (from_square * 64 + to_square)"""
        return move.from_square * 64 + move.to_square

    def _action_to_move(self, action: int) -> chess.Move:
        """Convert action index back to chess move, handling pawn promotion"""
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)

        # Handle pawn promotion (auto-promote to queen)
        p = self.state.board.piece_at(from_sq)
        if p and p.piece_type == chess.PAWN:
            rank = chess.square_rank(to_sq)
            if rank == 7 or rank == 0:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return move

    def _calc_material_delta(self, old_board: chess.Board, move: chess.Move) -> float:
        """Calculate immediate tactical reward from move"""
        reward = 0.0

        # Reward captures based on piece value
        if old_board.is_capture(move):
            captured = old_board.piece_at(move.to_square)
            if captured:
                reward += self.piece_values.get(captured.piece_type, 0) * 0.1

        # Bonus for tactical elements
        tmp = old_board.copy()
        tmp.push(move)
        if tmp.is_check():
            reward += 0.05
        if move.promotion:
            reward += 0.3
        if old_board.is_castling(move):
            reward += 0.1

        return reward

    def _calc_reward(self, old_board, move, think_time) -> Tuple[float, bool, Dict]:
        """
        Calculate reward for the current move and check game termination

        Reward structure:
        - Terminal rewards: ±15 for win/loss, 0 for draw
        - Material rewards: scaled by piece values
        - Small penalty for each move to encourage efficiency
        - Penalty for very long games
        """
        board = self.state.board

        # Handle game termination
        if board.is_game_over():
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.termination == chess.Termination.CHECKMATE:
                    reward = 15.0 if outcome.winner else -15.0
                    return reward, True, {"reason": "checkmate", "winner": "white" if outcome.winner else "black"}
                elif outcome.termination in {
                    chess.Termination.STALEMATE,
                    chess.Termination.INSUFFICIENT_MATERIAL,
                    chess.Termination.SEVENTYFIVE_MOVES,
                    chess.Termination.FIVEFOLD_REPETITION,
                    chess.Termination.FIFTY_MOVES,
                    chess.Termination.THREEFOLD_REPETITION
                }:
                    return 0.0, True, {"reason": "draw", "type": outcome.termination.name}
                else:
                    reward = 15.0 if outcome.winner else -15.0
                    return reward, True, {"reason": "timeout_or_resignation",
                                          "winner": "white" if outcome.winner else "black"}
            return 0.0, True, {"reason": "unknown_game_over"}

        # Calculate step reward
        r = self._calc_material_delta(old_board, move)
        r = np.clip(r, -1.5, 1.5)
        r -= 0.02  # Small move cost to encourage efficiency

        # Penalty for very long games
        if self.state.move_count > 120:
            r -= 0.01

        r = float(np.clip(r, -2.0, 2.0))
        return r, False, {"reason": "continue", "material": r}
    
    def get_move_count(self) -> int:
        """Get the current move count in the game"""
        return self.state.move_count
