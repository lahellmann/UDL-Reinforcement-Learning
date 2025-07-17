import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
import chess
import numpy as np
import os

# setup folders
os.makedirs("models", exist_ok=True)

### Environment ####

# src/bullet_chess_rl/environment/bullet_chess_env.py

@dataclass
class GameState:
   """Represents the complete state of a bullet chess game."""
   board: chess.Board  # Current chess board position
   white_time: float  # Remaining time for white player in seconds
   black_time: float  # Remaining time for black player in seconds
   move_count: int  # Total number of moves played
   last_move_time: float  # Timestamp of the last move
   game_start_time: float  # Timestamp when the game started


class BulletChessEnv:
   """
   1-minute bullet chess environment for RL training.

   Key differences from standard chess:
   - Time pressure affects reward structure
   - Games end on time flags
   - Move quality vs speed trade-off
   """

   def __init__(self, time_limit: int = 60, increment: float = 0.0):
       """
       Args:
           time_limit: Total time per player in seconds (default: 60 for 1-minute bullet)
           increment: Time increment per move in seconds (default: 0 for bullet)
       """
       self.time_limit = time_limit  # Store the time limit for each player
       self.increment = increment  # Store the time increment per move
       self.reset()  # Initialize the game state

   def reset(self) -> np.ndarray:
       """Reset environment to starting position."""
       # Create a new game state with starting position and full time
       self.game_state = GameState(
           board=chess.Board(),  # Fresh chess board in starting position
           white_time=self.time_limit,  # Full time allocation for white
           black_time=self.time_limit,  # Full time allocation for black
           move_count=0,  # No moves played yet
           last_move_time=time.time(),  # Current timestamp
           game_start_time=time.time()  # Current timestamp as game start
       )
       return self.get_observation()  # Return initial board state

   def step(self, action: Union[str, int], think_time: float = 0.1) -> Tuple[np.ndarray, float, bool, Dict]:
       """
       Execute a move and return (observation, reward, done, info).

       Args:
           action: Either UCI string or action index
           think_time: Time spent thinking about this move

       Returns:
           observation: Current board state representation
           reward: Reward for this move
           done: Whether game is finished
           info: Additional information
       """
       # Check if game is already over
       if self.is_game_over():
           return self.get_observation(), 0, True, {"reason": "game_already_over"}

       # Convert action to chess move format
       if isinstance(action, int):
           move = self._action_to_move(action)  # Convert integer action to move
       else:
           try:
               move = chess.Move.from_uci(action)  # Parse UCI string to move
           except ValueError:
               return self.get_observation(), -1, True, {"reason": "invalid_uci"}

       # Validate if the move is legal
       if move not in self.game_state.board.legal_moves:
           return self.get_observation(), -1, True, {"reason": "illegal_move"}

       # Calculate actual time spent on this move
       current_time = time.time()
       actual_think_time = current_time - self.game_state.last_move_time

       # Deduct time from the current player's clock
       if self.game_state.board.turn:  # White to move
           self.game_state.white_time -= actual_think_time
           remaining_time = self.game_state.white_time
       else:  # Black to move
           self.game_state.black_time -= actual_think_time
           remaining_time = self.game_state.black_time

       # Check if player ran out of time
       if remaining_time <= 0:
           reward = -1 if self.game_state.board.turn else 1  # Lose if you flag
           return self.get_observation(), reward, True, {"reason": "time_flag"}

       # Execute the move on the board
       self.game_state.board.push(move)
       self.game_state.move_count += 1  # Increment move counter
       self.game_state.last_move_time = current_time  # Update timestamp

       # Add time increment after the move (if any)
       if self.game_state.board.turn:  # Now it's white's turn (black just moved)
           self.game_state.black_time += self.increment
       else:  # Now it's black's turn (white just moved)
           self.game_state.white_time += self.increment

       # Calculate reward and check if game is finished
       reward, done, info = self._calculate_reward(remaining_time, actual_think_time)

       return self.get_observation(), reward, done, info

   def get_observation(self) -> np.ndarray:
       """
       Convert current game state to neural network input.

       Returns:
           8x8x15 representation:
           - 12 channels for pieces (6 types × 2 colors)
           - 1 channel for turn
           - 1 channel for time pressure (current player)
           - 1 channel for time pressure (opponent)
       """
       state = np.zeros((8, 8, 15), dtype=np.float32)  # Initialize empty state tensor

       # Map piece types to channel indices
       piece_map = {
           chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
           chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
       }

       # Encode pieces on the board
       for square in chess.SQUARES:
           piece = self.game_state.board.piece_at(square)
           if piece:
               row, col = divmod(square, 8)  # Convert square to row/column
               channel = piece_map[piece.piece_type] + (6 if piece.color else 0)  # White pieces +6 offset
               state[row, col, channel] = 1

       # Encode whose turn it is
       state[:, :, 12] = 1 if self.game_state.board.turn else 0

       # Encode time pressure for both players (normalized 0-1)
       current_time = (self.game_state.white_time if self.game_state.board.turn
                      else self.game_state.black_time)
       opponent_time = (self.game_state.black_time if self.game_state.board.turn
                       else self.game_state.white_time)

       state[:, :, 13] = current_time / self.time_limit  # Current player's time ratio
       state[:, :, 14] = opponent_time / self.time_limit  # Opponent's time ratio

       return state

   def get_legal_actions(self) -> List[int]:
       """Get list of legal action indices."""
       return [self._move_to_action(move) for move in self.game_state.board.legal_moves]

   def get_time_pressure_level(self) -> str:
       """Categorize current time pressure level."""
       current_time = (self.game_state.white_time if self.game_state.board.turn
                      else self.game_state.black_time)

       # Categorize time pressure based on remaining time
       if current_time > 30:
           return "relaxed"  # More than 30 seconds
       elif current_time > 10:
           return "moderate"  # 10-30 seconds
       elif current_time > 5:
           return "pressure"  # 5-10 seconds
       else:
           return "scramble"  # Less than 5 seconds

   def get_game_info(self) -> Dict:
       """Get comprehensive game information."""
       return {
           "white_time": self.game_state.white_time,
           "black_time": self.game_state.black_time,
           "move_count": self.game_state.move_count,
           "time_pressure_level": self.get_time_pressure_level(),
           "game_duration": time.time() - self.game_state.game_start_time,
           "board_fen": self.game_state.board.fen(),  # Board position in FEN notation
           "legal_moves": len(list(self.game_state.board.legal_moves))  # Number of legal moves
       }

   def is_game_over(self) -> bool:
       """Check if game is over (checkmate, stalemate, or time)."""
       return (self.game_state.board.is_game_over() or  # Standard chess endings
               self.game_state.white_time <= 0 or  # White out of time
               self.game_state.black_time <= 0)  # Black out of time

   def get_result(self) -> Optional[str]:
       """Get game result if finished."""
       if not self.is_game_over():
           return None  # Game still in progress

       # Check for time-based results first
       if self.game_state.white_time <= 0:
           return "0-1"  # Black wins on time
       elif self.game_state.black_time <= 0:
           return "1-0"  # White wins on time
       else:
           # Use standard chess result
           result = self.game_state.board.result()
           return result if result != "*" else None

   def _action_to_move(self, action: int) -> chess.Move:
       """Convert action index to chess move."""
       # Simple encoding: from_square * 64 + to_square
       from_square = action // 64  # Source square (0-63)
       to_square = action % 64  # Destination square (0-63)

       move = chess.Move(from_square, to_square)

       # Handle pawn promotion (auto-promote to queen)
       if (self.game_state.board.piece_at(from_square) and
           self.game_state.board.piece_at(from_square).piece_type == chess.PAWN and
           (chess.square_rank(to_square) == 7 or chess.square_rank(to_square) == 0)):
           move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

       return move

   def _move_to_action(self, move: chess.Move) -> int:
       """Convert chess move to action index."""
       return move.from_square * 64 + move.to_square  # Encode as single integer

   def _calculate_reward(self, remaining_time: float, think_time: float) -> Tuple[float, bool, Dict]:
       """Calculate reward based on game state and time usage."""

       # Check for game-ending conditions
       if self.game_state.board.is_checkmate():
           winner = "white" if not self.game_state.board.turn else "black"  # Winner is opposite of current turn
           reward = 1 if winner == "white" else -1
           return reward, True, {"reason": "checkmate", "winner": winner}

       elif self.game_state.board.is_stalemate():
           return 0, True, {"reason": "stalemate"}  # Draw

       elif self.game_state.board.is_insufficient_material():
           return 0, True, {"reason": "insufficient_material"}  # Draw

       elif self.game_state.board.is_seventyfive_moves():
           return 0, True, {"reason": "75_move_rule"}  # Draw

       elif self.game_state.board.is_fivefold_repetition():
           return 0, True, {"reason": "repetition"}  # Draw

       else:
           # Game continues - calculate intermediate rewards
           time_efficiency_bonus = 0.001 * (remaining_time / self.time_limit)  # Bonus for having time left

           # Penalty for using too much time when low on time
           time_penalty = 0
           if remaining_time < 10 and think_time > 2:  # Spending >2s when <10s left
               time_penalty = -0.01

           reward = time_efficiency_bonus + time_penalty
           return reward, False, {"reason": "continue", "time_remaining": remaining_time}

   def render(self) -> str:
       """Render current game state."""
       board_str = str(self.game_state.board)  # ASCII board representation
       time_str = f"White: {self.game_state.white_time:.1f}s | Black: {self.game_state.black_time:.1f}s"
       pressure_str = f"Time pressure: {self.get_time_pressure_level()}"

       return f"{board_str}\n{time_str}\n{pressure_str}"

   def copy(self):
       """Create a copy of the current environment state."""
       new_env = BulletChessEnv(self.time_limit, self.increment)  # Create new environment
       # Copy all game state information
       new_env.game_state = GameState(
           board=self.game_state.board.copy(),  # Deep copy of board
           white_time=self.game_state.white_time,
           black_time=self.game_state.black_time,
           move_count=self.game_state.move_count,
           last_move_time=self.game_state.last_move_time,
           game_start_time=self.game_state.game_start_time
       )
       return new_env