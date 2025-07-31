# SECTION 4: TRAINING SYSTEM
import os
import random
import logging
from datetime import datetime
from collections import deque, Counter
from typing import List, Tuple, Dict

import numpy as np
import utils
from environment import BulletChessEnv
from ddqn_agent import DDQNAgent

import chess

class BulletChessDDQNTrainer:
    """
    Complete training system for chess DDQN agent

    Features:
    - Self-play against configurable opponent
    - Progressive opponent strength curriculum
    - Comprehensive evaluation and logging
    - Model checkpointing and best model tracking
    - Detailed game statistics and analysis
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        utils.seed_everything(cfg.get("seed", 42))

        # Initialize environment
        self.env = BulletChessEnv(
            time_limit=cfg["env"]["time_limit"],
            increment=cfg["env"]["increment"],
            simulate_think=cfg["env"]["simulate_think"],
            think_lo=cfg["env"]["think_lo"],
            think_hi=cfg["env"]["think_hi"]
        )

        # Initialize agent
        self.agent = DDQNAgent(
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            epsilon_start=cfg["agent"]["epsilon_start"],
            epsilon_end=cfg["agent"]["epsilon_end"],
            epsilon_decay=cfg["agent"]["epsilon_decay"],
            batch_size=cfg["agent"]["batch_size"],
            target_update=cfg["agent"]["target_update"],
            tau=cfg["agent"]["tau"],
            use_soft_update=cfg["agent"]["use_soft_update"],
            n_actions=4096,
            per_alpha=cfg["replay"]["per_alpha"],
            per_beta_start=cfg["replay"]["per_beta_start"],
            per_beta_frames=cfg["replay"]["per_beta_frames"],
            n_step=cfg["agent"]["n_step"],
            dropout=cfg["agent"]["dropout"],
            noise_std=cfg["agent"]["noise_std"]
        )

        self.setup_logging()

        # Training statistics
        self.best_win_rate = -1.0
        self.stats = {
            "wins": 0, "losses": 0, "draws": 0,
            "wins_white": 0, "wins_black": 0,
            "losses_white": 0, "losses_black": 0,
            "illegal": 0
        }
        self.ended_by = Counter()

        # Opponent curriculum parameters
        self.opp_strength_start = cfg["opponent"]["strength_start"]
        self.opp_strength_end = cfg["opponent"]["strength_end"]

    def setup_logging(self):
        """Initialize logging system with file and console output"""
        os.makedirs(self.cfg["paths"]["models"], exist_ok=True)
        os.makedirs(self.cfg["paths"]["logs"], exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.cfg["paths"]["logs"], f"ddqn_truly_fixed_{timestamp}.log")

        # Clear existing handlers
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger("ddqn")

    def select_opponent_move(self, legal_actions: List[int], strength: float) -> int:
        """
        Rule-based opponent with configurable strength

        Args:
            legal_actions: List of legal action indices
            strength: Opponent skill level (0.0 = random, 1.0 = perfect tactical play)

        Returns:
            Selected action index
        """
        # Random move with probability (1 - strength)
        if random.random() > strength:
            return random.choice(legal_actions)

        board = self.env.state.board
        best = None
        best_score = -1e9

        # Define important squares for positional evaluation
        center = {chess.D4, chess.E4, chess.D5, chess.E5}
        ext_center = {
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D6, chess.E3, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        }

        # Evaluate each legal move
        for a in legal_actions:
            from_sq, to_sq = a // 64, a % 64
            move = chess.Move(from_sq, to_sq)
            score = 0.0

            # Prioritize captures by material value
            if board.is_capture(move):
                captured = board.piece_at(to_sq)
                if captured:
                    pv = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                    score += pv.get(captured.piece_type, 0) * 10

            # Test move consequences
            tmp = board.copy()
            tmp.push(move)

            # Bonus for checks and checkmate
            if tmp.is_check():
                score += 3.0
            if tmp.is_checkmate():
                score += 1000.0

            # Positional bonuses for center control
            if to_sq in center:
                score += 1.0
            elif to_sq in ext_center:
                score += 0.5

            # Avoid hanging pieces (basic safety check)
            piece = board.piece_at(from_sq)
            if piece and not board.is_capture(move):
                tmp2 = board.copy()
                tmp2.push(move)
                if tmp2.is_attacked_by(not tmp2.turn, to_sq):
                    pv = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                    score -= pv.get(piece.piece_type, 0) * 3

            # Add small random factor for variety
            score += random.uniform(-0.1, 0.1)

            if score > best_score:
                best_score = score
                best = a

        return best if best is not None else random.choice(legal_actions)

    def play_one(self, is_training: bool, episode_idx: int, total_episodes: int,
                 force_eval_strength: float = None) -> Dict:
        """
        Play one complete game between agent and opponent

        Args:
            is_training: Whether to collect training data
            episode_idx: Current episode number for curriculum
            total_episodes: Total episodes for curriculum scheduling
            force_eval_strength: Override opponent strength for evaluation

        Returns:
            Game result dictionary with statistics
        """
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        ended_by = "unknown"

        # Randomly assign colors
        agent_is_white = random.choice([True, False])

        # Calculate opponent strength from curriculum
        if force_eval_strength is not None:
            opp_strength = force_eval_strength
        else:
            opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end,
                                         episode_idx, total_episodes)

        # Handle first move if opponent plays first
        if (self.env.state.board.turn and not agent_is_white) or \
           ((not self.env.state.board.turn) and agent_is_white):
            legal_opp = self.env.get_legal_actions()
            if legal_opp:
                opp_action = self.select_opponent_move(legal_opp, opp_strength)
                state, _, done, info = self.env.step(opp_action)
                steps += 1
                if done:
                    ended_by = self._map_reason(info.get("reason", "unknown"))
                    return self._finalize_episode(total_reward, steps, agent_is_white, ended_by)

        # Reset n-step buffer for new episode
        if is_training:
            self.agent.nstep_buffer.reset()

        # Main game loop
        while steps < self.cfg["training"]["max_moves"] and not self.env.is_game_over():
            # Agent's turn
            legal_agent = self.env.get_legal_actions()
            if not legal_agent:
                ended_by = "no_legal_agent"
                break

            eval_mode = not is_training
            action = self.agent.select_action(state, legal_agent, eval_mode=eval_mode)
            next_state_after_agent, r_agent_raw, done_agent, info_agent = self.env.step(action)
            steps += 1

            # Adjust reward sign based on agent's color
            r_agent = r_agent_raw if agent_is_white else -r_agent_raw

            if done_agent:
                ended_by = self._map_reason(info_agent.get("reason", "unknown"))
                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                if is_training:
                    self.agent.store_transition(state, action, r_agent, next_state_after_agent, True, mask)
                total_reward += r_agent
                state = next_state_after_agent
                break

            # Opponent's turn
            legal_opp = self.env.get_legal_actions()
            if not legal_opp:
                ended_by = "no_legal_opponent"
                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                if is_training:
                    self.agent.store_transition(state, action, r_agent, next_state_after_agent, True, mask)
                total_reward += r_agent
                state = next_state_after_agent
                break

            opp_action = self.select_opponent_move(legal_opp, opp_strength)
            next_state, r_opp_raw, done_opp, info_opp = self.env.step(opp_action)
            steps += 1

            if done_opp:
                # Calculate terminal bonus based on game result
                result = self.env.get_result()
                terminal_bonus = 0.0
                if result == "1-0":
                    terminal_bonus = 15.0 if agent_is_white else -15.0
                elif result == "0-1":
                    terminal_bonus = 15.0 if not agent_is_white else -15.0
                else:
                    terminal_bonus = 0.0

                final_r = r_agent + terminal_bonus
                ended_by = self._map_reason(info_opp.get("reason", "unknown"))

                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                if is_training:
                    self.agent.store_transition(state, action, final_r, next_state, True, mask)
                total_reward += final_r
                state = next_state
                break
            else:
                # Continue game - prepare legal action mask for next state
                legal_next_agent = self.env.get_legal_actions()
                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                if legal_next_agent:
                    mask[np.array(legal_next_agent, dtype=np.int32)] = True

                if is_training:
                    self.agent.store_transition(state, action, r_agent, next_state, False, mask)

                total_reward += r_agent
                state = next_state

        # Handle move limit penalty
        if steps >= self.cfg["training"]["max_moves"] and ended_by == "unknown":
            ended_by = "move_limit"
            if is_training:
                final_pen = self.cfg["training"]["move_limit_penalty"]
                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                self.agent.store_transition(state, action if 'action' in locals() else 0,
                                          final_pen, state, True, mask)
                total_reward += final_pen

        return self._finalize_episode(total_reward, steps, agent_is_white, ended_by)

    def _map_reason(self, reason: str) -> str:
        """Map detailed termination reasons to categories"""
        if "checkmate" in reason:
            return "mate"
        if "time_flag" in reason:
            return "time_flag"
        if "draw" in reason:
            return "draw"
        if "illegal" in reason:
            return "illegal"
        if "timeout_or_resignation" in reason:
            return "timeout_or_resignation"
        return reason

    def _finalize_episode(self, total_reward, steps, agent_is_white, ended_by) -> Dict:
        """Process episode end and update statistics"""
        result = self.env.get_result()
        outcome = "draw"

        # Determine outcome from agent's perspective
        if result == "1-0":
            outcome = "win" if agent_is_white else "loss"
        elif result == "0-1":
            outcome = "win" if not agent_is_white else "loss"

        # Update statistics
        if outcome == "win":
            self.stats["wins"] += 1
            if agent_is_white:
                self.stats["wins_white"] += 1
            else:
                self.stats["wins_black"] += 1
        elif outcome == "loss":
            self.stats["losses"] += 1
            if agent_is_white:
                self.stats["losses_white"] += 1
            else:
                self.stats["losses_black"] += 1
        else:
            self.stats["draws"] += 1

        if ended_by:
            self.ended_by[ended_by] += 1

        return {
            "reward": total_reward,
            "steps": steps,
            "outcome": outcome,
            "result": result,
            "agent_white": agent_is_white,
            "ended_by": ended_by
        }

    def train(self):
        """
        Main training loop with curriculum learning and evaluation

        Training features:
        - Progressive opponent strength curriculum
        - Regular evaluation against stronger opponents
        - Best model tracking and checkpointing
        - Comprehensive logging of training progress
        """
        episodes = self.cfg["training"]["episodes"]
        warmup = self.cfg["training"]["warmup"]
        log_freq = self.cfg["training"]["log_freq"]
        eval_freq = self.cfg["training"]["eval_freq"]
        save_freq = self.cfg["training"]["save_freq"]

        losses = deque(maxlen=200)

        for ep in range(1, episodes + 1):
            # Play training game
            res = self.play_one(is_training=True, episode_idx=ep, total_episodes=episodes)

            # Perform learning update after warmup period
            loss_val = 0.0
            if ep >= warmup:
                loss_val = self.agent.update()
                if loss_val:
                    losses.append(loss_val)

            # Calculate statistics
            total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
            wr_all = self.stats["wins"] / max(1, total_games)  # Win rate including draws
            wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))  # Win rate excluding draws

            # Periodic logging
            if ep % log_freq == 0 or ep == 1:
                opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end,
                                           ep, episodes)
                ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)
                self.logger.info(
                    f"Ep {ep:4d}/{episodes} | {res['outcome']:4s} | "
                    f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                    f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                    f"L:{(np.mean(losses) if losses else 0):.4f} | "
                    f"eps:{self.agent.eps:.3f} | buf:{len(self.agent.per_buffer)} | "
                    f"Ww:{self.stats['wins_white']} Wb:{self.stats['wins_black']} "
                    f"Lw:{self.stats['losses_white']} Lb:{self.stats['losses_black']} | "
                    f"opp_str:{opp_strength:.3f} | move_limit_rate:{ml_rate:.3f}"
                )

            # Periodic evaluation
            if ep % eval_freq == 0:
                eval_strength = min(0.25, opp_strength + 0.05)
                wr, wr_true_eval = self.evaluate(self.cfg["training"]["eval_games"],
                                                ep, episodes, eval_strength)
                # Save best model
                if wr > self.best_win_rate:
                    self.best_win_rate = wr
                    best_path = os.path.join(self.cfg["paths"]["models"], f"best_ddqn_truly_fixed_ep{ep}.pth")
                    self.agent.save(best_path)
                    self.logger.info(f"New best model saved (win_rate={wr:.3f}) -> {best_path}")

            # Periodic checkpointing
            if ep % save_freq == 0:
                ckpt_path = os.path.join(self.cfg["paths"]["models"], f"ckpt_ddqn_truly_fixed_ep{ep}.pth")
                self.agent.save(ckpt_path)
                self.logger.info(f"Checkpoint saved -> {ckpt_path}")

        # Final comprehensive evaluation across multiple opponent strengths
        self.logger.info("=== FINAL EVALUATION ===")
        for strength in [0.1, 0.15, 0.2, 0.25, 0.3]:
            wr, wr_true = self.evaluate(30, episodes, episodes, force_strength=strength)
            self.logger.info(f"vs {strength:.2f} strength: WR={wr:.3f} (true={wr_true:.3f})")

    def evaluate(self, n_games: int, ep: int, total_episodes: int,
                 force_strength: float = None) -> Tuple[float, float]:
        """
        Evaluate agent performance against opponent

        Args:
            n_games: Number of evaluation games
            ep: Current episode for curriculum
            total_episodes: Total episodes for curriculum
            force_strength: Override opponent strength

        Returns:
            (win_rate_all, win_rate_true) - with and without draws
        """
        wins = 0
        draws = 0
        losses = 0
        lens = []
        wins_white = wins_black = 0
        losses_white = losses_black = 0

        # Disable exploration during evaluation
        old_eps = self.agent.eps
        self.agent.eps = 0.0

        eval_strength = force_strength or min(0.25,
            utils.linear_anneal(self.opp_strength_start, self.opp_strength_end, ep, total_episodes) + 0.05)

        # Play evaluation games
        for i in range(n_games):
            res = self.play_one(is_training=False, episode_idx=ep,
                              total_episodes=total_episodes,
                              force_eval_strength=eval_strength)

            lens.append(res["steps"])

            # Log first few games for debugging
            if i < 3:
                self.logger.info(f"EVAL Game {i+1}: {res['outcome']} in {res['steps']} moves, "
                                f"ended_by: {res['ended_by']}, agent_white: {res['agent_white']}")

            # Update evaluation statistics
            if res["outcome"] == "win":
                wins += 1
                if res["agent_white"]:
                    wins_white += 1
                else:
                    wins_black += 1
            elif res["outcome"] == "loss":
                losses += 1
                if res["agent_white"]:
                    losses_white += 1
                else:
                    losses_black += 1
            else:
                draws += 1

        # Restore exploration
        self.agent.eps = old_eps

        # Calculate win rates
        wr_all = wins / max(1, (wins + losses + draws))
        wr_true = wins / max(1, (wins + losses))

        self.logger.info(f"[EVAL] vs {eval_strength:.2f} | WR_all {wr_all:.3f} | WR_true {wr_true:.3f} | "
                         f"W:{wins} L:{losses} D:{draws} | "
                         f"W_white:{wins_white} W_black:{wins_black} | "
                         f"L_white:{losses_white} L_black:{losses_black} | "
                         f"len:{np.mean(lens):.1f}")
        return wr_all, wr_true


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
            "warmup": 100,         # Episodes before learning starts
            "log_freq": 50,        # Logging frequency
            "eval_freq": 150,      # Evaluation frequency
            "eval_games": 30,      # Games per evaluation
            "save_freq": 300       # Model saving frequency
        },
        "opponent": {
            "strength_start": 0.0, # Initial opponent strength
            "strength_end": 0.3    # Final opponent strength
        },
        "paths": {
            "models": "models_ddqn_truly_fixed", # Model save directory
            "logs": "logs_ddqn_truly_fixed"      # Log save directory
        }
    }

