import os
import random
import logging
from datetime import datetime
from collections import deque, Counter
from typing import List, Tuple, Dict
import chess
import time
import numpy as np
import json
import utils


from environment import BulletChessEnv
from ddqn_agent import DDQNAgent
from policyvalue_agent import MCTSAgent


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

    def __init__(self, cfg: dict, env: BulletChessEnv):
        """
        Initializes the BulletChessDDQNTrainer with configuration and environment.
        Args:
            cfg (dict): Configuration dictionary containing training parameters.
            env (BulletChessEnv): The chess environment to interact with.
        
        Initializes the trainer with the provided configuration and environment.
        Sets up the DDQN agent, logging, and training statistics.
        """

        self.cfg = cfg
        utils.seed_everything(cfg.get("seed", 42))

        # Initialize environment
        self.env = env
       
        # Initialize agent
        self.agent = DDQNAgent(cfg, env)

        self.current_training_type = "episodes"  # Default training type
        self.setup_logging()
        self.training_log = {}

        # Training statistics
        self.best_win_rate = -1.0
        self.stats = {
            "wins": 0, "losses": 0, "draws": 0,
            "wins_white": 0, "wins_black": 0,
            "losses_white": 0, "losses_black": 0,
            "illegal": 0
        }

        
        self.ended_by = Counter()
        self.losses = []

        # Opponent curriculum parameters
        self.opp_strength_start = cfg["opponent"]["strength_start"]
        self.opp_strength_end = cfg["opponent"]["strength_end"]

    def setup_logging(self):
        """
        Initialize logging system with file and console output
        Creates necessary directories for model and log files.
        Sets up logging configuration to log messages to both a file and the console.
        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        os.makedirs(self.cfg["paths"]["models"], exist_ok=True)
        os.makedirs(self.cfg["paths"]["ddqn_logs"], exist_ok=True)

        model_name = "DDQN"
        episodes = self.cfg["training"]["episodes"]
        log_file = os.path.join(
            self.cfg["paths"]["ddqn_logs"],
            f"{model_name}_{self.current_training_type}_{episodes}.log"
        )


        # Clear existing handlers
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger("DDQNLogger")

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

    def play_one(self, is_training: bool, episode_idx: int, total_episodes: int = 30,
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
        """
        Map detailed termination reasons to categories
        Args:
            reason: Detailed reason string from environment
        Returns:
            str: Mapped category string
        """
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
        """
        Process episode end and update statistics
        Args:
            total_reward: Total reward accumulated during the episode
            steps: Number of steps taken in the episode
            agent_is_white: Whether the agent played as white
            ended_by: Reason for game termination
        Returns:
            Dictionary with episode statistics
        """
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

    def train(self, training_type: str = "episodes"):
        """
        Main training loop with curriculum learning and evaluation

        Training features:
        - Progressive opponent strength curriculum
        - Regular evaluation against stronger opponents
        - Best model tracking and checkpointing
        - Comprehensive logging of training progress

        Runs the training loop for the specified number of episodes.
        Collects training data, performs learning updates, and evaluates performance.
        Tracks statistics such as win rates, losses, and draws.
        """

        overall_time = self.cfg["training"]["overall_time"]
        if training_type == "episodes":
            episodes = self.cfg["training"]["episodes"]
        elif training_type == "time":
            episodes = 5 * (overall_time / 60)  # Assuming 5 episodes per minute
        else:
            raise ValueError(f"Invalid training type: {training_type}. Use 'episodes' or 'time'.")

        warmup = self.cfg["training"]["warmup"]
        log_freq = self.cfg["training"]["log_freq"]
        eval_freq = self.cfg["training"]["eval_freq"]
        save_freq = self.cfg["training"]["save_freq"]

        self.current_training_type = training_type

        if training_type == "episodes":
            for ep in range(1, episodes + 1):
                # Play training game
                res = self.play_one(is_training=True, episode_idx=ep, total_episodes=episodes)

                # Perform learning update after warmup period
                loss_val = 0.0
                if ep >= warmup:
                    loss_val = self.agent.update()
                    if loss_val:
                        self.losses.append(loss_val)

                # Calculate statistics
                total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                wr_all = self.stats["wins"] / max(1, total_games)  # Win rate including draws
                wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))  # Win rate excluding draws

                # Update training log
                self.training_log[ep] = {
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "outcome": res["outcome"],
                    "result": res["result"],
                    "agent_white": res["agent_white"],
                    "ended_by": res["ended_by"],
                    "loss": loss_val,
                    "win_rate_all": wr_all,
                    "win_rate_true": wr_true
                }

                # Periodic logging
                if ep % log_freq == 0 or ep == 1:
                    opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end,
                                            ep, episodes)
                    ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)
                    self.logger.info(
                        f"Ep {ep:4d}/{episodes} | {res['outcome']:4s} | "
                        f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                        f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                        f"L:{(np.mean(self.losses) if self.losses else 0):.4f} | "
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
                        best_path = os.path.join(self.cfg["paths"]["models"],f"/{self.current_training_type}/",f"best_ddqn_truly_fixed_ep{ep}.pth")
                        self.agent.save(best_path)
                        self.logger.info(f"New best model saved (win_rate={wr:.3f}) -> {best_path}")

                # Periodic checkpointing
                if ep % save_freq == 0:
                    ckpt_path = os.path.join(self.cfg["paths"]["models"], f"ckpt_ddqn_truly_fixed_ep{ep}.pth")
                    self.agent.save(ckpt_path)
                    self.logger.info(f"Checkpoint saved -> {ckpt_path}")
                elif type == "time":
                    end_time = time.time() + overall_time
                    ep = 0

                    while time.time() <= end_time:
                        ep += 1
                        res = self.play_one(is_training=True, episode_idx=ep, total_episodes=0)

                        # Training
                        loss_val = 0.0
                        if ep >= warmup:
                            loss_val = self.agent.update()
                            if loss_val:
                                self.losses.append(loss_val)

                        # Stats
                        total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                        wr_all = self.stats["wins"] / max(1, total_games)
                        wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))

                        # Logging
                        if ep % log_freq == 0 or ep == 1:
                            elapsed_time = time.time() - (end_time - overall_time)
                            opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end, ep, overall_time)
                            ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)

                            self.logger.info(
                                f"Time {elapsed_time:.2f}/{overall_time:.2f} | {res['outcome']:4s} | "
                                f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                                f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                                f"L:{(np.mean(self.losses) if self.losses else 0):.4f} | "
                                f"eps:{self.agent.eps:.3f} | buf:{len(self.agent.per_buffer)} | "
                                f"Ww:{self.stats['wins_white']} Wb:{self.stats['wins_black']} "
                                f"Lw:{self.stats['losses_white']} Lb:{self.stats['losses_black']} | "
                                f"opp_str:{opp_strength:.3f} | move_limit_rate:{ml_rate:.3f}"
                            )

                        # Evaluation
                        if ep % eval_freq == 0:
                            eval_strength = min(0.25, opp_strength + 0.05)
                            wr, wr_true_eval = self.evaluate(self.cfg["training"]["eval_games"], ep, overall_time, eval_strength)
                            if wr > self.best_win_rate:
                                self.best_win_rate = wr
                                best_path = os.path.join(self.cfg["paths"]["models"], f"best_ddqn_time_ep{ep}.pth")
                                self.agent.save(best_path)
                                self.logger.info(f"New best model saved (win_rate={wr:.3f}) -> {best_path}")

                        # Checkpoint
                        if ep % save_freq == 0:
                            ckpt_path = os.path.join(self.cfg["paths"]["models"], f"ckpt_ddqn_time_ep{ep}.pth")
                            self.agent.save(ckpt_path)
                            self.logger.info(f"Checkpoint saved -> {ckpt_path}")

        elif training_type == "time":
            end_time = time.time() + overall_time
            ep = 0

            while time.time() <= end_time:
                ep += 1
                res = self.play_one(is_training=True, episode_idx=ep, total_episodes=0)

                # Training
                loss_val = 0.0
                if ep >= warmup:
                    loss_val = self.agent.update()
                    if loss_val:
                        self.losses.append(loss_val)

                # Stats
                total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                wr_all = self.stats["wins"] / max(1, total_games)
                wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))

                # Update training log
                self.training_log[ep] = {
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "outcome": res["outcome"],
                    "result": res["result"],
                    "agent_white": res["agent_white"],
                    "ended_by": res["ended_by"],
                    "loss": loss_val,
                    "win_rate_all": wr_all,
                    "win_rate_true": wr_true
                }

                # Logging
                if ep % log_freq == 0 or ep == 1:
                    elapsed_time = time.time() - (end_time - overall_time)
                    opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end, ep, episodes)
                    ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)

                    self.logger.info(
                        f"Time {elapsed_time:.2f}/{overall_time:.2f} | {res['outcome']:4s} | "
                        f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                        f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                        f"L:{(np.mean(self.losses) if self.losses else 0):.4f} | "
                        f"eps:{self.agent.eps:.3f} | buf:{len(self.agent.per_buffer)} | "
                        f"Ww:{self.stats['wins_white']} Wb:{self.stats['wins_black']} "
                        f"Lw:{self.stats['losses_white']} Lb:{self.stats['losses_black']} | "
                        f"opp_str:{opp_strength:.3f} | move_limit_rate:{ml_rate:.3f}"
                    )

                # Evaluation
                if ep % eval_freq == 0:
                    eval_strength = min(0.25, opp_strength + 0.05)
                    wr, wr_true_eval = self.evaluate(self.cfg["training"]["eval_games"], ep, overall_time, eval_strength)
                    if wr > self.best_win_rate:
                        self.best_win_rate = wr
                        best_path = os.path.join(self.cfg["paths"]["models"], f"best_ddqn_time_ep{ep}.pth")
                        self.agent.save(best_path)
                        self.logger.info(f"New best model saved (win_rate={wr:.3f}) -> {best_path}")

                # Checkpoint
                if ep % save_freq == 0:
                    ckpt_path = os.path.join(self.cfg["paths"]["models"], f"ckpt_ddqn_time_ep{ep}.pth")
                    self.agent.save(ckpt_path)
                    self.logger.info(f"Checkpoint saved -> {ckpt_path}")


        # Final comprehensive evaluation across multiple opponent strengths
        self.logger.info("=== FINAL EVALUATION ===")
        for strength in [0.1, 0.15, 0.2, 0.25, 0.3]:
            wr, wr_true = self.evaluate(30, episodes, episodes, force_strength=strength)
            self.logger.info(f"vs {strength:.2f} strength: WR={wr:.3f} (true={wr_true:.3f})")

        log_path = os.path.join(
            self.cfg["paths"]["ddqn_logs"],
            f"DDQN_{self.current_training_type}_{episodes}.json"
        )
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)

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

    def get_losses(self) -> Dict[int, float]:
        """
        Get training losses recorded during training
        Returns:
            List of losses recorded during training
        """
        return self.losses
    
    def get_ended_by(self) -> Dict[str, int]:
        """
        Get reasons for game endings
        Returns:
            Counter mapping reason to count
        """
        return dict(self.ended_by)
    

class BulletChessAlphaZeroTrainer:
    """Complete training system for AlphaZero-style MCTS agent
    Features:
    - Self-play with MCTS-guided moves
    - Progressive opponent strength curriculum
    - Comprehensive evaluation and logging
    - Model checkpointing and best model tracking
    - Detailed game statistics and analysis
    """
    def __init__(self, cfg: dict, env: BulletChessEnv):
        """
        Initializes the BulletChessAlphaZeroTrainer with configuration and environment.
        Args:
            cfg (dict): Configuration dictionary containing training parameters.
            env (BulletChessEnv): The chess environment to interact with.
        
        Initializes the trainer with the provided configuration and environment.
        Sets up the MCTS agent, replay buffer, and logging.
        Also initializes training statistics and opponent curriculum parameters.
        Features:
            - MCTS agent for self-play
            - Replay buffer for training data
            - Logging system for tracking progress
            - Statistics for game outcomes and performance
            - Opponent strength curriculum for progressive training
        """
        self.cfg = cfg
        self.env = env
        self.agent = MCTSAgent(cfg, env)
        self.replay_buffer = []  # stores (state, policy, value) tuples
        self.max_buffer_size = cfg["replay"]["capacity"]
        self.batch_size = cfg["agent"]["batch_size"]
        self.epochs = cfg["training"]["episodes"]
        self.best_win_rate = -1.0
        

        self.current_training_type = "episodes"  # Default training type
        utils.seed_everything(cfg.get("seed", 42))
        self.training_log = {}
        self.logger = self.setup_logging()

        # Initialise training statistics
        self.best_win_rate = -1.0
        self.stats = {
            "wins": 0, "losses": 0, "draws": 0,
            "wins_white": 0, "wins_black": 0,
            "losses_white": 0, "losses_black": 0,
            "illegal": 0, "games_played": 0
        }

        
        self.ended_by = Counter() # Counter to track reasons for game endings
        self.losses = []

        # Opponent curriculum parameters
        self.opp_strength_start = cfg["opponent"]["strength_start"]
        self.opp_strength_end = cfg["opponent"]["strength_end"]

    def setup_logging(self):
        """Initialize logging system with file and console output
        
        Creates necessary directories for model and log files.
        Sets up logging configuration to log messages to both a file and the console.
        Returns:
            logger (logging.Logger): Configured logger instance.
        """
        os.makedirs(self.cfg["paths"]["models"], exist_ok=True)
        os.makedirs(self.cfg["paths"]["mcts_logs"], exist_ok=True)

        model_name = "policy_value"
        episodes = self.cfg["training"]["episodes"]
        mcts_simulations = self.cfg["mcts"]["mcts_simulations"]
        log_file = os.path.join(
            self.cfg["paths"]["mcts_logs"],
            f"{model_name}_{self.current_training_type}_{episodes}_mcts:{mcts_simulations}.log"
        )


        # Clear existing handlers
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger("PolicyValueLogger")
        return self.logger

    def self_play_game(self, is_training:bool = True):
        """Play one full game using MCTS-guided moves, collecting training data.
        Runs a complete self-play game where the agent plays against itself using MCTS.
        Collects training examples in the form of (state, policy, value) tuples.
        Returns:
            training_data (List[Tuple]): List of training examples collected during the game.
            result (str): Game result in standard format (e.g. '1-0', '0-1', '1/2-1/2').
            reason (str): Reason for game end (e.g. 'checkmate', 'time_flag', 'draw').
            current_player (int): Current player's perspective at game end (+1 for white, -1 for black).
        """
        state = self.env.reset()
        done = False
        training_examples = []
        current_player = 1  # +1 for white, -1 for black
        total_reward = 0.0
        steps = 0

        while not done and steps < self.cfg["training"]["max_moves"]:
            board_state = state  # Store current board state
            # Run MCTS to get action probabilities and selected action
            pi = self.agent.run_mcts(self.env)  # pi is array of action probabilities
            action = self.agent.select_action_from_pi(pi) # Select action based on MCTS policy
            steps += 1

            # Store (state, pi, current_player) for training after game ends
            training_examples.append((board_state, pi, current_player))

            # Step environment
            state, reward, done, info = self.env.step(action)
            r_agent = reward if current_player == 1 else -reward
            current_player = -current_player  # switch perspective after each move
            total_reward += r_agent

        # Handle move limit penalty
        if steps >= self.cfg["training"]["max_moves"] and ended_by == "unknown":
            ended_by = "move_limit"
            if is_training:
                final_pen = self.cfg["training"]["move_limit_penalty"]
                mask = np.zeros((self.agent.n_actions,), dtype=np.bool_)
                self.agent.store_transition(state, action if 'action' in locals() else 0,
                                          final_pen, state, True, mask)
                total_reward += final_pen
        # Assign values to each example based on game outcome
        result = self.env.get_result()  # e.g. '1-0', '0-1', '1/2-1/2'
        if result == "1-0":
            winner = 1
        elif result == "0-1":
            winner = -1
        else:
            winner = 0

        # Determine reason for game end
        reason = info['reason']  # e.g. 'checkmate', 'time_flag', 'draw', etc.

        training_data = []
        for (state, pi, player) in training_examples:
            # Value is +1 if player won, -1 if lost, 0 if draw
            value = winner * player
            training_data.append((state, pi, value))

        return training_data, result, reason, current_player , total_reward

    def add_to_buffer(self, data):
        """        Add training data to replay buffer, maintaining maximum size.
        Args:
            data (List[Tuple]): List of training examples to add to buffer.
        
        Adds new training examples to the replay buffer.
        If the buffer exceeds the maximum size, it removes the oldest examples.
        Features:
            - Efficient storage of training data for later use
            - Maintains a fixed-size buffer to prevent memory overflow
            - Automatically manages buffer size by removing oldest entries
        """
        self.replay_buffer.extend(data)
        if len(self.replay_buffer) > self.max_buffer_size:
            excess = len(self.replay_buffer) - self.max_buffer_size
            self.replay_buffer = self.replay_buffer[excess:]

    def train_network(self):
        """Train the neural network on a batch from replay buffer.
        Samples a random batch from the replay buffer and trains the agent's neural network.
        Returns:
            loss (float): Training loss value from the agent's training step.
        Features:
            - Random sampling from replay buffer for training stability
            - Supports prioritized experience replay if configured
        """
        import random
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, pis, values = zip(*batch)

        examples = list(zip(states, pis, values))
        loss = self.agent.train_step(examples, epochs=self.epochs)

        return loss

    def evaluate(self, n_games: int = 30, ep: int = 1, total_episodes: int = 10,
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
        old_eps = getattr(self.agent, "eps", 0.0)  # dummy in case no .eps
        if hasattr(self.agent, "eps"):
            self.agent.eps = 0.0

        eval_strength = force_strength if force_strength is not None else min(0.25,
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
        if hasattr(self.agent, "eps"):
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

    def train(self, training_type: str = "episodes"):
        """
        Main training loop with curriculum learning and evaluation
        Runs the complete training process for the AlphaZero-style agent.
        Features:
            - Progressive opponent strength curriculum
            - Regular evaluation against stronger opponents
            - Best model tracking and checkpointing
            - Comprehensive logging of training progress
        """
        overall_time = self.cfg["training"]["overall_time"]
        if training_type == "episodes":
            episodes = self.cfg["training"]["episodes"]
        elif training_type == "time":
            episodes = 1 * (overall_time / 60)  # Assuming 1 episode per minute
        else:
            raise ValueError(f"Invalid training type: {training_type}. Use 'episodes' or 'time'.")

        warmup = self.cfg["training"]["warmup"]
        log_freq = self.cfg["training"]["log_freq"]
        eval_freq = self.cfg["training"]["eval_freq"]
        save_freq = self.cfg["training"]["save_freq"]

        self.current_training_type = training_type

        if training_type == "episodes":
            for ep in range(1, episodes + 1):
                res = self.play_one(is_training=True, episode_idx=ep, total_episodes=episodes)

                # Training
                loss_val = 0.0
                if ep >= warmup:
                    loss_val = self.train_network()
                    if loss_val:
                        self.losses.append(loss_val)

                # Stats
                total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                wr_all = self.stats["wins"] / max(1, total_games)
                wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))

                # Update training log
                self.training_log[ep] = {
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "outcome": res["outcome"],
                    "result": res["result"],
                    "agent_white": res["agent_white"],
                    "ended_by": res["ended_by"],
                    "loss": loss_val,
                    "win_rate_all": wr_all,
                    "win_rate_true": wr_true
                }
                # Logging
                if ep % log_freq == 0 or ep == 1:
                    opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end, ep, episodes)
                    ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)
                    self.logger.info(
                        f"Ep {ep}/{episodes} | {res['outcome']:4s} | "
                        f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                        f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                        f"L:{(np.mean(self.losses) if self.losses else 0):.4f} | "
                        f"Ww:{self.stats['wins_white']} Wb:{self.stats['wins_black']} "
                        f"Lw:{self.stats['losses_white']} Lb:{self.stats['losses_black']} | "
                        f"opp_str:{opp_strength:.3f} | move_limit_rate:{ml_rate:.3f}"
                    )

                # Evaluation
                if ep % eval_freq == 0:
                    eval_strength = min(0.25, opp_strength + 0.05)
                    wr, wr_true_eval = self.evaluate(self.cfg["training"]["eval_games"], ep, episodes, eval_strength)
                    if wr > self.best_win_rate:
                        self.best_win_rate = wr
                        best_path = os.path.join(self.cfg["paths"]["models"], f"best_alphazero_ep{ep}.pth")
                        self.agent.save(best_path)
                        self.logger.info(f"New best model saved (win_rate={wr:.3f}) -> {best_path}")

                # Checkpoint
                if ep % save_freq == 0:
                    ckpt_path = os.path.join(self.cfg["paths"]["models"], f"ckpt_alphazero_ep{ep}.pth")
                    self.agent.save(ckpt_path)
                    self.logger.info(f"Checkpoint saved -> {ckpt_path}")

        elif training_type == "time":
            end_time = time.time() + overall_time
            ep = 0
            warmup = self.cfg["training"]["warmup"]
            log_freq = self.cfg["training"]["log_freq"]
            eval_freq = self.cfg["training"]["eval_freq"]
            save_freq = self.cfg["training"]["save_freq"]

            while time.time() <= end_time:
                res = self.play_one(is_training=True, episode_idx=ep, total_episodes=0)

                # Training
                loss_val = 0.0
                if len(self.replay_buffer) >= warmup:
                    loss_val = self.train_network()
                    if loss_val:
                        self.losses.append(loss_val)

                # Stats
                total_games = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
                wr_all = self.stats["wins"] / max(1, total_games)
                wr_true = self.stats["wins"] / max(1, (self.stats["wins"] + self.stats["losses"]))

                # Update training log
                self.training_log[ep] = {
                    "reward": res["reward"],
                    "steps": res["steps"],
                    "outcome": res["outcome"],
                    "result": res["result"],
                    "agent_white": res["agent_white"],
                    "ended_by": res["ended_by"],
                    "loss": loss_val,
                    "win_rate_all": wr_all,
                    "win_rate_true": wr_true
                }

                # Logging
                if ep % log_freq == 0 or ep == 0:
                    current_time = time.time() - (end_time - overall_time)
                    opp_strength = utils.linear_anneal(self.opp_strength_start, self.opp_strength_end, ep, episodes)
                    ml_rate = self.ended_by.get("move_limit", 0) / max(1, total_games)
                    self.logger.info(
                        f"Time {current_time:.2f}/{overall_time:.2f} | {res['outcome']:4s} | "
                        f"R:{res['reward']:6.2f} | M:{res['steps']:3d} | "
                        f"WR_all:{wr_all:.3f} | WR_true:{wr_true:.3f} | "
                        f"L:{(np.mean(self.losses) if self.losses else 0):.4f} | "
                        f"Ww:{self.stats['wins_white']} Wb:{self.stats['wins_black']} "
                        f"Lw:{self.stats['losses_white']} Lb:{self.stats['losses_black']} | "
                        f"opp_str:{opp_strength:.3f} | move_limit_rate:{ml_rate:.3f}"
                    )

                # Evaluation
                if ep % eval_freq == 0:
                    eval_strength = min(0.25, opp_strength + 0.05)
                    wr, wr_true_eval = self.evaluate(self.cfg["training"]["eval_games"], ep, overall_time, eval_strength)
                    if wr > self.best_win_rate:
                        self.best_win_rate = wr
                        best_path = os.path.join(self.cfg["paths"]["models"], f"best_alphazero_time{ep}.pth")
                        self.agent.save(best_path)

                # Checkpoint
                if ep % save_freq == 0:
                    ckpt_path = os.path.join(self.cfg["paths"]["models_mcts"], f"_{self.current_training_type}_",f"ckpt_time_ep{ep}.pth")
                    self.agent.save(ckpt_path)

                ep += 1


        self.logger.info("=== FINAL EVALUATION ===")
        num_eval_games = self.cfg["training"]["eval_games"]
        for strength in [0.1, 0.15, 0.2, 0.25, 0.3]:
            wr, wr_true = self.evaluate(num_eval_games, ep, ep, force_strength=strength)
            self.logger.info(f"vs {strength:.2f} strength: WR={wr:.3f} (true={wr_true:.3f})")
        
        
        log_path = os.path.join(
            self.cfg["paths"]["mcts_logs"],
            f"Policy_Value_{self.current_training_type}_{episodes}_mcts_{self.cfg['mcts']['mcts_simulations']}.json"
        )
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)



    def play_one(self, is_training=True, episode_idx=0, total_episodes=0, force_eval_strength=None):
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
        training_data, result, reason, agent_white, reward = self.self_play_game(is_training)


        if is_training:
            self.add_to_buffer(training_data)

        # Compute basic values
        steps = self.env.get_move_count()
        total_reward = 1 if result == "1-0" else -1 if result == "0-1" else 0
        ended_by = reason

        # Finalize and return
        return self.finalize_episode(reward, ended_by, steps, agent_white)


        """
        Map detailed termination reasons to categories
        Args:
            reason (str): Detailed reason for game end (e.g. 'checkmate', 'time_flag', etc.)
        Returns:
            str: Mapped category for the termination reason
        Maps specific game end reasons to broader categories for easier tracking.
        Categories include:
            - mate
            - time_flag
            - draw
            - illegal
            - timeout_or_resignation
        """
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

    def finalize_episode(self, result, reason, steps, agent_white):
        """
        Finalize the episode and update statistics
        Args:
            result (int): Game result from agent's perspective (1 for win, -1 for loss, 0 for draw)
            reason (str): Reason for game end
            steps (int): Number of moves played in the game
            agent_white (bool): Whether the agent played as white
        Returns:
            dict: Summary of the episode with statistics
        
        Finalizes the episode by logging the result, updating statistics,
        and saving the model if necessary.
        Features:
            - Logs the game result and reason
            - Updates win/loss/draw statistics
            - Tracks wins/losses by color
            - Saves model checkpoint at specified intervals
            - Returns a summary dictionary with episode details
        """
        # track general result
        outcome = "draws"
        if result == 1:
            outcome = "wins"
        elif result == -1:
            outcome = "losses"

         # Track by color
        if result == 1:
            if agent_white:
                self.stats["wins_white"] += 1
            else:
                self.stats["wins_black"] += 1
        elif result == -1:
            if agent_white:
                self.stats["losses_white"] += 1
            else:
                self.stats["losses_black"] += 1

        # Count termination reasons
        self.ended_by[reason] += 1
        self.stats[outcome] += 1
        self.stats['games_played'] += 1
        return {
            "reward": result,
            "steps": steps,
            "outcome": outcome,
            "result": "1-0" if result == 1 else "0-1" if result == -1 else "1/2-1/2",
            "agent_white": agent_white,
            "ended_by": reason
        }

    def get_losses(self) -> Dict[int, float]:
        """
        Get training losses recorded during training
        Returns:
            List of losses recorded during training
        """
        return self.losses
    
    def get_ended_by(self) -> Dict[str, int]:
        """
        Get reasons for game endings
        Returns:
            Counter mapping reason to count
        """
        return dict(self.ended_by)
