import os
import time
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
import chess
import numpy as np
import utils

from ddqn_agent import DDQNAgent
from policyvalue_agent import MCTSAgent
from environment import BulletChessEnv  
from policyvalue_agent import load_agent_from_path as load_policyvalue_agent
from ddqn_agent import load_agent_from_path as load_ddqn_agent

# -------- Helper to load models --------
def load_agent_from_file(filepath, cfg, env):
    filename = os.path.basename(filepath).lower()
    if "ddqn" in filename:
        return load_ddqn_agent(filepath, cfg, env)
    else:
        return load_policyvalue_agent(filepath, cfg, env)

# -------- UI --------
class ChessApp:
    def __init__(self, root):
        self.root = root
        self.env = BulletChessEnv()
        self.agent = None
        self.user_color = chess.WHITE  # Default
        self.moves_history = []
        self.is_agent_turn = False

        # Display
        self.text = tk.Text(root, height=20, width=60)
        self.text.pack()

        self.input_box = tk.Entry(root)
        self.input_box.pack()
        self.input_box.bind("<Return>", self.on_user_submit)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Restart", command=self.restart_game).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Replay", command=self.replay_game).pack(side=tk.LEFT)

        # Start
        self.select_model_and_start()

    def select_model_and_start(self):
        models = [f for f in os.listdir("models") if f.endswith(".pth")]
        if not models:
            messagebox.showerror("Error", "No models found in models/")
            self.root.quit()
            return
        model = simpledialog.askstring("Model selection", f"Available models:\n{models}\nType filename to load:")
        if not model or model not in models:
            messagebox.showerror("Error", "Invalid model selected")
            self.root.quit()
            return
        path = os.path.join("models", model)
        self.agent = load_agent_from_file(path, cfg=utils.get_truly_fixed_cfg(), env=self.env)

        color = simpledialog.askstring("Choose Color", "Play as white or black? (w/b):")
        if color and color.lower() == "b":
            self.user_color = chess.BLACK
        else:
            self.user_color = chess.WHITE

        self.restart_game()

    def restart_game(self):
        self.env.reset()
        self.moves_history.clear()
        self.is_agent_turn = (self.env.state.board.turn != self.user_color)
        self.update_display()
        if self.is_agent_turn:
            self.root.after(1000, self.agent_move)

    def update_display(self):
        board = self.env.state.board
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, str(board) + "\n")
        self.text.insert(tk.END, f"Move count: {self.env.get_move_count()}\n")
        self.text.insert(tk.END, f"White time: {self.env.state.white_time:.1f} sec\n")
        self.text.insert(tk.END, f"Black time: {self.env.state.black_time:.1f} sec\n")
        self.text.insert(tk.END, f"Last move: {board.peek().uci() if board.move_stack else 'None'}\n")

    def on_user_submit(self, event=None):
        move = self.input_box.get().strip()
        if not self.is_agent_turn and self.env.state.board.turn == self.user_color:
            self.user_move(move)
        self.input_box.delete(0, tk.END)

    def user_move(self, uci_move):
        if self.env.is_game_over():
            messagebox.showinfo("Game Over", "Game is already over.")
            return
        try:
            move = chess.Move.from_uci(uci_move)
        except:
            messagebox.showerror("Invalid move", "Move not in UCI format or invalid.")
            return
        if move not in self.env.state.board.legal_moves:
            messagebox.showerror("Illegal move", "Move is illegal.")
            return
        obs, reward, done, info = self.env.step(uci_move)
        self.moves_history.append(uci_move)
        self.update_display()
        if done:
            self.handle_game_end(info)
        else:
            self.is_agent_turn = True
            self.root.after(1000, self.agent_move)

    def agent_move(self):
        if self.env.is_game_over():
            self.handle_game_end({})
            return

        obs = self.env.get_observation()
        legal_actions = self.env.get_legal_actions()

        # Nur für MCTS-Agenten:
        if isinstance(self.agent, MCTSAgent):
            pi = self.agent.run_mcts(self.env)
            action_index = self.agent.select_action_from_pi(pi, temperature=1.0)
        else:
            action_index = self.agent.act(obs, legal_actions)

        move = self.env._action_to_move(action_index)
        uci_move = move.uci()

        obs, reward, done, info = self.env.step(uci_move)
        self.moves_history.append(uci_move)
        self.update_display()

        if done:
            self.handle_game_end(info)
            return

        self.is_agent_turn = False



    def handle_game_end(self, info):
        reason = info.get("reason", "Game over")
        winner = info.get("winner", None)
        if winner:
            messagebox.showinfo("Game Over", f"{reason}. Winner: {winner}")
        else:
            messagebox.showinfo("Game Over", f"{reason}. Draw or unknown outcome.")

    def replay_game(self):
        self.env.reset()
        self.update_display()
        self.root.after(500, lambda: self._play_moves(0))

    def _play_moves(self, idx):
        if idx >= len(self.moves_history):
            return
        move = self.moves_history[idx]
        self.env.step(move)
        self.update_display()
        self.root.after(1000, lambda: self._play_moves(idx + 1))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bullet Chess AI")
    app = ChessApp(root)
    root.mainloop()
