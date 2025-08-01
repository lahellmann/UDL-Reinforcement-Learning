import os
import chess
<<<<<<< Updated upstream
import threading
import time
import torch
import utils
from environment import BulletChessEnv
from ddqn_agent import load_ddqn_from_path , DDQNAgent
from policyvalue_agent import load_mcts_from_path , MCTSAgent

=======
import asyncio
import ipywidgets as widgets
from IPython.display import display
import chess.svg

from ddqn_agent import DDQNAgent
from policyvalue_agent import MCTSAgent
from environment import BulletChessEnv
from debug import load_agent_from_file
import utils
>>>>>>> Stashed changes

class JupyterChessApp:
    def __init__(self, model_path, user_color=chess.WHITE):
        self.env = BulletChessEnv()
        self.agent = load_agent_from_file(model_path, cfg=utils.get_truly_fixed_cfg(), env=self.env)
        self.user_color = user_color
        self.moves_history = []
        self.is_agent_turn = (self.env.state.board.turn != self.user_color)
        self.selected_square = None

        self.board_svg = widgets.HTML()
        self.info = widgets.Textarea(layout=widgets.Layout(width='400px', height='150px'))
        self.restart_btn = widgets.Button(description="Restart")
        self.restart_btn.on_click(self.restart_game)
        self.replay_btn = widgets.Button(description="Replay")
        self.replay_btn.on_click(self.replay_game)

        self.move_input = widgets.Combobox(
            placeholder='Type your move (e.g. e2e4 or Nf3)',
            description='Move:',
            layout=widgets.Layout(width='300px')
        )
        self.move_input.observe(self.on_move_input_submit, names='value')

        self.ui = widgets.VBox([
            self.board_svg,
            self.info,
            widgets.HBox([self.move_input, self.restart_btn, self.replay_btn])
        ])

        self.restart_game(None)

    def update_display(self):
        board = self.env.state.board
        svg = chess.svg.board(board=board, size=400)
        self.board_svg.value = svg

        last_move = board.peek().uci() if board.move_stack else "None"
        turn_text = "Your turn." if not self.is_agent_turn else "Waiting for agent..."
        self.info.value = (
            f"Move count: {self.env.get_move_count()}\n"
            f"White time: {self.env.state.white_time:.1f} sec\n"
            f"Black time: {self.env.state.black_time:.1f} sec\n"
            f"Last move: {last_move}\n"
            f"{turn_text}"
        )

        if not self.is_agent_turn:
            self.update_move_suggestions()

    def update_move_suggestions(self):
        legal_moves = list(self.env.state.board.legal_moves)
        suggestions = [self.env.state.board.san(m) for m in legal_moves]
        self.move_input.options = suggestions

<<<<<<< Updated upstream
restart_button.on_click(on_restart_clicked)
cancel_button.on_click(on_cancel_clicked)

board = env.state.board


def square_color(square):
    """Return True if square is light-colored, False otherwise."""
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return (rank + file) % 2 == 0

def build_board():
    """Create and display the chessboard buttons."""
    global buttons, container
    buttons = {}
    rows = []
    for rank in reversed(range(1, 9)):
        row = []
        for file in "abcdefgh":
            sq = file + str(rank)
            btn = widgets.Button(layout=widgets.Layout(width='45px', height='45px'))
            btn.value = sq
            btn.on_click(on_click)

            # Set button background color according to square color
            if square_color(chess.parse_square(sq)):
                btn.style.button_color = '#f0d9b5'  # light
            else:
                btn.style.button_color = '#b58863'  # dark

            buttons[sq] = btn
            row.append(btn)
        rows.append(widgets.HBox(row))
    container = widgets.VBox(rows)
    display(container)
    display(timer_label)

def update_buttons():
    """Update button labels and styles to reflect the current board state."""
    for sq, btn in buttons.items():
        btn.description = ''
        btn.icon = ''
        piece = board.piece_at(chess.parse_square(sq))
        if piece:
            btn.description = unicode_pieces[piece.symbol()]
            btn.style.font_weight = 'bold'
            btn.style.font_size = '28px'
        else:
            btn.style.font_weight = 'normal'
            btn.style.font_size = '14px'

        if square_color(chess.parse_square(sq)):
            btn.style.button_color = '#f0d9b5'
        else:
            btn.style.button_color = '#b58863'
        btn.button_style = ''

    # Refresh container children to force UI redraw
    container.children = tuple(
        widgets.HBox([buttons[file + str(rank)] for file in "abcdefgh"])
        for rank in reversed(range(1, 9))
    )

def highlight_legal_moves(from_sq):
    """Highlight legal moves for the selected square."""
    clear_highlights()
    from_square = chess.parse_square(from_sq)
    for move in board.legal_moves:
        if move.from_square == from_square:
            to_sq = chess.square_name(move.to_square)
            buttons[to_sq].button_style = 'success'  # green highlight
    buttons[from_sq].button_style = 'info'  # blue for selected

def clear_highlights():
    """Clear all highlights on the board."""
    for sq, btn in buttons.items():
        btn.button_style = ''
        if square_color(chess.parse_square(sq)):
            btn.style.button_color = '#f0d9b5'
        else:
            btn.style.button_color = '#b58863'

def disable_all_buttons():
    """Disable all board buttons to prevent further input."""
    for btn in buttons.values():
        btn.disabled = True

def enable_all_buttons():
    """Enable all board buttons."""
    for btn in buttons.values():
        btn.disabled = False

game_started = False  # Global flag to track if game started

def start_game():
    global game_started
    if not game_started:
        game_started = True
        start_timer()
        print("Game started!")

def on_click(b):
    global selected_square, board, env, time_up, game_started

    if not game_started:
        print("Press 'Start Game' first!")
        return

    if time_up:
        print("Game is over. No more moves allowed.")
        return

    print("on_click started, button:", b.value)

    sq = b.value
    piece = board.piece_at(chess.parse_square(sq))

    current_player = "white" if env.state.board.turn else "black"
    current_color = current_player  # e.g., "white" or "black"

    # Check if it's player's turn
    if (current_color == "white" and selected_white != "Player") or (current_color == "black" and selected_black != "Player"):
        print("Wait for the agent to move.")
        return

    if selected_square is None:
        # First click: select player's piece
        if piece and ((current_color == "white" and piece.color == chess.WHITE) or (current_color == "black" and piece.color == chess.BLACK)):
            selected_square = sq
            highlight_legal_moves(sq)
        else:
            print(f"Select a {current_color} piece.")
    else:
        # Second click: attempt move
        if sq == selected_square:
            selected_square = None
            clear_highlights()
            update_buttons()
=======
    def on_move_input_submit(self, change):
        if self.is_agent_turn:
            self.info.value += "\nPlease wait for the agent."
>>>>>>> Stashed changes
            return

        move_str = change['new']
        board = self.env.state.board
        try:
            move = board.parse_san(move_str)
        except:
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    raise ValueError
            except:
                self.info.value += "\nInvalid move."
                return

        self.make_user_move(move.uci())
        self.move_input.value = ''

    def make_user_move(self, uci_move):
        obs, reward, done, info = self.env.step(uci_move)
        self.moves_history.append(uci_move)
        self.update_display()
        if done:
            self.handle_game_end(info)
            return
        self.is_agent_turn = True
        asyncio.create_task(self.agent_move_async())

    async def agent_move_async(self):
        await asyncio.sleep(1)
        if self.env.is_game_over():
            self.handle_game_end({})
            return

        obs = self.env.get_observation()
        legal_actions = self.env.get_legal_actions()

        if isinstance(self.agent, MCTSAgent):
            pi = self.agent.run_mcts(self.env)
            action_index = self.agent.select_action_from_pi(pi, temperature=1.0)
        else:
            action_index = self.agent.select_action(obs, legal_actions)

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
        msg = f"{reason}. "
        if winner:
            msg += f"Winner: {winner}"
        else:
            msg += "Draw or unknown outcome."
        self.info.value += "\n" + msg

    def restart_game(self, _):
        self.env.reset()
        self.moves_history.clear()
        self.is_agent_turn = (self.env.state.board.turn != self.user_color)
        self.update_display()
        if self.is_agent_turn:
            asyncio.create_task(self.agent_move_async())

    def replay_game(self, _):
        async def replay():
            self.env.reset()
            self.update_display()
            for move in self.moves_history:
                await asyncio.sleep(1)
                self.env.step(move)
                self.update_display()
        asyncio.create_task(replay())


class JupyterChessLauncher:
    def __init__(self):
        self.model_dropdown = widgets.Dropdown(
            options=[f for f in os.listdir("models") if f.endswith(".pth")],
            description="Model:",
            layout=widgets.Layout(width='300px')
        )
        self.color_dropdown = widgets.Dropdown(
            options=[("White", chess.WHITE), ("Black", chess.BLACK)],
            description="Your Color:",
            layout=widgets.Layout(width='300px')
        )
        self.start_btn = widgets.Button(description="Start Game", button_style="success")
        self.start_btn.on_click(self.start_game)

        self.ui = widgets.VBox([
            self.model_dropdown,
            self.color_dropdown,
            self.start_btn
        ])
        self.app = None

<<<<<<< Updated upstream


def print_game_over(info):
    """Print a message when the game ends."""
    print("\n--- Game over ---")
    reason = info.get("reason", "unknown")
    if reason == "time":
        print("Time out!")
    elif reason == "checkmate":
        print("Checkmate!")
    else:
        print("Draw or unknown.")

def update_timer_label():
    """Update the timer display."""
    white_time = env.state.white_time
    black_time = env.state.black_time
    timer_label.value = f"White time: {white_time:.1f}s | Black time: {black_time:.1f}s"

def stop_timer():
    """Stop the timer updater thread."""
    global timer_running
    timer_running = False

def start_timer():
    """Start timer thread if not running."""
    global timer_running, time_up, timer_thread
    if timer_thread and timer_thread.is_alive():
        return
    timer_running = True
    time_up = False
    timer_thread = threading.Thread(target=timer_updater, daemon=True)
    timer_thread.start()

def timer_updater():
    """Background thread to update timer and check for timeouts."""
    global timer_running, time_up
    while timer_running and not time_up:
        update_timer_label()
        white_time = env.state.white_time
        black_time = env.state.black_time

        if white_time <= 0:
            print("Time is up! White loses.")
            time_up = True
            disable_all_buttons()
            stop_timer()
            break
        if black_time <= 0:
            print("Time is up! Black loses.")
            time_up = True
            disable_all_buttons()
            stop_timer()
            break

        time.sleep(1)



import ipywidgets as widgets
from IPython.display import display, clear_output
import os

def select_agents(models_dir="models"):
    import utils
    import os
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    global env, board, selected_white, selected_black, white_agent, black_agent, restart_button, cancel_button

    models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    options = ["Player"] + models

    white_dropdown = widgets.Dropdown(options=options, description="White:")
    black_dropdown = widgets.Dropdown(options=options, description="Black:")
    start_button = widgets.Button(description="Start Game")
    output = widgets.Output()

    def on_start_clicked(b):
        global selected_white, selected_black, white_agent, black_agent, env, board, game_started
        with output:
            clear_output()
            print(f"White: {white_dropdown.value}, Black: {black_dropdown.value}")
            selected_white = white_dropdown.value
            selected_black = black_dropdown.value

            cfg = utils.get_truly_fixed_cfg()

            
            if selected_white in ["Player", "None", None]:
                white_agent = None
            else:
                if selected_white.startswith("ckpt_ddqn") and selected_white.endswith(".pth"):
                    white_agent = load_ddqn_from_path(f"{models_dir}/{selected_white}", cfg, env)
                else:
                    white_agent = load_mcts_from_path(f"{models_dir}/{selected_white}", cfg, env)
                #white_agent = load_agent_from_path(f"{models_dir}/{selected_white}", cfg, env)

            if selected_black in ["Player", "None", None]:
                black_agent = None
            else:
                if selected_black.startswith("ckpt_ddqn") and selected_black.endswith(".pth"):
                    black_agent = load_ddqn_from_path(f"{models_dir}/{selected_black}", cfg, env)
                else:
                    black_agent = load_mcts_from_path(f"{models_dir}/{selected_black}", cfg, env)

            env.reset()
            start_game()

            board = env.state.board
            build_board()
            update_buttons()
            start_timer()

            if env.state.board.turn == True and white_agent is not None:
                agent_turn()
            elif env.state.board.turn == False and black_agent is not None:
                agent_turn()

    start_button.on_click(on_start_clicked)

    display(widgets.VBox([
        white_dropdown,
        black_dropdown,
        start_button,
        widgets.HBox([restart_button, cancel_button]),
        output
    ]))



=======
    def start_game(self, _):
        model_path = os.path.join("models", self.model_dropdown.value)
        user_color = self.color_dropdown.value
        self.app = JupyterChessApp(model_path, user_color)
        display(self.app.ui)
>>>>>>> Stashed changes
