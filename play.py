import ipywidgets as widgets
from IPython.display import display, clear_output
import chess
import threading
import time
import utils
from environment import BulletChessEnv
from ddqn_agent import load_agent_from_path



timer_thread = None
timer_running = True
time_up = False
env = BulletChessEnv()
agent = None

white_agent = None
black_agent = None
selected_white = None
selected_black = None


selected_square = None
buttons = {}
container = None  # Container holding the chessboard buttons
timer_label = widgets.Label()
unicode_pieces = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}
# --- Buttons for restart and cancel ---
restart_button = widgets.Button(description="Restart")
cancel_button = widgets.Button(description="Cancel")

def on_restart_clicked(b):
    global env, board, selected_square, timer_running, time_up, game_started
    timer_running = False
    time_up = False
    game_started = True  # Make sure game is marked as started
    env.reset()
    board = env.state.board
    selected_square = None
    enable_all_buttons()
    update_buttons()
    update_timer_label()
    start_timer()
    print("Game restarted.")

    # Let agent move if it's their turn
    if env.state.board.turn and white_agent is not None:
        agent_turn()
    elif not env.state.board.turn and black_agent is not None:
        agent_turn()

def on_cancel_clicked(b):
    global timer_running, game_started
    timer_running = False
    game_started = False  # Ensure game is marked as ended
    disable_all_buttons()
    print("Game canceled.")

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
            return

        from_square = chess.parse_square(selected_square)
        to_square = chess.parse_square(sq)

        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_square, to_square, promotion=promo_piece)
                if promo_move in board.legal_moves:
                    move = promo_move
                    break
            else:
                print("Illegal move. Try again.")
                return

        obs, reward, done, info = env.step(move.uci())
        board = env.state.board
        selected_square = None
        clear_highlights()
        update_buttons()
        update_timer_label()

        if done:
            print_game_over(info)
            disable_all_buttons()
            stop_timer()
            return

        agent_turn()

def agent_turn():
    global board, env, white_agent, black_agent, game_started, selected_white, selected_black
    print("agent turn is called")

    if not game_started:
        print("Game not started, returning")
        return

    current_player = "white" if env.state.board.turn else "black"
    print("Current player:", current_player)
    print("Selected white:", selected_white)
    print("Selected black:", selected_black)

    # Check if it’s human turn, skip agent move if so
    if (current_player == "white" and selected_white == "Player") or (current_player == "black" and selected_black == "Player"):
        print("Human turn, no agent move")
        return

    agent = white_agent if current_player == "white" else black_agent
    print("Agent:", agent)

    if agent is None:
        print("No agent assigned, returning")
        return

    obs = env.get_observation()
    legal_actions = env.get_legal_actions()

    print("Obs type:", type(obs), "Shape:", getattr(obs, 'shape', None))
    print("Legal actions:", legal_actions)

    action = agent.select_action(obs, legal_actions)
    uci_move = env._action_to_move(action).uci()
    print(f"Agent moves: {uci_move}")

    obs, reward, done, info = env.step(uci_move)
    board = env.state.board
    update_buttons()
    update_timer_label()

    if done:
        print_game_over(info)
        disable_all_buttons()
        stop_timer()




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

def select_agents(models_dir="models_ddqn_truly_fixed"):
    import utils
    from ddqn_agent import load_agent_from_path
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
                white_agent = load_agent_from_path(f"{models_dir}/{selected_white}", cfg)

            if selected_black in ["Player", "None", None]:
                black_agent = None
            else:
                black_agent = load_agent_from_path(f"{models_dir}/{selected_black}", cfg)

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
