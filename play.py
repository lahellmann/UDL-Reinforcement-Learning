import ipywidgets as widgets
from IPython.display import display, clear_output
import chess
import threading
import time

from environment import BulletChessEnv
timer_thread = None
timer_running = True
time_up = False
env = BulletChessEnv()
agent = None
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
    display(widgets.HBox([restart_button, cancel_button]))

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

def on_click(b):
    """Handle user clicks on board buttons for moving pieces."""
    global selected_square, board, env, time_up

    if time_up:
        print("Game is over. No more moves allowed.")
        return

    sq = b.value
    piece = board.piece_at(chess.parse_square(sq))

    if selected_square is None:
        # First click: select white piece
        if piece and piece.color == chess.WHITE:
            selected_square = sq
            highlight_legal_moves(sq)
        else:
            print("Select a white piece.")
    else:
        # Second click: attempt move
        if sq == selected_square:
            # Deselect piece
            selected_square = None
            clear_highlights()
            update_buttons()
            return

        from_square = chess.parse_square(selected_square)
        to_square = chess.parse_square(sq)

        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            # try promotions
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
    """Let the agent perform its move."""
    global board, env, agent

    obs = env.get_observation()
    legal_actions = env.get_legal_actions()
    action = agent.select_action(obs, legal_actions)
    uci_move = env._action_to_move(action).uci()
    obs, reward, done, info = env.step(uci_move)
    print(f"Agent moves: {uci_move}")

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


def on_restart_clicked(b):
    global env, board, selected_square, timer_running, time_up
    timer_running = False
    time_up = False
    env.reset()
    board = env.state.board
    selected_square = None
    enable_all_buttons()
    update_buttons()
    update_timer_label()
    start_timer()
    print("Game restarted.")

def on_cancel_clicked(b):
    global timer_running
    timer_running = False
    disable_all_buttons()
    print("Game canceled.")

