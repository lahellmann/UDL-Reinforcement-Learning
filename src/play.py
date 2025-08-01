import os
import chess
import asyncio
import ipywidgets as widgets
from IPython.display import display
import chess.svg

from src.policyvalue_agent import MCTSAgent
from src.environment import BulletChessEnv
import src.utils as utils

class JupyterChessApp:
    """
    A Jupyter Notebook application for playing chess against an AI agent.
    This application allows users to play chess against a trained AI agent, with
    options to restart the game or replay the moves made so far. The user can
    select their color and input moves in standard algebraic notation or UCI format.
    The application uses ipywidgets for the user interface and chess.svg for rendering
    the chessboard.
    """
    def __init__(self, model_path, user_color=chess.WHITE):
        """
        Initialize the chess application.
        Args:
            model_path (str): Path to the trained agent model file.
            user_color (chess.Color): Color of the user (white or black).
        """
        self.env = BulletChessEnv()
        self.agent = utils.load_agent_from_file(model_path, cfg=utils.get_truly_fixed_cfg(), env=self.env)
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
        """Update the chessboard and game information display.
        This method renders the current state of the chessboard and updates
        the game information such as move count, time remaining for each player,
        last move made, and whose turn it is.
        """
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
        """
        Update the move suggestions based on the current board state.
        This method retrieves the legal moves from the current board state and
        updates the move input options to suggest valid moves to the user.
        """
        legal_moves = list(self.env.state.board.legal_moves)
        suggestions = [self.env.state.board.san(m) for m in legal_moves]
        self.move_input.options = suggestions

    def on_move_input_submit(self, change):
        """ Handle the move input submission.
        This method is called when the user submits a move through the input field.
        It validates the move, updates the game state, and triggers the agent's turn if necessary.
        Args:
            change (dict): The change event containing the new value of the move input.
        """
        if self.is_agent_turn:
            self.info.value += "\nPlease wait for the agent."
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
        """Make a move as the user.
        This method applies the user's move to the game environment, updates the moves history,
        and refreshes the display.
        Args:
            uci_move (str): The move in UCI format (e.g., 'e2e4').
        """
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
        """Handle the end of the game.
        This method updates the game information display with the reason for the game ending,
        the winner if applicable, and any additional information provided.
        Args:
            info (dict): Information about the game ending, including reason and winner.
        """
        reason = info.get("reason", "Game over")
        winner = info.get("winner", None)
        msg = f"{reason}. "
        if winner:
            msg += f"Winner: {winner}"
        else:
            msg += "Draw or unknown outcome."
        self.info.value += "\n" + msg

    def restart_game(self, _):
        """Restart the game.
        This method resets the game environment, clears the moves history, and updates the display.
        """
        self.env.reset()
        self.moves_history.clear()
        self.is_agent_turn = (self.env.state.board.turn != self.user_color)
        self.update_display()
        if self.is_agent_turn:
            asyncio.create_task(self.agent_move_async())

    def replay_game(self, _):
        """Replay the game.
        This method replays the moves made so far in the game, updating the display after each
        move with a delay to simulate the game being played out.
        """
        async def replay():
            self.env.reset()
            self.update_display()
            for move in self.moves_history:
                await asyncio.sleep(1)
                self.env.step(move)
                self.update_display()
        asyncio.create_task(replay())


class JupyterChessLauncher:
    """A launcher for the JupyterChess application.
    This class provides a user interface for selecting a model and color,
    and starting a game of chess against the AI agent.
    """
    def __init__(self):
        """ Initialize the launcher.
        This method sets up the user interface components such as dropdowns for model selection
        and color selection, as well as the start game button.
        """
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

    def start_game(self, _):
        """Start the chess game.
        This method is called when the user clicks the start game button.
        """
        model_path = os.path.join("models", self.model_dropdown.value)
        user_color = self.color_dropdown.value
        self.app = JupyterChessApp(model_path, user_color)
        display(self.app.ui)
