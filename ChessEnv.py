"""
Prequisites:
    Install stockfish engine from https://stockfishchess.org/
"""
import chess
import random
from typing import Tuple, List, Optional, Dict, Any
from stockfish import Stockfish

class ChessEnv:
    """
    A chess environment for reinforcement learning algorithms like Monte Carlo
    that uses python-chess as the underlying chess engine and python-stockfish for position evaluation.
    """

    def __init__(self, stockfish_path: str, stockfish_params: Dict = None, max_steps: int = 100, stockfish_depth: int = 10):
        """
        Initialize the chess environment with Stockfish integration using the Python package.

        Args:
            stockfish_path: Path to the Stockfish executable.
            stockfish_params: Dictionary of parameters for the Stockfish engine.
                If None, default parameters will be used.
            max_steps: Maximum number of steps before the game is considered a draw.
            stockfish_depth: Depth for Stockfish evaluation.
        """
        self.board = chess.Board()
        self.max_steps = max_steps
        self.stockfish_depth = stockfish_depth
        self.steps = 0
        self.done = False
        self.result = None

        # Default Stockfish parameters
        default_params = {
            "Threads": 1,
            "Hash": 16,
        }

        # Use provided parameters or defaults
        if stockfish_params is None:
            stockfish_params = default_params
        else:
            # Merge with defaults, preferring provided values
            for key, value in default_params.items():
                if key not in stockfish_params:
                    stockfish_params[key] = value

        # Initialize Stockfish engine using the python package
        self.stockfish = Stockfish(path=stockfish_path, parameters=stockfish_params)
        self.stockfish.set_depth(self.stockfish_depth)

    def __del__(self):
        """Destructor."""
        self.close()

    def close(self):
        """Explicitly close the engine."""
        if hasattr(self, 'stockfish') and self.stockfish is not None:
          del self.stockfish
          self.stockfish = None

    def reset(self) -> chess.Board:
        """
        Reset the environment to the starting position.

        Returns:
            The initial state (chess board).
        """
        self.board = chess.Board()
        self.stockfish.set_fen_position(self.board.fen()) # corrected to use fen.
        self.steps = 0
        self.done = False
        self.result = None
        return self.board

    def evaluate_position(self) -> float:
        """
        Evaluate the current position using Stockfish.

        Returns:
            Numerical evaluation from white's perspective, in pawns (1.0 = 1 pawn advantage)
        """
        # Skip evaluation if game is over
        if self.board.is_game_over():
            outcome = self.board.outcome()
            if outcome.winner == chess.WHITE:
                return 10.0  # White wins
            elif outcome.winner == chess.BLACK:
                return -10.0  # Black wins
            else:
                return 0.0  # Draw

        # Update Stockfish with the current position using fen (official state string for chess)
        self.stockfish.set_fen_position(self.board.fen())

        # Get evaluation from stockfish
        evaluation = self.stockfish.get_evaluation()

        # Parse the evaluation
        if evaluation["type"] == "cp":
            # Centipawn evaluation (convert to pawns)
            return evaluation["value"] / 100.0
        else:
            # Mate evaluation
            mate_in = evaluation["value"]
            if mate_in == 0:
                # Handle the edge case where mate_in is 0
                # This could indicate an immediate mate (checkmate on the board)
                # or potentially an error condition
                return 10.0 if self.board.turn == chess.BLACK else -10.0  # Immediate mate for the side that just moved
            elif mate_in > 0:
                return 9.0 + (1.0 / mate_in)  # Positive for white winning
            else:
                return -9.0 - (1.0 / mate_in)  # Negative for black winning

    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by making a move.

        Args:
            action: A chess move.

        Returns:
            Tuple containing:
            - next_state: The new board state after the move
            - reward: The reward for the action based on Stockfish evaluation (-10 to 10 scale)
            - done: Whether the game is finished
            - info: Additional information
        """
        if action not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {action}")

        # Get evaluation before the move
        if self.steps > 0:  # Skip on first move
            eval_before = self.evaluate_position()
        else:
            eval_before = 0.0

        # Make the move
        self.board.push(action)
        self.steps += 1

        # Check if the game is over
        if self.board.is_game_over():
            self.done = True
            self.result = self.board.outcome()

            # Determine reward based on game result
            if self.result.winner == chess.WHITE:
                reward = 1.0  # White wins
            elif self.result.winner == chess.BLACK:
                reward = -1.0  # Black wins
            else:
                reward = 0.0  # Draw
        else:
            # Game continues
            self.done = False

            # Get evaluation after the move
            eval_after = self.evaluate_position()

            # Calculate reward based on position improvement/deterioration
            # Perspective: positive reward if white's position improves or black's position deteriorates
            player_perspective = 1 if self.board.turn == chess.BLACK else -1
            reward = player_perspective * (eval_after - eval_before)

            # Terminate when max steps are reached
            if self.steps >= self.max_steps:
                self.done = True
                reward = 0.0 

        info = {
            "steps": self.steps,
            "result": self.result,
            "legal_moves": list(self.board.legal_moves),
            "evaluation": self.evaluate_position()
        }

        return self.board, reward, self.done, info

    def get_legal_actions(self) -> List[chess.Move]:
        """
        Get all legal moves from the current position.

        Returns:
            List of legal moves.
        """
        return list(self.board.legal_moves)

    def get_random_action(self) -> Optional[chess.Move]:
        """
        Get a random legal move from the current position.

        Returns:
            A random legal move, or None if there are no legal moves.
        """
        legal_moves = self.get_legal_actions()
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def render(self) -> str:
        """
        Render the board as a string.

        Returns:
            String representation of the board.
        """
        return str(self.board)


# Example usage
stockfish_path = "/Users/kaust/stockfish/stockfish-windows-x86-64-avx2.exe" # Change this to your path to stockfish
env = ChessEnv(stockfish_path)
state = env.reset()
print(state)
print("")

# Example of a step
action = state.parse_san('d4')
if action:
    next_state, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    print(next_state)
else:
    print("No legal moves available.")
env.close()