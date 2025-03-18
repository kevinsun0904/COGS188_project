import math
import chess
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import time
import tensorflow as tf
from ChessEnv import ChessEnv
import random

# Load CNN model
model_path = 'chess_cnn_model.h5'
cnn_model = None

try:
    cnn_model = tf.keras.models.load_model(model_path, compile=False)
    print("CNN model loaded successfully (without compilation).")
    
    # Recompile with a supported optimizer
    from tensorflow.keras.optimizers import Adam
    cnn_model.compile(
        optimizer=Adam(learning_rate=0.001),  
        loss="mse",
        metrics=["mae"]
    )
except Exception as e:
    print(f"Error loading or compiling CNN model: {e}")
    print("Will use SimpleChessEvaluator as fallback.")

# Add the split_dims function
squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
    # Create a 3D matrix
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    # Add pieces view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # Add attacks and valid moves
    aux = board.turn
    board.turn = chess.WHITE

    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1

    board.turn = chess.BLACK

    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1

    board.turn = aux

    return board3d


class ChessCNN:
    """CNN-based position evaluator"""
    def __init__(self, model_path=None):
        # Initialize with either a path or use global model
        if model_path and isinstance(model_path, str):
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                # Recompile with a supported optimizer
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse",
                    metrics=["mae"]
                )
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                self.model = None
        else:
            # Use global model if available
            self.model = cnn_model
            print("Using global CNN model")
    
    def predict(self, board):
        """Evaluate chess position using CNN model"""
        if self.model is None:
            print("Model not loaded correctly")
            return 0.0
            
        try:
            # Convert board to CNN input format if it's a chess.Board object
            if isinstance(board, chess.Board):
                board_tensor = split_dims(board)
            else:
                board_tensor = board
            
            # Make sure board tensor has correct shape for model input
            if len(board_tensor.shape) == 3:  # If it's missing batch dimension
                board_tensor = np.expand_dims(board_tensor, axis=0)
                
            # Convert to float32 for model input
            board_tensor = board_tensor.astype(np.float32)
            
            # Get raw evaluation from model with verbose=0 to suppress progress bar
            raw_eval = self.model.predict(board_tensor, verbose=0)[0][0]
            
            # Scale to [-10, 10] range based on tanh activation in output layer
            scaled_eval = raw_eval * 10.0
            
            # Return the evaluation with clamping for safety
            return max(-10.0, min(10.0, scaled_eval))
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0.0

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Convert chess board to tensor representation for CNN input.
    
    Args:
        board: Chess board to encode
        
    Returns:
        Tensor representation of the board
    """
    board3d = split_dims(board)

    return board3d


class MCTSNode:
    """Node in the MCTS tree."""
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = {}
        self.visits = 0
        self.results = {chess.WHITE: 0, chess.BLACK: 0, 'draw': 0}
        self.untried_moves = list(board.legal_moves)
        
        # Policy prior probabilities from CNN
        self.prior_probs = {}
        
    def is_fully_expanded(self) -> bool:
        """Check if all possible child nodes have been expanded."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state (game over)."""
        return self.board.is_game_over()

    def get_result(self, side_to_move: chess.Color) -> float:
        """Get the game result from the perspective of the given side."""
        if not self.board.is_game_over():
            return 0.0
        
        outcome = self.board.outcome()
        if outcome.winner == side_to_move:
            return 1.0
        elif outcome.winner is None:  # Draw
            return 0.5
        else:
            return 0.0
    
    def get_ucb_score(self, parent_visits: int, exploration_weight: float = 1.41) -> float:
        """
        Calculate the UCB score for this node.
        
        Args:
            parent_visits: Number of visits to the parent node
            exploration_weight: Controls exploration vs exploitation
            
        Returns:
            UCB score value
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unexplored nodes

        if self.parent and self.parent.board.turn == chess.WHITE:
            win_ratio = self.results[chess.WHITE] / self.visits
        else:
            win_ratio = self.results[chess.BLACK] / self.visits
        
        # Exploration term
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        
        return win_ratio + exploration
    
    def select_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """Select the child with the highest UCB score."""
        if not self.children:
            raise ValueError("Cannot select child from a node with no children")
        
        # Find the child with the highest UCB score
        return max(self.children.values(), 
                   key=lambda child: child.get_ucb_score(self.visits, exploration_weight))
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a child node.
        
        Returns:
            The new child node or None if no untried moves.
        """
        if not self.untried_moves:
            return None
        
        # Choose a random untried move
        move = self.untried_moves.pop()
        
        # Create a new board state
        new_board = self.board.copy()
        new_board.push(move)
        
        # Create and link the new child node
        child = MCTSNode(new_board, parent=self, move=move)
        self.children[move] = child
        
        return child
    
    def update(self, result: Dict[chess.Color, float]) -> None:
        """
        Update node statistics with the result of a simulation.
        
        Args:
            result: Dictionary with results for each side
        """
        self.visits += 1
        self.results[chess.WHITE] += result[chess.WHITE]
        self.results[chess.BLACK] += result[chess.BLACK]
        self.results['draw'] += result.get('draw', 0)
    
    def __str__(self) -> str:
        """String representation of the node."""
        move_str = str(self.move) if self.move else "root"
        return f"Node(move={move_str}, visits={self.visits}, results={self.results})"

class ChessMCTS:
    """Monte Carlo Tree Search with CNN policy network."""
    def __init__(self, env: ChessEnv = None, evaluator = None, exploration_weight=1.41, 
                 simulation_limit=100, time_limit=5.0):
        self.env = env  # Can be None
        self.evaluator = evaluator  # Can be ChessCNN, SimpleChessEvaluator, or None
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.root = None
    
    def get_policy_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get move probabilities from CNN evaluator.
        
        Args:
            board: Current chess board
            
        Returns:
            Dictionary mapping legal moves to their prior probabilities
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}
            
        move_probs = {}
        move_evals = []
        
        # If we have an evaluator, use it to evaluate each move
        if self.evaluator is not None:
            for move in legal_moves:
                # Make the move on a copy of the board
                board_copy = board.copy()
                board_copy.push(move)
                
                # Get evaluation for this position
                if hasattr(self.evaluator, 'predict'):
                    # Negate because we're evaluating after our move (opponent's perspective)
                    eval_score = -self.evaluator.predict(board_copy)
                elif hasattr(self.evaluator, 'evaluate'):
                    eval_score = -self.evaluator.evaluate(board_copy)
                else:
                    # Default to equal probability
                    eval_score = 0
                    
                move_evals.append((move, eval_score))
            
            # Convert evaluations to probabilities using softmax
            if move_evals:
                evals = np.array([e for _, e in move_evals])
                
                # Scale evaluations to avoid numerical issues with softmax
                evals = evals - np.max(evals)
                
                # Apply softmax with temperature
                temperature = 1.0  # Adjust this as needed
                exps = np.exp(evals / temperature)
                probs = exps / np.sum(exps)
                
                # Create dictionary with moves and their probabilities
                for i, (move, _) in enumerate(move_evals):
                    move_probs[move] = float(probs[i])
        else:
            # Default: equal probabilities if no evaluator
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probs[move] = prob
        
        return move_probs
    
    def _get_move_index(self, move):
        """
        Helper method to map a chess move to its index in the CNN output.
        """
        # TODO: implement after CNN is complete
        from_square = move.from_square
        to_square = move.to_square
        # There are 64 squares, so from_square * 64 + to_square gives a unique index
        # This creates a space of 64*64=4096 possible moves (though many are invalid)
        return from_square * 64 + to_square
    
    def select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using PUCT formula (used in AlphaZero).
        Combines UCB with policy priors from the CNN.
        """
        if not node.children:
            raise ValueError("Cannot select child from node with no children")
        
        # PUCT formula: Q(s,a) + U(s,a)
        # where U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        c_puct = self.exploration_weight
        total_visits = node.visits
        
        def puct_score(child):
            # Exploitation term - Q(s,a)
            if child.visits == 0:
                q_value = 0
            else:
                # Calculate Q-value based on the perspective of the player to move
                if node.board.turn == chess.WHITE:
                    q_value = child.results[chess.WHITE] / child.visits
                else:
                    q_value = child.results[chess.BLACK] / child.visits
            
            # Exploration term with policy prior - U(s,a)
            move = child.move
            prior_prob = node.prior_probs.get(move, 1.0 / len(node.children))
            u_value = c_puct * prior_prob * math.sqrt(total_visits) / (1 + child.visits)
            
            return q_value + u_value
        
        return max(node.children.values(), key=puct_score)
    
    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand node by adding a child, with move selection informed by CNN policy.
        """
        if not node.untried_moves:
            return None
        
        # Get policy priors if not already calculated
        if not node.prior_probs:
            node.prior_probs = self.get_policy_priors(node.board)
        
        # Choose moves with probability proportional to policy network output
        if node.prior_probs:
            untried_probs = {m: node.prior_probs.get(m, 0.01) for m in node.untried_moves}
            total = sum(untried_probs.values())
            if total > 0:  # Normalize
                probs = [untried_probs[m]/total for m in node.untried_moves]
                move = np.random.choice(node.untried_moves, p=probs)
            else:
                move = np.random.choice(node.untried_moves)
        else:
            move = np.random.choice(node.untried_moves)
        
        # Remove the chosen move from untried moves
        node.untried_moves.remove(move)
        
        # Create new board state
        new_board = node.board.copy()
        new_board.push(move)
        
        # Create and link the new child node
        child = MCTSNode(new_board, parent=node, move=move)
        node.children[move] = child
        
        return child
    
    def _policy_simulation(self, board: chess.Board) -> Dict[chess.Color, float]:
            """
            Run a simulation using the policy network.
            
            Args:
                board: Current board state to simulate from
                
            Returns:
                Dictionary with simulation results
            """
            # Copy the board to avoid modifying the original
            sim_board = board.copy()
            
            # Run simulation until game is over or move limit reached
            move_limit = 100  # Prevent infinite games
            move_count = 0
            
            # Get evaluator for this simulation
            evaluator = self.cnn_model if hasattr(self, 'cnn_model') else self.evaluator
            
            while not sim_board.is_game_over() and move_count < move_limit:
                # Get legal moves
                legal_moves = list(sim_board.legal_moves)
                if not legal_moves:
                    break
                
                # Select move (simple approach - either use evaluator or random)
                if random.random() < 0.8:  # 80% use evaluator logic, 20% random
                    # Simple approach: evaluate each move and choose probabilistically
                    move_values = []
                    for move in legal_moves:
                        # Make the move
                        sim_board.push(move)
                        # Evaluate the position
                        # Use predict method if available (ChessCNN uses predict)
                        if hasattr(evaluator, 'predict'):
                            value = evaluator.predict(sim_board)
                        # Fallback to evaluate method if it exists
                        elif hasattr(evaluator, 'evaluate'):
                            value = evaluator.evaluate(sim_board)
                        else:
                            # Fallback to a random value
                            value = random.random() * 2 - 1  # Value between -1 and 1
                        # Undo the move
                        sim_board.pop()
                        
                        # Store the value
                        move_values.append((move, value))
                    
                    # Convert values to probabilities (higher value = higher probability)
                    values = np.array([v for _, v in move_values])
                    # Apply softmax with temperature
                    temperature = 1.0
                    values = np.exp(values / temperature)
                    probabilities = values / np.sum(values)
                    
                    # Choose move based on probabilities
                    move_idx = np.random.choice(len(legal_moves), p=probabilities)
                    move = legal_moves[move_idx]
                else:
                    # Choose a random move
                    move = random.choice(legal_moves)
                
                # Apply the selected move
                sim_board.push(move)
                move_count += 1
            
            # Process game result
            if sim_board.is_game_over():
                outcome = sim_board.outcome()
                if outcome is None:
                    # Draw
                    return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
                elif outcome.winner == chess.WHITE:
                    return {chess.WHITE: 1.0, chess.BLACK: 0.0, 'draw': 0.0}
                else:  # Black wins
                    return {chess.WHITE: 0.0, chess.BLACK: 1.0, 'draw': 0.0}
            else:
                # If move limit was reached, use the evaluation
                # Use predict method if available (ChessCNN uses predict)
                if hasattr(evaluator, 'predict'):
                    evaluation = evaluator.predict(sim_board)
                # Fallback to evaluate method if it exists
                elif hasattr(evaluator, 'evaluate'):
                    evaluation = evaluator.evaluate(sim_board)
                else:
                    # Fallback to a simple evaluator
                    evaluation = 0.0  # Neutral evaluation
                
                # Convert evaluation to result probabilities
                white_score = min(max((evaluation + 10) / 20, 0), 1)
                black_score = 1 - white_score
                
                return {chess.WHITE: white_score, chess.BLACK: black_score, 'draw': 0.0}
    
    def search(self, board: chess.Board) -> chess.Move:
        """
        Perform CNN-enhanced MCTS from the given board state to find the best move.

        Args:
            board: Current chess board state
            
        Returns:
            The best move according to MCTS
        """

        # Initialize the root node with the current board state
        self.root = MCTSNode(board)

        # Get policy priors for the root node from CNN
        self.root.prior_probs = self.get_policy_priors(board)

        # Set up timing variables
        start_time = time.time()
        num_simulations = 0

        # Run simulations until we hit our limits
        while (num_simulations < self.simulation_limit and 
                time.time() - start_time < self.time_limit):
            # Phase 1: Selection - traverse tree to find a node to expand
            node = self.root
            while node.untried_moves == [] and node.children:  # Fully expanded and non-terminal
                node = self.select_child(node)  # Using the PUCT formula
            
            # Phase 2: Expansion - unless we're at a terminal state, expand tree by one node
            if not node.board.is_game_over() and node.untried_moves:
                node = self.expand(node)  # Using policy-guided expansion
            
            # Phase 3: Simulation - play out the game from the new node using policy network
            result = self._policy_simulation(node.board)
            
            # Phase 4: Backpropagation - update all nodes in the path
            while node is not None:
                # Update visit count
                node.visits += 1
                
                # Update results
                for color in [chess.WHITE, chess.BLACK, 'draw']:
                    node.results[color] += result[color]
                
                node = node.parent
            
            num_simulations += 1

        # Choose the best move based on visits (most robust)
        if not self.root.children:
            # No simulations completed, return a random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return np.random.choice(legal_moves)
            else:
                return None

        # Choose the move with the highest number of visits
        best_move = max(self.root.children.items(), 
                            key=lambda item: item[1].visits)[0]

        # Print some stats
        print(f"CNN-MCTS completed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")
        print(f"Selected move: {best_move}, visits: {self.root.children[best_move].visits}")

        return best_move


# Simple test code
if __name__ == "__main__":
    # Create a board
    board = chess.Board()
    
    # Create a CNN evaluator
    cnn_evaluator = ChessCNN()
    
    # Test evaluation
    print("Testing CNN evaluation:")
    try:
        evaluation = cnn_evaluator.predict(board)
        print(f"Initial position evaluation: {evaluation}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    # Create MCTS
    mcts = ChessMCTS(evaluator=cnn_evaluator, simulation_limit=1000, time_limit=20.0)
    
    # Number of moves to play (or set max_moves=None to play until game over)
    max_moves = None
    move_count = 0
    
    print("\nStarting game:")
    print(board)
    
    # Play the game until it's over or we reach max_moves
    while not board.is_game_over() and (max_moves is None or move_count < max_moves):
        move_count += 1
        print(f"\nMove {move_count}:")
        
        # Get whose turn it is
        side = "White" if board.turn == chess.WHITE else "Black"
        print(f"{side} to move")
        
        # Search for the best move
        start_time = time.time()
        try:
            best_move = mcts.search(board)
            search_time = time.time() - start_time
            
            print(f"Best move found: {best_move} (search time: {search_time:.2f}s)")
            
            # Make the move
            board.push(best_move)
            print("\nBoard after move:")
            print(board)
            
            # Evaluate the new position
            try:
                evaluation = cnn_evaluator.predict(board)
                print(f"Position evaluation: {evaluation:.2f}")
            except Exception as e:
                print(f"Error during evaluation: {e}")
                
        except Exception as e:
            print(f"Error during search: {e}")
            break
    
    # Print game result
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner == chess.WHITE:
            print("\nGame over: White wins!")
        elif outcome.winner == chess.BLACK:
            print("\nGame over: Black wins!")
        else:
            print("\nGame over: Draw!")
        print(f"Termination: {outcome.termination}")
    else:
        print(f"\nGame stopped after {move_count} moves")