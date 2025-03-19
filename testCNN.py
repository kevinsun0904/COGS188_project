import chess
import random
import time
import numpy as np
import math
from typing import Dict, Optional, List, Tuple, Any
import argparse
import os
from ChessEnv import ChessEnv
# ------ Simple Material Evaluator ------

class SimpleMaterialEvaluator:
    """Simple material counting evaluator as fallback when CNN is not available"""
    def __init__(self):
        self.name = "Material Evaluator"
        self.debug = True
        
    def predict(self, board):
        # Piece values
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate material balance
        white_score = sum(len(board.pieces(p, chess.WHITE)) * values[p] for p in values)
        black_score = sum(len(board.pieces(p, chess.BLACK)) * values[p] for p in values)
        
        # Scale to match CNN range (-10 to 10)
        score = (white_score - black_score) * 0.5
        
        if self.debug and random.random() < 0.01:  # Print 1% of evaluations to avoid spam
            print(f"Material evaluation: {score:.2f}")
            
        return score


# ------ Board Representation Functions ------

squares_index = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7
}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
    """Convert a chess board to a 3D tensor representation."""
    # Create a 3D matrix with 14 channels
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    # Add pieces view on the matrix (channels 0-11)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # Add attacks and valid moves (channels 12-13)
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

# ------ CNN Evaluator ------

class ChessCNN:
    """CNN-based position evaluator"""
    def __init__(self, model_path=None):
        self.model = None
        self.max_eval_time = 0.2  # Maximum time for a single evaluation in seconds
        self.debug = True  # Add debug mode
        self.eval_count = 0  # Track number of evaluations
        self.fallback_count = 0  # Track fallback to material evaluation
        
        try:
            import tensorflow as tf
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
                # Try to use global model if available
                try:
                    from tensorflow.keras.models import load_model
                    self.model = load_model('chess_cnn_model.h5', compile=False)
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss="mse",
                        metrics=["mae"]
                    )
                    print("Using global CNN model")
                except Exception as e:
                    print(f"Error loading global model: {e}")
                    self.model = None
        except ImportError:
            print("TensorFlow not available. CNN evaluation will not be used.")
            self.model = None
        
        # Verify model works by testing on an empty board
        if self.model is not None:
            try:
                import chess
                test_board = chess.Board()
                test_eval = self.predict(test_board)
                print(f"Test evaluation on initial position: {test_eval:.2f}")
            except Exception as e:
                print(f"Error testing model: {e}")
                self.model = None
    
    def predict(self, board):
        """Evaluate chess position with better time management"""
        start_time = time.time()
        self.eval_count += 1
        
        # Use cached evaluation if available
        board_key = board.fen()
        if hasattr(self, 'eval_cache') and board_key in self.eval_cache:
            return self.eval_cache[board_key]
        
        # Initialize cache if not exists
        if not hasattr(self, 'eval_cache'):
            self.eval_cache = {}
        
        # If model isn't available, fall back to material count
        if self.model is None:
            eval_score = self._evaluate_material(board)
            self.eval_cache[board_key] = eval_score
            return eval_score
        
        try:
            # Convert board to tensor format
            board_tensor = split_dims(board)
            board_tensor = np.expand_dims(board_tensor, axis=0).astype(np.float32)
            
            # Get prediction with timing safeguard
            raw_eval = self.model.predict(board_tensor, verbose=0)[0][0]
            
            # Scale to [-10, 10] range
            scaled_eval = raw_eval * 10.0
            eval_score = max(-10.0, min(10.0, scaled_eval))
            
            # Cache the result
            self.eval_cache[board_key] = eval_score
            
            # Debug output for significant evaluations
            if self.debug and abs(eval_score) > 3.0 and random.random() < 0.1:
                print(f"Strong evaluation: {eval_score:.2f} for position:")
                print(board)
            
            return eval_score
        except Exception as e:
            if self.debug:
                print(f"Error in CNN prediction: {e}")
            # Fall back to material evaluation
            eval_score = self._evaluate_material(board)
            self.eval_cache[board_key] = eval_score
            return eval_score
    
    def _evaluate_material(self, board):
        """Simple material count evaluation as fallback"""
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        }
        
        white_material = 0
        black_material = 0
        
        for piece_type, value in piece_values.items():
            white_material += len(board.pieces(piece_type, chess.WHITE)) * value
            black_material += len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Scale to match CNN range (-10 to 10)    
        score = (white_material - black_material) * 0.5
        
        if self.debug and self.fallback_count % 100 == 0:
            print(f"Material evaluation: {score:.2f}")
            
        return score

# ------ Monte Carlo Tree Search Node ------

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
        
        # Policy prior probabilities
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
    
    def get_ucb_score(self, parent_visits: int, exploration_weight: float = 2.0) -> float:
        """
        Calculate the UCB score for this node with improved perspective handling.
        
        Args:
            parent_visits: Number of visits to the parent node
            exploration_weight: Controls exploration vs exploitation (higher = more exploration)
            
        Returns:
            UCB score value
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unexplored nodes
        
        # Correctly calculate exploitation term from the current player's perspective
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
        Ensure proper perspective handling.
        """
        self.visits += 1
        
        # Update results based on the provided dictionary
        self.results[chess.WHITE] += result.get(chess.WHITE, 0)
        self.results[chess.BLACK] += result.get(chess.BLACK, 0)
        self.results['draw'] += result.get('draw', 0)
        
        # Validate that results are non-negative
        for color in [chess.WHITE, chess.BLACK, 'draw']:
            if self.results[color] < 0:
                self.results[color] = 0

# ------ Plain Monte Carlo Tree Search ------

class ChessMCTS:
    """Monte Carlo Tree Search without CNN."""
    def __init__(self, env: Optional[ChessEnv] = None, exploration_weight=1.0, 
                 simulation_limit=500, time_limit=5.0, max_depth=25):
        self.env = env
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.max_depth = max_depth  # Max depth for simulations
        self.root = None
        
        # Debug info
        self.debug = True  # Set to False to disable debug prints
    
    def _policy_simulation(self, board: chess.Board) -> Dict[chess.Color, float]:
        """
        Run a random simulation from the given board state.
        
        Args:
            board: Current board state to simulate from
            
        Returns:
            Dictionary with simulation results
        """
        # Copy the board to avoid modifying the original
        sim_board = board.copy()
        
        # Run simulation until game is over or move limit reached
        move_limit = min(self.max_depth, 25)  # Reduced from 100 to speed up simulations
        move_count = 0
        
        # Track time to ensure simulations don't run too long
        sim_start_time = time.time()
        max_sim_time = 0.5  # Max 0.5 seconds per simulation
        
        while (not sim_board.is_game_over() and 
               move_count < move_limit and 
               time.time() - sim_start_time < max_sim_time):
               
            # Get legal moves
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
                
            # Choose a random move
            move = random.choice(legal_moves)
                
            # Apply the selected move
            sim_board.push(move)
            move_count += 1
            
            # Quick time check to bail early if simulation taking too long
            if move_count % 5 == 0 and time.time() - sim_start_time > max_sim_time:
                break
        
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
            # If move limit was reached and we have an environment, use its evaluation
            if self.env is not None:
                # Try to use the environment's evaluation function if available
                try:
                    self.env.board = sim_board.copy()
                    evaluation = self.env.evaluate_position()
                    # Restore original board
                    self.env.board = board.copy()
                    
                    # Convert evaluation to result probabilities (-10 to 10 scale)
                    white_score = min(max((evaluation + 10) / 20, 0), 1)
                    black_score = 1 - white_score
                    
                    return {chess.WHITE: white_score, chess.BLACK: black_score, 'draw': 0.0}
                except Exception:
                    # If evaluation fails, use a draw result
                    return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
            else:
                # No evaluator available, use a draw result
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
    
    def search(self, board: chess.Board) -> chess.Move:
        """
        Perform MCTS from the given board state to find the best move.
        
        Args:
            board: Current chess board state
            
        Returns:
            The best move according to MCTS
        """
        if self.debug:
            print(f"Starting MCTS search with simulation_limit={self.simulation_limit}, time_limit={self.time_limit}s")
        
        # Initialize the root node with the current board state
        self.root = MCTSNode(board)
        
        # Set up timing variables
        start_time = time.time()
        num_simulations = 0
        
        # Hard timeout - will never go beyond this regardless of other parameters
        absolute_timeout = time.time() + min(30, self.time_limit * 2)  # Hard cap at 30 seconds or 2x time_limit
        
        # For timing analysis
        selection_time = 0
        expansion_time = 0
        simulation_time = 0
        backprop_time = 0
        
        # Run simulations until we hit our limits
        while (num_simulations < self.simulation_limit and 
               time.time() - start_time < self.time_limit and
               time.time() < absolute_timeout):
               
            # Exit early if we're close to the time limit
            if time.time() - start_time > self.time_limit * 0.95:
                if self.debug:
                    print(f"Breaking early, close to time limit ({time.time() - start_time:.3f}s/{self.time_limit}s)")
                break
                
            loop_start = time.time()
                
            # Phase 1: Selection - traverse tree to find a node to expand
            selection_start = time.time()
            node = self.root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(self.exploration_weight)
                # Break if taking too long
                if time.time() - selection_start > 0.1:  # Max 0.1s for selection
                    break
            selection_time += time.time() - selection_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 2: Expansion - unless we're at a terminal state, expand tree by one node
            expansion_start = time.time()
            if not node.is_terminal():
                expanded_node = node.expand()
                if expanded_node is not None:
                    node = expanded_node
            expansion_time += time.time() - expansion_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 3: Simulation - play out the game from this node
            simulation_start = time.time()
            result = self._policy_simulation(node.board)
            simulation_time += time.time() - simulation_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 4: Backpropagation - update all nodes in the path
            backprop_start = time.time()
            while node is not None:
                node.update(result)
                node = node.parent
            backprop_time += time.time() - backprop_start
            
            num_simulations += 1
            
            # Break if this single iteration took too long
            iteration_time = time.time() - loop_start
            if iteration_time > 0.5:  # If a single iteration takes more than 0.5s
                if self.debug:
                    print(f"Breaking after long iteration: {iteration_time:.3f}s")
                break
        
        # Choose the best move based on visits (most robust)
        if not self.root.children:
            # No simulations completed, return a random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = random.choice(legal_moves)
                if self.debug:
                    print(f"No simulations completed. Choosing random move: {random_move}")
                return random_move
            else:
                return None
        
        total_time = time.time() - start_time
        
        # Choose the move with the highest number of visits
        best_move = max(self.root.children.items(), 
                         key=lambda item: item[1].visits)[0]
        
        # Print timing stats if debug is enabled
        if self.debug:
            print(f"MCTS completed {num_simulations} simulations in {total_time:.2f} seconds")
            print(f"Time usage: Selection {selection_time/total_time*100:.1f}%, "
                 f"Expansion {expansion_time/total_time*100:.1f}%, "
                 f"Simulation {simulation_time/total_time*100:.1f}%, "
                 f"Backprop {backprop_time/total_time*100:.1f}%")
            print(f"Selected move: {best_move}, visits: {self.root.children[best_move].visits}")
            
            # Print top moves by visit count
            sorted_moves = sorted(
                self.root.children.items(),
                key=lambda item: item[1].visits,
                reverse=True
            )
            print("Top moves by visit count:")
            for move, node in sorted_moves[:3]:
                print(f"  {move}: {node.visits} visits, "
                     f"win%: {node.results[chess.WHITE]/node.visits*100:.1f}% "
                     f"(White perspective)")
        
        return best_move
    
# Add this class immediately after the ChessMCTS class in your code

# ------ Monte Carlo Tree Search with CNN ------

class ChessMCTSCNN:
    """Monte Carlo Tree Search with CNN policy network."""
    def __init__(self, evaluator=None, exploration_weight=1.0,  # Reduced from 2.0
                 simulation_limit=500, time_limit=5.0, max_depth=25):
        self.evaluator = evaluator  # ChessCNN instance
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.max_depth = max_depth  # Max depth for simulations
        self.root = None
        
        # Debug info
        self.debug = True  # Set to False to disable debug prints
        
        # Stats counters
        self.policy_fallbacks = 0
        self.total_policy_calls = 0
    
    def get_policy_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get move probabilities from CNN evaluator.
        
        Args:
            board: Current chess board
            
        Returns:
            Dictionary mapping legal moves to their prior probabilities
        """
        self.total_policy_calls += 1
        start_time = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}
            
        move_probs = {}
        move_evals = []
        
        # If we have an evaluator, use it to evaluate each move
        if self.evaluator is not None:
            # Limit per-move evaluation time
            max_eval_time = min(1.0, self.time_limit / 5)  # Increased from 0.5s to 1.0s
            
            for move in legal_moves:
                # Check if we're over time budget for all evaluations
                if time.time() - start_time > max_eval_time:
                    # Fall back to equal probabilities if taking too long
                    self.policy_fallbacks += 1
                    if self.debug:
                        print(f"Evaluation time limit exceeded, using equal probabilities ({time.time() - start_time:.3f}s)")
                        print(f"Policy fallbacks: {self.policy_fallbacks}/{self.total_policy_calls}")
                    prob = 1.0 / len(legal_moves)
                    return {move: prob for move in legal_moves}
                
                # Make the move on a copy of the board
                board_copy = board.copy()
                board_copy.push(move)
                
                # Get evaluation for this position
                if hasattr(self.evaluator, 'predict'):
                    # Negate because we're evaluating after our move (opponent's perspective)
                    eval_score = -self.evaluator.predict(board_copy)
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
                temperature = 0.5  # Reduced from 1.0 for more deterministic selection
                exps = np.exp(evals / temperature)
                probs = exps / np.sum(exps)
                
                # Create dictionary with moves and their probabilities
                for i, (move, _) in enumerate(move_evals):
                    move_probs[move] = float(probs[i])
                
                # Debug: print top moves by probability
                if self.debug:
                    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    if random.random() < 0.1:  # Only print 10% of the time to avoid spam
                        print("\nTop moves by probability:")
                        for move, prob in sorted_moves:
                            print(f"  {move}: {prob:.3f} (eval: {move_evals[list(move_probs.keys()).index(move)][1]:.2f})")
        else:
            # Default: equal probabilities if no evaluator
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probs[move] = prob
        
        return move_probs
    
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
        
        best_child = max(node.children.values(), key=puct_score)
        return best_child
    
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
        move_limit = min(self.max_depth, 25)  # Reduced from 100 to speed up simulations
        move_count = 0
        
        # Track time to ensure simulations don't run too long
        sim_start_time = time.time()
        max_sim_time = 0.5  # Max 0.5 seconds per simulation
        
        while (not sim_board.is_game_over() and 
               move_count < move_limit and 
               time.time() - sim_start_time < max_sim_time):
               
            # Get legal moves
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
            
            # Quick time check to bail early if simulation taking too long
            if move_count % 5 == 0 and time.time() - sim_start_time > max_sim_time:
                break
            
            # Select move (90% use evaluator logic, 10% random) - increased from 80/20
            use_evaluator = random.random() < 0.9 and self.evaluator is not None
            
            if use_evaluator:
                # Evaluate each move and choose probabilistically
                move_values = []
                
                # Only evaluate if we have time
                if time.time() - sim_start_time < max_sim_time * 0.8:
                    for move in legal_moves[:min(5, len(legal_moves))]:  # Limit to top 5 moves for speed
                        # Make the move
                        sim_board.push(move)
                        # Evaluate the position
                        if hasattr(self.evaluator, 'predict'):
                            value = self.evaluator.predict(sim_board)
                        else:
                            # Fallback to a random value
                            value = random.random() * 2 - 1  # Value between -1 and 1
                        # Undo the move
                        sim_board.pop()
                        
                        # Store the value
                        move_values.append((move, value))
                    
                    # Convert values to probabilities if we have evaluations
                    if move_values:
                        values = np.array([v for _, v in move_values])
                        # Apply softmax with temperature
                        temperature = 0.5  # Reduced from 1.0
                        values = values - np.max(values)  # For numerical stability
                        exps = np.exp(values / temperature)
                        probabilities = exps / np.sum(exps)
                        
                        # Choose move based on probabilities
                        move_idx = np.random.choice(len(move_values), p=probabilities)
                        move = move_values[move_idx][0]
                    else:
                        move = random.choice(legal_moves)
                else:
                    # Not enough time, use random move
                    move = random.choice(legal_moves)
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
            if self.evaluator is not None and time.time() - sim_start_time < max_sim_time:
                if hasattr(self.evaluator, 'predict'):
                    evaluation = self.evaluator.predict(sim_board)
                else:
                    evaluation = 0.0  # Neutral evaluation
                
                # Convert evaluation to result probabilities
                white_score = min(max((evaluation + 10) / 20, 0), 1)
                black_score = 1 - white_score
                
                return {chess.WHITE: white_score, chess.BLACK: black_score, 'draw': 0.0}
            else:
                # No evaluator or out of time, return draw
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
    
    def search(self, board: chess.Board) -> chess.Move:
        """
        Perform CNN-enhanced MCTS from the given board state to find the best move.
        
        Args:
            board: Current chess board state
            
        Returns:
            The best move according to MCTS
        """
        if self.debug:
            print(f"Starting CNN-MCTS search with simulation_limit={self.simulation_limit}, time_limit={self.time_limit}s")
        
        # Initialize the root node with the current board state
        self.root = MCTSNode(board)
        
        # Set up timing variables
        start_time = time.time()
        
        # Get policy priors for the root node from CNN
        policy_start = time.time()
        self.root.prior_probs = self.get_policy_priors(board)
        policy_time = time.time() - policy_start
        
        if self.debug:
            if policy_time > 0.5:
                print(f"Warning: Initial policy evaluation took {policy_time:.2f}s")
            print(f"Policy priors calculated for {len(self.root.prior_probs)} moves")
        
        num_simulations = 0
        
        # Hard timeout - will never go beyond this regardless of other parameters
        absolute_timeout = time.time() + min(30, self.time_limit * 2)  # Hard cap at 30 seconds or 2x time_limit
        
        # For timing analysis
        selection_time = 0
        expansion_time = 0
        simulation_time = 0
        backprop_time = 0
        
        # Run simulations until we hit our limits
        while (num_simulations < self.simulation_limit and 
               time.time() - start_time < self.time_limit and
               time.time() < absolute_timeout):
               
            # Exit early if we're close to the time limit
            if time.time() - start_time > self.time_limit * 0.95:
                if self.debug:
                    print(f"Breaking early, close to time limit ({time.time() - start_time:.3f}s/{self.time_limit}s)")
                break
                
            loop_start = time.time()
                
            # Phase 1: Selection - traverse tree to find a node to expand
            selection_start = time.time()
            node = self.root
            select_depth = 0
            while not node.is_terminal() and node.is_fully_expanded():
                # Use self.select_child instead of node.select_child
                node = self.select_child(node)  # Fixed the recursive bug
                select_depth += 1
                # Break if taking too long or getting too deep
                if time.time() - selection_start > 0.1 or select_depth > 30:  # Max 0.1s for selection
                    break
            selection_time += time.time() - selection_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 2: Expansion - unless we're at a terminal state, expand tree by one node
            expansion_start = time.time()
            if not node.is_terminal():
                expanded_node = self.expand(node)  # Using policy-guided expansion
                if expanded_node is not None:
                    node = expanded_node
            expansion_time += time.time() - expansion_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 3: Simulation - play out the game from the new node using policy network
            simulation_start = time.time()
            result = self._policy_simulation(node.board)
            simulation_time += time.time() - simulation_start
            
            # Check time again
            if time.time() - start_time > self.time_limit:
                break
            
            # Phase 4: Backpropagation - update all nodes in the path
            backprop_start = time.time()
            backprop_nodes = 0
            while node is not None:
                node.update(result)
                node = node.parent
                backprop_nodes += 1
                # Safety check to prevent infinite loops
                if backprop_nodes > 100:  # Should never be this deep
                    print("Warning: Backpropagation hit safety limit")
                    break
            backprop_time += time.time() - backprop_start
            
            num_simulations += 1
            
            # Break if this single iteration took too long
            iteration_time = time.time() - loop_start
            if iteration_time > 0.5:  # If a single iteration takes more than 0.5s
                if self.debug:
                    print(f"Breaking after long iteration: {iteration_time:.3f}s")
                break
        
        # Choose the best move based on visits (most robust)
        if not self.root.children:
            # No simulations completed, return a random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = random.choice(legal_moves)
                if self.debug:
                    print(f"No simulations completed. Choosing random move: {random_move}")
                return random_move
            else:
                return None
        
        total_time = time.time() - start_time
        
        # Choose the best move using a combination of visits and evaluation score
        # This helps when simulation count is low
        best_move = None
        best_score = float('-inf')
        min_visits_threshold = max(1, num_simulations // 10)  # At least 10% of simulations
        
        for move, child in self.root.children.items():
            if child.visits < min_visits_threshold:
                continue
                
            # Calculate score based on win ratio and visit count
            win_ratio = child.results[board.turn] / child.visits if child.visits > 0 else 0
            visit_ratio = child.visits / num_simulations if num_simulations > 0 else 0
            
            # Combined score with higher weight on visits
            score = 0.7 * win_ratio + 0.3 * visit_ratio
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Fall back to most visited if no move meets threshold
        if best_move is None:
            best_move = max(self.root.children.items(), 
                           key=lambda item: item[1].visits)[0]
        
        # Print timing stats if debug is enabled
        if self.debug:
            print(f"CNN-MCTS completed {num_simulations} simulations in {total_time:.2f} seconds")
            print(f"Time usage: Selection {selection_time/total_time*100:.1f}%, "
                 f"Expansion {expansion_time/total_time*100:.1f}%, "
                 f"Simulation {simulation_time/total_time*100:.1f}%, "
                 f"Backprop {backprop_time/total_time*100:.1f}%")
            print(f"Selected move: {best_move}, visits: {self.root.children[best_move].visits}")
            
            # Print top moves by visit count
            sorted_moves = sorted(
                self.root.children.items(),
                key=lambda item: item[1].visits,
                reverse=True
            )
            print("Top moves by visit count:")
            for move, node in sorted_moves[:3]:
                print(f"  {move}: {node.visits} visits, "
                     f"win%: {node.results[board.turn]/node.visits*100:.1f}% "
                     f"(perspective of side to move)")
                
            # Compare with material evaluation's preferences
            print("\nComparing with material evaluation:")
            material_evals = []
            for move in [m for m, _ in sorted_moves[:3]]:
                board_copy = board.copy()
                board_copy.push(move)
                eval_score = SimpleMaterialEvaluator().predict(board_copy)
                material_evals.append((move, eval_score))
            
            for move, eval_score in sorted(material_evals, key=lambda x: x[1], reverse=(board.turn == chess.WHITE)):
                print(f"  {move}: material eval = {eval_score:.2f}")
        
        return best_move

# ------ Chess Agent Classes ------

class RandomAgent:
    """Agent that selects moves randomly"""
    def __init__(self):
        self.name = "Random Agent"
    
    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)

class HumanAgent:
    """Human player agent"""
    def __init__(self):
        self.name = "Human Player"
    
    def select_move(self, board):
        valid_move = False
        while not valid_move:
            try:
                print("\nCurrent board:")
                print(board)
                move_input = input("\nEnter your move (UCI format, e.g., 'e2e4') or 'help' for legal moves: ")
                
                if move_input.lower() == 'help':
                    print("\nLegal moves:")
                    for move in board.legal_moves:
                        print(f"  {move} ({board.san(move)})")
                    continue
                
                if move_input.lower() == 'quit' or move_input.lower() == 'exit':
                    print("Exiting game...")
                    return None
                
                # Parse move
                move = chess.Move.from_uci(move_input)
                
                # Check if move is legal
                if move in board.legal_moves:
                    valid_move = True
                    return move
                else:
                    print("Illegal move! Try again.")
                    
            except ValueError:
                print("Invalid input! Please use UCI format (e.g., 'e2e4') or type 'help'.")

class MCTSAgent:
    """Agent that uses Monte Carlo Tree Search"""
    def __init__(self, env=None, simulation_limit=500, time_limit=5.0):
        self.name = "MCTS Agent"
        self.mcts = ChessMCTS(
            env=env,
            simulation_limit=simulation_limit,
            time_limit=time_limit
        )
    
    def select_move(self, board):
        return self.mcts.search(board)

class SimpleMaterialEvaluator:
    """Simple material counting evaluator as fallback when CNN is not available"""
    def __init__(self):
        self.name = "Material Evaluator"
        
    def predict(self, board):
        # Piece values
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Calculate material balance
        white_score = sum(len(board.pieces(p, chess.WHITE)) * values[p] for p in values)
        black_score = sum(len(board.pieces(p, chess.BLACK)) * values[p] for p in values)
        
        # Scale to match CNN range (-10 to 10)
        return (white_score - black_score) * 0.5

class MCTSCNNAgent:
    """Agent that uses Monte Carlo Tree Search with CNN evaluation"""
    def __init__(self, model_path=None, simulation_limit=500, time_limit=5.0):
        self.name = "MCTS+CNN Agent"
        
        # Try to load CNN with better error handling
        try:
            self.evaluator = ChessCNN(model_path)
            print("Using CNN evaluator")
            
            # Test evaluator
            test_board = chess.Board()
            test_eval = self.evaluator.predict(test_board)
            print(f"Test evaluation: {test_eval:.2f}")
            
            # If test evaluation is close to 0 (within ±0.5), warn about potential issues
            if abs(test_eval) < 0.5:
                print("WARNING: CNN evaluation close to 0 on initial position. Model may need training.")
        except Exception as e:
            print(f"CNN evaluator not available: {e}")
            self.evaluator = SimpleMaterialEvaluator()
            self.name = "MCTS+Material Agent"
        
        # Initialize MCTS with better parameters
        self.mcts = ChessMCTSCNN(
            evaluator=self.evaluator,
            exploration_weight=1.5,    # Increased from 1.0
            simulation_limit=simulation_limit,
            time_limit=time_limit
        )
    # Add this to your MCTSCNNAgent class if it's missing
    def select_move(self, board):
        """Choose the best move using MCTS search"""
        move = self.mcts.search(board)
        
        # Validate move is legal
        if move is not None and move not in board.legal_moves:
            print(f"WARNING: {self.name} tried to play illegal move {move}!")
            # Fall back to random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
            
        return move

# ------ Chess Match Manager ------

class ChessMatch:
    """Class to manage chess matches between agents"""
    def __init__(self, white_agent, black_agent, stockfish_path=None, max_moves=200):
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.max_moves = max_moves
        self.stockfish_path = stockfish_path
        
        # Initialize environment if Stockfish path is provided
        if stockfish_path:
            try:
                from stockfish import Stockfish
                self.env = ChessEnv(stockfish_path=stockfish_path)
                print(f"Using Stockfish at: {stockfish_path}")
            except (ImportError, FileNotFoundError) as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                print("Will play without Stockfish evaluation.")
                self.env = None
        else:
            self.env = None
        
        self.board = chess.Board()
        self.move_history = []
    
    def play_game(self, display=True):
        """Play a complete game between the two agents"""
        self.board = chess.Board()
        self.move_history = []
        move_count = 0
        
        if display:
            print(f"\nNew game: {self.white_agent.name} (White) vs {self.black_agent.name} (Black)")
            print("\nInitial board:")
            print(self.board)
        
        while not self.board.is_game_over() and move_count < self.max_moves:
            move_count += 1
            
            # Get the current agent
            current_agent = self.white_agent if self.board.turn == chess.WHITE else self.black_agent
            current_color = "White" if self.board.turn == chess.WHITE else "Black"
            
            if display:
                print(f"\nMove {move_count}: {current_color} ({current_agent.name}) thinking...")
            
            start_time = time.time()
            
            # Save the board state before making the move (for SAN notation)
            board_before_move = self.board.copy()
            
            move = current_agent.select_move(self.board)
            elapsed = time.time() - start_time
            
            # Check for user exit in human agent
            if move is None:
                if display:
                    print(f"{current_color} ({current_agent.name}) has resigned or quit.")
                break
            
            if move not in self.board.legal_moves:
                if display:
                    print(f"Error: {current_color} produced an illegal move: {move}")
                break
            
            # Make the move
            self.board.push(move)
            self.move_history.append(move)
            
            if display:
                # Get the SAN notation before making the move
                try:
                    san_move = board_before_move.san(move)
                except Exception:
                    san_move = move.uci()  # Fallback to UCI notation if SAN fails
                
                print(f"{current_color} plays: {move} ({san_move}) in {elapsed:.2f} seconds")
                print(self.board)
                
                # Show evaluation if available
                if self.env:
                    try:
                        self.env.board = self.board.copy()
                        eval_score = self.env.evaluate_position()
                        print(f"Position evaluation: {eval_score:.2f}")
                    except Exception as e:
                        print(f"Error getting evaluation: {e}")
        
        # Game over
        if display:
            print("\nGame over!")
            result = self.get_result()
            print(f"Result: {result['result_text']}")
            print(f"Termination: {result['termination']}")
        
        return self.get_result()
    
    def get_result(self):
        """Get the game result"""
        if not self.board.is_game_over():
            return {
                "result": "unfinished",
                "result_text": "Game not finished",
                "termination": None,
                "winner": None
            }
        
        outcome = self.board.outcome()
        if outcome is None:
            return {
                "result": "1/2-1/2",
                "result_text": "Draw",
                "termination": "unknown",
                "winner": None
            }
        
        if outcome.winner == chess.WHITE:
            winner = "White"
            result = "1-0"
        elif outcome.winner == chess.BLACK:
            winner = "Black"
            result = "0-1"
        else:
            winner = None
            result = "1/2-1/2"
        
        return {
            "result": result,
            "result_text": f"{winner} wins" if winner else "Draw",
            "termination": outcome.termination.name,
            "winner": winner
        }
    
    def export_pgn(self):
        """Export the game in PGN format"""
        import datetime
        import chess.pgn
        
        game = chess.pgn.Game()
        game.headers["Event"] = "AI Chess Match"
        game.headers["Site"] = "Python Chess Engine"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = self.white_agent.name
        game.headers["Black"] = self.black_agent.name
        game.headers["Result"] = self.board.result()
        
        # Add moves
        node = game
        for move in self.move_history:
            node = node.add_variation(move)
        
        return str(game)

# ------ Tournament Management ------

class ChessTournament:
    """Class to manage a tournament between different agents"""
    def __init__(self, agents, stockfish_path=None, games_per_matchup=2, max_moves=200):
        self.agents = agents
        self.stockfish_path = stockfish_path
        self.games_per_matchup = games_per_matchup
        self.max_moves = max_moves
        self.results = {}
        
    def run_tournament(self, display=True):
        """Run a round-robin tournament between all agents"""
        # Initialize results dictionary
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent1 != agent2:
                    self.results[(agent1.name, agent2.name)] = {
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "points": 0
                    }
        
        # Play all matchups
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i != j:  # Don't play against self
                    if display:
                        print(f"\n==== {agent1.name} vs {agent2.name} ====")
                    
                    # Play specified number of games per matchup
                    for game_num in range(self.games_per_matchup):
                        # Alternate colors
                        if game_num % 2 == 0:
                            white_agent, black_agent = agent1, agent2
                        else:
                            white_agent, black_agent = agent2, agent1
                        
                        # Create match and play game
                        match = ChessMatch(
                            white_agent=white_agent,
                            black_agent=black_agent,
                            stockfish_path=self.stockfish_path,
                            max_moves=self.max_moves
                        )
                        result = match.play_game(display=display)
                        
                        # Update results
                        if result["winner"] == "White":
                            if white_agent == agent1:
                                self.results[(agent1.name, agent2.name)]["wins"] += 1
                                self.results[(agent1.name, agent2.name)]["points"] += 1
                            else:
                                self.results[(agent1.name, agent2.name)]["losses"] += 1
                        elif result["winner"] == "Black":
                            if black_agent == agent1:
                                self.results[(agent1.name, agent2.name)]["wins"] += 1
                                self.results[(agent1.name, agent2.name)]["points"] += 1
                            else:
                                self.results[(agent1.name, agent2.name)]["losses"] += 1
                        else:  # Draw
                            self.results[(agent1.name, agent2.name)]["draws"] += 1
                            self.results[(agent1.name, agent2.name)]["points"] += 0.5
        
        # Display tournament results
        if display:
            self.display_results()
        
        return self.results
    
    def display_results(self):
        """Display tournament results in a nice format"""
        print("\n==== Tournament Results ====")
        
        # Calculate total points for each agent
        total_points = {}
        for agent in self.agents:
            total_points[agent.name] = 0
        
        for (agent1_name, agent2_name), result in self.results.items():
            total_points[agent1_name] += result["points"]
        
        # Sort agents by total points
        sorted_agents = sorted(
            [agent.name for agent in self.agents],
            key=lambda name: total_points[name],
            reverse=True
        )
        
        # Display standings
        print("\nStandings:")
        for i, agent_name in enumerate(sorted_agents):
            print(f"{i+1}. {agent_name}: {total_points[agent_name]} points")
        
        # Display matchup details
        print("\nMatchup Details:")
        for (agent1_name, agent2_name), result in self.results.items():
            print(f"{agent1_name} vs {agent2_name}: +{result['wins']} -{result['losses']} ={result['draws']} ({result['points']} points)")


# ------ Monte Carlo Tree Search with CNN ------

class ChessMCTSCNN:
    """Monte Carlo Tree Search with CNN policy network."""
    def __init__(self, evaluator=None, exploration_weight=1.0, 
                 simulation_limit=500, time_limit=5.0, max_depth=25):
        self.evaluator = evaluator  # ChessCNN instance
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.max_depth = max_depth  # Max depth for simulations
        self.root = None
        
        # Debug info
        self.debug = True  # Set to False to disable debug prints
    
    def get_policy_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get move probabilities from CNN evaluator with improved handling
        """
        start_time = time.time()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}
            
        move_probs = {}
        
        # If we have an evaluator, use it to evaluate each move
        if self.evaluator is not None:
            # Make a copy of the board for evaluations
            board_copy = board.copy()
            
            # Evaluate each move
            move_evals = []
            for move in legal_moves:
                # Make the move
                board_copy.push(move)
                
                try:
                    # Evaluate the resulting position - IMPORTANT: negate for opponent's perspective
                    eval_score = -self.evaluator.predict(board_copy)
                    move_evals.append((move, eval_score))
                except Exception as e:
                    # Log error but continue with default value
                    if self.debug:
                        print(f"Error evaluating move {move}: {e}")
                    move_evals.append((move, 0))
                
                # Unmake the move
                board_copy.pop()
                
                # Check for timeout
                if time.time() - start_time > 3:  # Increased timeout for more reliable evaluations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if self.debug:
                        print("Policy prior calculation timeout")
                    break
            
            # If we evaluated at least some moves, use softmax to convert to probabilities
            if move_evals:
                evals = np.array([e for _, e in move_evals])
                
                # For numerical stability
                evals = evals - np.max(evals)
                
                # Lower temperature (0.3) for more deterministic selection
                temperature = 0.3
                exps = np.exp(evals / temperature)
                probs = exps / np.sum(exps)
                
                # Assign probabilities to moves
                for i, (move, _) in enumerate(move_evals):
                    move_probs[move] = float(probs[i])
                
                # Fill in any moves that weren't evaluated
                evaluated_moves = {m for m, _ in move_evals}
                for move in legal_moves:
                    if move not in evaluated_moves:
                        move_probs[move] = 0.01  # Small non-zero probability
            else:
                # Default to uniform distribution if no evaluations
                prob = 1.0 / len(legal_moves)
                for move in legal_moves:
                    move_probs[move] = prob
        else:
            # No evaluator, use uniform distribution
            prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_probs[move] = prob
        
        # Normalize probabilities
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            for move in move_probs:
                move_probs[move] /= total_prob
        
        return move_probs
    
    def select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child node using corrected PUCT formula
        """
        if not node.children:
            raise ValueError("Cannot select child from node with no children")
        
        # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        c_puct = self.exploration_weight
        total_visits = max(1, sum(child.visits for child in node.children.values()))
        
        def puct_score(child):
            # Exploitation term (Q value) - perspective correction
            if child.visits == 0:
                q_value = 0
            else:
                # Important: Use the perspective of the player to move in the parent node
                if node.board.turn == chess.WHITE:
                    q_value = child.results[chess.WHITE] / child.visits
                else:
                    q_value = child.results[chess.BLACK] / child.visits
            
            # Exploration term with policy prior
            move = child.move
            prior_prob = node.prior_probs.get(move, 1.0 / max(1, len(node.children)))
            u_value = c_puct * prior_prob * math.sqrt(total_visits) / (1 + child.visits)
            
            return q_value + u_value
        
        best_child = max(node.children.values(), key=puct_score)
        return best_child
    
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
        Run a more focused simulation using the policy network
        """
        # For terminal nodes, return the actual game result
        if board.is_game_over():
            outcome = board.outcome()
            if outcome is None:  # Draw
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
            elif outcome.winner == chess.WHITE:
                return {chess.WHITE: 1.0, chess.BLACK: 0.0, 'draw': 0.0}
            else:  # Black wins
                return {chess.WHITE: 0.0, chess.BLACK: 1.0, 'draw': 0.0}
        
        # For non-terminal nodes, evaluate with the neural network if available
        if self.evaluator is not None and hasattr(self.evaluator, 'predict'):
            try:
                # Get network evaluation (scaled to [0,1] range)
                evaluation = self.evaluator.predict(board)
                
                # Scale to win probability for white (between 0 and 1)
                white_win_prob = min(max((evaluation + 10) / 20, 0), 1)
                
                # Create result dictionary
                return {
                    chess.WHITE: white_win_prob,
                    chess.BLACK: 1 - white_win_prob,
                    'draw': 0.0
                }
            except Exception as e:
                if self.debug:
                    print(f"Error in simulation evaluation: {e}")
                # Fall back to a balanced result on error
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 0.0}
        else:
            # No evaluator, return balanced result
            return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 0.0}
    
    def search(self, board: chess.Board) -> chess.Move:
        """
        Perform CNN-enhanced MCTS with optimized search
        """
        if self.debug:
            print(f"Starting CNN-MCTS search with simulation_limit={self.simulation_limit}, time_limit={self.time_limit}s")
        
        # Initialize the root node
        self.root = MCTSNode(board)
        
        # Set up timing variables
        start_time = time.time()
        num_simulations = 0
        
        # Get policy priors for the root node
        self.root.prior_probs = self.get_policy_priors(board)
        
        # Run simulations until we hit our limits
        while (num_simulations < self.simulation_limit and 
            time.time() - start_time < self.time_limit):
            
            # Phase 1: Selection
            node = self.root
            while not node.is_terminal() and node.is_fully_expanded():
                node = self.select_child(node)
            
            # Phase 2: Expansion
            if not node.is_terminal():
                expanded_node = self.expand(node)
                if expanded_node is not None:
                    node = expanded_node
            
            # Phase 3: Simulation/Evaluation
            result = self._policy_simulation(node.board)
            
            # Phase 4: Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent
            
            num_simulations += 1
        
        # Choose the best move
        if not self.root.children:
            # No simulations completed, return random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
        
        # PRIMARY METHOD: Select move with highest visit count
        best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
        
        # Debug output
        if self.debug:
            print(f"Completed {num_simulations} simulations in {time.time() - start_time:.2f}s")
            print(f"Selected move: {best_move}")
            
            # Print top moves
            sorted_moves = sorted(
                self.root.children.items(),
                key=lambda item: item[1].visits,
                reverse=True
            )
            print("Top moves:")
            for move, node in sorted_moves[:3]:
                win_rate = node.results[board.turn] / node.visits if node.visits > 0 else 0
                print(f"  {move}: {node.visits} visits, win%: {win_rate*100:.1f}%")
        
        return best_move
    

# ------ Main Function ------

def main():
    """Main function to run the chess agent comparison"""
    parser = argparse.ArgumentParser(description="Chess Agents Comparison")
    parser.add_argument("--stockfish", type=str, help="Path to Stockfish executable", default=None)
    parser.add_argument("--model", type=str, help="Path to CNN model", default=None)
    parser.add_argument("--simulations", type=int, default=5000, help="Number of MCTS simulations")
    parser.add_argument("--time", type=float, default=60, help="Time limit for MCTS search (seconds)")
    args = parser.parse_args()
    
    # Find Stockfish path if not provided
    stockfish_path = args.stockfish
    if stockfish_path is None:
        # Try some common locations
        common_paths = [
            "stockfish",  # In PATH
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "C:\\Program Files\\stockfish\\stockfish.exe",
            os.path.expanduser("~/stockfish/stockfish")
        ]
        for path in common_paths:
            if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                stockfish_path = path
                break
    
    # Create agents
    random_agent = RandomAgent()
    mcts_agent = MCTSAgent(simulation_limit=args.simulations, time_limit=args.time)
    human_agent = HumanAgent()
    
    # Always create the CNN agent - it will fall back to material evaluation if needed
    print(f"Attempting to load CNN model from: {args.model if args.model else 'chess_cnn_model.h5'}")
    cnn_agent = MCTSCNNAgent(model_path=args.model, simulation_limit=args.simulations, time_limit=args.time)
    
    # Display menu
    while True:
        print("\n===== Chess Agents Comparison =====")
        print("1. Random Agent vs MCTS Agent")
        print("2. Random Agent vs MCTS+CNN Agent")
        print("3. MCTS Agent vs MCTS+CNN Agent")
        print("4. Play as Human vs Random Agent")
        print("5. Play as Human vs MCTS Agent")
        print("6. Play as Human vs MCTS+CNN Agent")
        print("7. Full Tournament (All Agents)")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        # Process choice
        if choice == "1":
            # Random vs MCTS
            run_match(random_agent, mcts_agent, stockfish_path)
        elif choice == "2":
            # Random vs MCTS+CNN
            if cnn_agent is None:
                print("CNN agent not available. Please choose another option.")
                continue
            run_match(random_agent, cnn_agent, stockfish_path)
        elif choice == "3":
            # MCTS vs MCTS+CNN
            if cnn_agent is None:
                print("CNN agent not available. Please choose another option.")
                continue
            run_match(mcts_agent, cnn_agent, stockfish_path)
        elif choice == "4":
            # Human vs Random
            play_as_human(random_agent, stockfish_path)
        elif choice == "5":
            # Human vs MCTS
            play_as_human(mcts_agent, stockfish_path)
        elif choice == "6":
            # Human vs MCTS+CNN
            if cnn_agent is None:
                print("CNN agent not available. Please choose another option.")
                continue
            play_as_human(cnn_agent, stockfish_path)
        elif choice == "7":
            # Full tournament
            agents = [random_agent, mcts_agent]
            if cnn_agent is not None:
                agents.append(cnn_agent)
            
            # Create and run tournament
            tournament = ChessTournament(
                agents=agents,
                stockfish_path=stockfish_path,
                games_per_matchup=2
            )
            tournament.run_tournament(display=True)
        elif choice == "8":
            # Exit
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 8.")

def run_match(agent1, agent2, stockfish_path):
    """Run a match between two AI agents"""
    print(f"\nMatch: {agent1.name} vs {agent2.name}")
    
    # Ask for color assignment
    print("\nWho should play as White?")
    print(f"1. {agent1.name}")
    print(f"2. {agent2.name}")
    print("3. Alternate colors for multiple games")
    
    color_choice = input("Enter your choice (1-3): ").strip()
    num_games = 1
    
    if color_choice == "1":
        white_agent, black_agent = agent1, agent2
    elif color_choice == "2":
        white_agent, black_agent = agent2, agent1
    elif color_choice == "3":
        num_games = int(input("How many games to play? "))
        # Will alternate within the loop
        white_agent, black_agent = agent1, agent2
    else:
        print("Invalid choice. Using default: agent1 as White, agent2 as Black")
        white_agent, black_agent = agent1, agent2
    
    # Results tracking
    results = {
        agent1.name: {"wins": 0, "losses": 0, "draws": 0},
        agent2.name: {"wins": 0, "losses": 0, "draws": 0}
    }
    
    # Play the specified number of games
    for game_num in range(num_games):
        if color_choice == "3" and game_num % 2 == 1:
            # Alternate colors
            white_agent, black_agent = black_agent, white_agent
        
        print(f"\n=== Game {game_num + 1} ===")
        print(f"White: {white_agent.name}")
        print(f"Black: {black_agent.name}")
        
        # Create match and play
        match = ChessMatch(
            white_agent=white_agent,
            black_agent=black_agent,
            stockfish_path=stockfish_path,
            max_moves=1000
        )
        result = match.play_game(display=True)
        
        # Update results
        if result["winner"] == "White":
            results[white_agent.name]["wins"] += 1
            results[black_agent.name]["losses"] += 1
        elif result["winner"] == "Black":
            results[black_agent.name]["wins"] += 1
            results[white_agent.name]["losses"] += 1
        else:  # Draw
            results[white_agent.name]["draws"] += 1
            results[black_agent.name]["draws"] += 1
    
    # Print overall results
    print("\n=== Match Results ===")
    for agent_name, stats in results.items():
        print(f"{agent_name}: {stats['wins']} wins, {stats['losses']} losses, {stats['draws']} draws")
    
    # Export PGN if only one game was played
    if num_games == 1:
        pgn = match.export_pgn()
        print("\nPGN of the game:")
        print(pgn)
    
    input("\nPress Enter to continue...")

def play_as_human(ai_agent, stockfish_path):
    """Play as human against an AI agent"""
    human_agent = HumanAgent()
    
    # Ask player for color preference
    color_choice = input("\nPlay as (w)hite or (b)lack? ").lower()
    if color_choice.startswith('b'):
        white_agent, black_agent = ai_agent, human_agent
        print(f"You will play as Black against {ai_agent.name}")
    else:
        white_agent, black_agent = human_agent, ai_agent
        print(f"You will play as White against {ai_agent.name}")
    
    # Create match and play
    match = ChessMatch(
        white_agent=white_agent,
        black_agent=black_agent,
        stockfish_path=stockfish_path,
        max_moves=1000
    )
    result = match.play_game(display=True)
    
    # Print final result
    print(f"\nGame over! Result: {result['result_text']}")
    
    # Export PGN
    pgn = match.export_pgn()
    print("\nPGN of the game:")
    print(pgn)
    
    input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()