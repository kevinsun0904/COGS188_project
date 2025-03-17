import math
import chess
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import time
from ChessEnv import ChessEnv

class ChessCNN:
    """Placeholder for your CNN implementation"""
    def __init__(self):
        # Your CNN architecture definition here
        pass
        
    def predict(self, board_tensor):
        # Your forward pass implementation
        pass

def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Convert chess board to tensor representation for CNN input.
    This is a placeholder - you'll need to implement based on your CNN's requirements.
    """
    # Implementation depends on your CNN's expected input format
    # Example implementation shown in previous response
    pass

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
        
        # Exploitation term
        wins = self.results[chess.WHITE] if self.board.turn == chess.WHITE else self.results[chess.BLACK]
        exploitation = wins / self.visits
        
        # Exploration term
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration
    
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
    def __init__(self, env: ChessEnv, cnn_model, exploration_weight=1.41, simulation_limit=100, time_limit=5.0):
        self.env = env
        self.cnn_model = cnn_model
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.root = None
    
    def get_policy_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Get move probabilities from CNN policy network.
        
        Args:
            board: Current chess board
            
        Returns:
            Dictionary mapping legal moves to their prior probabilities
        """
        # Encode board for CNN input
        board_tensor = encode_board(board)
        
        # Get policy predictions from CNN
        with torch.no_grad():
            logits = self.cnn_model.predict(board_tensor)
        
        # Map outputs to legal moves
        legal_moves = list(board.legal_moves)
        move_probs = {}
        
        # TODO: May have to change this depending on CNN implementations
        for move in legal_moves:
            # Get probability for this move from CNN output
            move_index = self._get_move_index(move)  # Helper to map move to index in CNN output
            move_probs[move] = logits[move_index].item()
        
        # Normalize probabilities
        if move_probs:
            total = sum(move_probs.values())
            if total > 0:
                move_probs = {m: p/total for m, p in move_probs.items()}
        
        return move_probs
    
    def _get_move_index(self, move):
        """
        Helper method to map a chess move to its index in the CNN output.
        """
        # TODO: implement after CNN is complete
        pass
    
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
        
        while not sim_board.is_game_over() and move_count < move_limit:
            # Get move probabilities from policy network
            board_tensor = encode_board(sim_board)
            with torch.no_grad():
                logits = self.cnn_model.predict(board_tensor)
            
            # Select moves based on policy probabilities
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
                
            # Get probabilities for legal moves only
            legal_move_indices = [self._get_move_index(move) for move in legal_moves]
            legal_move_probs = logits[legal_move_indices].softmax(dim=0).numpy()
            
            # Select move (you can use temperature parameter to control exploration)
            selected_idx = np.random.choice(len(legal_moves), p=legal_move_probs)
            move = legal_moves[selected_idx]
                
            # Apply the selected move
            sim_board.push(move)
            move_count += 1
        
        # Process game result as in your original code
        if sim_board.is_game_over():
            outcome = sim_board.outcome()
            if outcome.winner == chess.WHITE:
                return {chess.WHITE: 1.0, chess.BLACK: 0.0, 'draw': 0.0}
            elif outcome.winner == chess.BLACK:
                return {chess.WHITE: 0.0, chess.BLACK: 1.0, 'draw': 0.0}
            else:  # Draw
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
        else:
            # If move limit was reached, use the Stockfish evaluation
            original_fen = self.env.board.fen()
            self.env.board = sim_board.copy()
            self.env.stockfish.set_fen_position(sim_board.fen())
            evaluation = self.env.evaluate_position()
            
            # Restore original board
            self.env.board = chess.Board(original_fen)
            self.env.stockfish.set_fen_position(original_fen)
            
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
