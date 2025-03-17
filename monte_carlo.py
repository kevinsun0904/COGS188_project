import math
import chess
import numpy as np
import time
from typing import Dict, Optional
from ChessEnv import ChessEnv

class MCTSNode:
    """
    Node class for Monte Carlo Tree Search.
    Represents a board state and stores statistics for actions from this state.
    """
    def __init__(self, board: chess.Board, parent=None, move=None):
        """
        Initialize a new MCTS node.
        
        Args:
            board: Chess board representing this state
            parent: Parent node
            move: Move that led to this state from parent
        """
        self.board = board.copy()  # Create a copy of the board to avoid reference issues
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.visits = 0
        self.results = {chess.WHITE: 0, chess.BLACK: 0, 'draw': 0}
        self.untried_moves = list(board.legal_moves)

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
    """
    Monte Carlo Tree Search implementation for chess.
    """
    def __init__(self, env: ChessEnv, 
                 exploration_weight=1.41, simulation_limit=100, time_limit=5.0):
        """
        Initialize the MCTS solver.
        
        Args:
            env: Chess environment instance
            exploration_weight: UCB exploration parameter
            simulation_limit: Maximum number of simulations to run per search
            time_limit: Maximum time (in seconds) for the search
        """
        self.env = env
        self.exploration_weight = exploration_weight
        self.simulation_limit = simulation_limit
        self.time_limit = time_limit
        self.root = None
    
    def _policy_simulation(self, board: chess.Board) -> Dict[chess.Color, float]:
        """
        Run a simulation using the policy network (when available) or a fallback method.
        
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
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
                
            # Apply the selected move
            sim_board.push(move)
            move_count += 1
        
        # Get game result
        if sim_board.is_game_over():
            outcome = sim_board.outcome()
            if outcome.winner == chess.WHITE:
                return {chess.WHITE: 1.0, chess.BLACK: 0.0, 'draw': 0.0}
            elif outcome.winner == chess.BLACK:
                return {chess.WHITE: 0.0, chess.BLACK: 1.0, 'draw': 0.0}
            else:  # Draw
                return {chess.WHITE: 0.5, chess.BLACK: 0.5, 'draw': 1.0}
        else:
            # If move limit was reached, use the Stockfish evaluation to estimate result
            # Temporarily set the actual board in the environment to get evaluation
            original_fen = self.env.board.fen()
            self.env.board = sim_board.copy()
            self.env.stockfish.set_fen_position(sim_board.fen())
            evaluation = self.env.evaluate_position()
            
            # Restore the original environment board
            self.env.board = chess.Board(original_fen)
            self.env.stockfish.set_fen_position(original_fen)
            
            # Convert evaluation to result probabilities
            # Scale evaluation (-10 to 10) to probability (0 to 1)
            white_score = min(max((evaluation + 10) / 20, 0), 1)
            black_score = 1 - white_score
            
            return {chess.WHITE: white_score, chess.BLACK: black_score, 'draw': 0.0}
        
    
    def search(self, board: chess.Board) -> chess.Move:
        """
        Perform MCTS from the given board state to find the best move.
        
        Args:
            board: Current chess board state
            
        Returns:
            The best move according to MCTS
        """
        # Initialize the root node with the current board state
        self.root = MCTSNode(board)
        
        # Set up timing variables
        start_time = time.time()
        num_simulations = 0
        
        # Run simulations until we hit our limits
        while (num_simulations < self.simulation_limit and 
               time.time() - start_time < self.time_limit):
            # Phase 1: Selection - traverse tree to find a node to expand
            node = self.root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(self.exploration_weight)
            
            # Phase 2: Expansion - unless we're at a terminal state, expand tree by one node
            if not node.is_terminal():
                node = node.expand()
                if node is None:  # No untried moves
                    continue
            
            # Phase 3: Simulation - play out the game from the new node
            result = self._policy_simulation(node.board)
            
            # Phase 4: Backpropagation - update all nodes in the path
            while node is not None:
                node.update(result)
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
        print(f"MCTS completed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")
        print(f"Selected move: {best_move}, visits: {self.root.children[best_move].visits}")
        
        return best_move
    
    def get_move_probabilities(self) -> Dict[chess.Move, float]:
        """
        Get the probability distribution over moves from the current root.
        
        Returns:
            Dictionary mapping moves to their probabilities
        """
        if not self.root or not self.root.children:
            return {}
        
        total_visits = sum(child.visits for child in self.root.children.values())
        if total_visits == 0:
            return {move: 1.0 / len(self.root.children) for move in self.root.children}
        
        return {move: child.visits / total_visits 
                for move, child in self.root.children.items()}


# Example usage
def test_mcts():
    stockfish_path = "/Users/kevin/stockfish/stockfish-windows-x86-64-avx2.exe"  # Change this to your path to stockfish
    env = ChessEnv(stockfish_path=stockfish_path)
    env.reset()
    
    mcts = ChessMCTS(
        env=env,
        simulation_limit=200,  # Run up to 200 simulations
        time_limit=10.0  # Limit each move decision to 10 seconds
    )
    
    # Make some moves
    for _ in range(10):
        if env.board.is_game_over():
            break
            
        # Get the best move from MCTS
        best_move = mcts.search(env.board)
        
        if best_move:
            print(f"\nSelected move: {best_move}")
            # Get move probabilities for potential training data
            probs = mcts.get_move_probabilities()
            top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top move probabilities:")
            for move, prob in top_moves:
                print(f"{move}: {prob:.4f}")
                
            # Make the move
            next_state, reward, done, info = env.step(best_move)
            print("Board after move:")
            print(env.render())
            print(f"Evaluation: {info['evaluation']:.2f}")
            
            if done:
                print("Game over!")
                print(f"Result: {info['result']}")
                break
        else:
            print("No legal moves available.")
            break
    
    env.close()

# Uncomment to run test
test_mcts()