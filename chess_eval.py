import chess
import chess.engine
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import json
from pathlib import Path

# Import your AI implementation
from ChessEnv import ChessEnv
from monte_carlo_with_cnn import ChessCNN
from monte_carlo_with_cnn import ChessMCTS  # Replace with actual import

class EloEvaluator:
    """Class to evaluate the ELO rating of a chess AI using Stockfish."""
    
    def __init__(self, stockfish_path, cnn_model_path=None, time_limit=2.0, simulation_limit=100):
        """
        Initialize the ELO evaluator.
        
        Args:
            stockfish_path: Path to the Stockfish executable
            cnn_model_path: Path to your CNN model
            time_limit: Time limit for MCTS search in seconds
            simulation_limit: Maximum number of simulations for MCTS
        """
        self.stockfish_path = stockfish_path
        self.cnn_model_path = cnn_model_path
        self.time_limit = time_limit
        self.simulation_limit = simulation_limit
        self.results = {}
        
        # Check if Stockfish exists
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish executable not found at {stockfish_path}")
            
        # Initialize your AI
        self.env = ChessEnv(stockfish_path = "/Users/kaust/stockfish/stockfish-windows-x86-64-avx2.exe")
        self.cnn = ChessCNN(model_path=cnn_model_path)
        self.mcts_ai = ChessMCTS(
            env=self.env,
            evaluator=self.cnn,
            time_limit=time_limit,
            simulation_limit=simulation_limit
        )
    
    def play_game(self, ai_color, stockfish_elo):
        """
        Play a game between your AI and Stockfish.
        
        Args:
            ai_color: Color for your AI (chess.WHITE or chess.BLACK)
            stockfish_elo: ELO rating to set for Stockfish
            
        Returns:
            Result: 1 if AI wins, 0.5 for draw, 0 if Stockfish wins
        """
        board = chess.Board()
        
        # Start the Stockfish engine
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            # Set Stockfish's ELO rating
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
            
            # Set time control for Stockfish
            limit = chess.engine.Limit(time=self.time_limit)
            
            # Play until the game is over
            while not board.is_game_over():
                if board.turn == ai_color:
                    # Your AI's turn
                    start_time = time.time()
                    move = self.mcts_ai.search(board)
                    end_time = time.time()
                    print(f"AI move: {move} (took {end_time - start_time:.2f}s)")
                else:
                    # Stockfish's turn
                    result = engine.play(board, limit)
                    move = result.move
                    print(f"Stockfish move: {move}")
                
                # Make the move
                board.push(move)
                print(board)
                print("--------------------")
            
            # Determine the result
            outcome = board.outcome()
            if outcome.winner == ai_color:
                return 1.0  # AI wins
            elif outcome.winner is None:
                return 0.5  # Draw
            else:
                return 0.0  # Stockfish wins
    
    def evaluate_elo(self, elo_range=(1000, 3000), step=200, games_per_level=10):
        """
        Evaluate the ELO rating of your AI by playing against Stockfish at different levels.
        
        Args:
            elo_range: Tuple of (min_elo, max_elo) to test against
            step: Step size between ELO levels
            games_per_level: Number of games to play at each ELO level
            
        Returns:
            Estimated ELO rating
        """
        # Create a range of ELO levels to test
        elo_levels = list(range(elo_range[0], elo_range[1] + step, step))
        
        # Store results for each level
        self.results = {elo: {"wins": 0, "draws": 0, "losses": 0, "score": 0, "games": 0} for elo in elo_levels}
        
        # Play games at each level
        for elo in elo_levels:
            print(f"\nTesting against Stockfish ELO {elo}")
            
            for game in tqdm(range(games_per_level)):
                # Alternate colors
                ai_color = chess.WHITE if game % 2 == 0 else chess.BLACK
                result = self.play_game(ai_color, elo)
                
                # Update results
                self.results[elo]["games"] += 1
                if result == 1.0:
                    self.results[elo]["wins"] += 1
                elif result == 0.5:
                    self.results[elo]["draws"] += 1
                else:
                    self.results[elo]["losses"] += 1
                
                # Update score
                self.results[elo]["score"] += result
            
            # Calculate win rate
            games = self.results[elo]["games"]
            score = self.results[elo]["score"]
            win_rate = score / games
            self.results[elo]["win_rate"] = win_rate
            
            print(f"Win rate against ELO {elo}: {win_rate:.2f}")
            
            # Early stopping if we find a good bracket
            if win_rate <= 0.25:
                print(f"Stopping early as win rate is low against ELO {elo}")
                break
        
        # Calculate approximate ELO
        estimated_elo = self.estimate_elo()
        self.plot_results()
        
        return estimated_elo
    
    def estimate_elo(self):
        """
        Estimate the ELO rating of your AI based on the results.
        
        Returns:
            Estimated ELO rating
        """
        # Get ELO levels and win rates
        elo_levels = []
        win_rates = []
        
        for elo, data in self.results.items():
            if data["games"] > 0:
                elo_levels.append(elo)
                win_rates.append(data["win_rate"])
        
        # Find the ELO level where win rate is closest to 0.5
        win_rate_diffs = [abs(rate - 0.5) for rate in win_rates]
        if win_rate_diffs:
            best_idx = np.argmin(win_rate_diffs)
            closest_elo = elo_levels[best_idx]
            
            # Linear interpolation for more precise estimate
            if best_idx > 0 and best_idx < len(elo_levels) - 1:
                elo_lower = elo_levels[best_idx - 1]
                elo_higher = elo_levels[best_idx + 1]
                rate_lower = win_rates[best_idx - 1]
                rate_higher = win_rates[best_idx + 1]
                
                if rate_lower != rate_higher:
                    # Interpolate to find ELO where win rate would be 0.5
                    estimated_elo = elo_lower + (0.5 - rate_lower) * (elo_higher - elo_lower) / (rate_higher - rate_lower)
                else:
                    estimated_elo = closest_elo
            else:
                estimated_elo = closest_elo
                
            return estimated_elo
        else:
            return None
    
    def plot_results(self):
        """Plot the results of the ELO evaluation."""
        elo_levels = []
        win_rates = []
        
        for elo, data in sorted(self.results.items()):
            if data["games"] > 0:
                elo_levels.append(elo)
                win_rates.append(data["win_rate"])
        
        plt.figure(figsize=(10, 6))
        plt.plot(elo_levels, win_rates, 'o-')
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% win rate')
        plt.xlabel('Stockfish ELO')
        plt.ylabel('Win Rate')
        plt.title('AI Win Rate vs Stockfish ELO')
        plt.grid(True)
        plt.legend()
        plt.savefig('elo_evaluation_results.png')
        plt.close()
    
    def save_results(self, filename='elo_evaluation_results.json'):
        """Save the results to a JSON file."""
        # Add estimated ELO to results
        results_with_elo = {
            "results": self.results,
            "estimated_elo": self.estimate_elo(),
            "evaluation_parameters": {
                "time_limit": self.time_limit,
                "simulation_limit": self.simulation_limit
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_with_elo, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the ELO rating of a chess AI using Stockfish")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    parser.add_argument("--model", default=None, help="Path to CNN model")
    parser.add_argument("--min_elo", type=int, default=1000, help="Minimum Stockfish ELO to test against")
    parser.add_argument("--max_elo", type=int, default=2000, help="Maximum Stockfish ELO to test against")
    parser.add_argument("--step", type=int, default=200, help="Step size between ELO levels")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play at each ELO level")
    parser.add_argument("--time", type=float, default=2.0, help="Time limit for MCTS search in seconds")
    parser.add_argument("--simulations", type=int, default=100, help="Maximum number of simulations for MCTS")
    
    args = parser.parse_args()
    
    # Create the evaluator
    evaluator = EloEvaluator(
        stockfish_path=args.stockfish,
        cnn_model_path=args.model,
        time_limit=args.time,
        simulation_limit=args.simulations
    )
    
    # Evaluate ELO
    estimated_elo = evaluator.evaluate_elo(
        elo_range=(args.min_elo, args.max_elo),
        step=args.step,
        games_per_level=args.games
    )
    
    # Print results
    print(f"\nEstimated ELO rating: {estimated_elo:.0f}")
    
    # Save results
    evaluator.save_results()

if __name__ == "__main__":
    main()