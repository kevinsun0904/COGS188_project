import chess
import chess.svg
import chess.engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
from tqdm import tqdm
import sys
sys.path.append('.')
from ChessEnv import ChessEnv
from models.chessModels import ChessCNN  
from models.chessModels import MCTSNode  
from models.chessModels import MCTSCNNAgent 
from models.chessModels import ChessMCTSCNN 

class ChessEvaluator:
    def __init__(self, stockfish_path, low_elo=1400, high_elo=3640, num_games=100, start_row=12000):
        """
        Initialize the chess evaluator with two Stockfish engines at different ELO ratings.
        
        Args:
            stockfish_path: Path to Stockfish binary
            low_elo: ELO rating for the "true move" Stockfish (default: 1400)
            high_elo: ELO rating for the "best move" Stockfish (default: 3640)
            num_games: Number of games to evaluate
            start_row: Starting row in the CSV file
        """
        self.stockfish_path = stockfish_path
        self.low_elo = low_elo
        self.high_elo = high_elo
        self.num_games = num_games
        self.start_row = start_row
        
        # Metrics storage
        self.results = {
            'game_id': [],
            'move_number': [],
            'true_move': [],  # 1400 ELO Stockfish
            'best_move': [],  # 3640 ELO Stockfish
            'pred_move': [],  #  method's move
            'pred_equals_true': [],  # Prediction matches 1400 ELO
            'pred_equals_best': [],  # Prediction matches 3640 ELO
            'true_equals_best': [],  # 1400 ELO matches 3640 ELO
            'true_positive': [],     # pred == true == best
            'false_positive': [],    # pred == true != best
            'false_negative': []     # pred == best != true
        }
        
        # Initialize the engines
        self.true_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.true_engine.configure({"UCI_Elo": low_elo})
        
        self.best_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.best_engine.configure({"UCI_Elo": high_elo})
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'true_engine'):
            self.true_engine.quit()
        if hasattr(self, 'best_engine'):
            self.best_engine.quit()
    
    def my_method(self, board):
        """
         chess move prediction method.
        
        Args:
            board: A chess.Board object representing the current position
            
        Returns:
            A chess.Move object or None if no legal moves
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0
        
        self.mcts_ai = MCTSCNNAgent(model_path="chess_cnn_model.h5", simulation_limit=5000, time_limit=60.0)
        move = self.mcts_ai.select_move(board)
        
        return move
    
    def get_engine_best_move(self, board, engine, time_limit=0.1):
        """
        Get an engine's best move for the given position.
        
        Args:
            board: A chess.Board object
            engine: A chess engine instance
            time_limit: Time limit for analysis in seconds
            
        Returns:
            A chess.Move object or None on error
        """
        try:
            result = engine.play(
                board,
                chess.engine.Limit(time=time_limit)
            )
            return result.move
        except Exception as e:
            print(f"Engine error: {e}")
            return None
    
    def load_games(self, csv_path):
        """
        Load chess games from CSV file.
        
        Args:
            csv_path: Path to the CSV file containing games
            
        Returns:
            List of game move strings
        """
        df = pd.read_csv(csv_path)
        return df.iloc[self.start_row:self.start_row + self.num_games]['moves'].tolist()
    
    def evaluate_game(self, game_id, move_string):
        """
        Evaluate a single game.
        
        Args:
            game_id: ID for the game
            move_string: String of moves in the game (e.g. "e4 e5 Nf3 Nc6")
        """
        moves = move_string.strip().split()
        board = chess.Board()
        
        # Make first move from the game
        if moves:
            try:
                first_move = chess.Move.from_uci(moves[0])
                board.push(first_move)
            except ValueError:
                try:
                    # Try as algebraic notation
                    first_move = board.parse_san(moves[0])
                    board.push(first_move)
                except ValueError:
                    print(f"Could not parse first move: {moves[0]}")
                    return
        
        # Start from the second move
        for move_idx, move_str in enumerate(moves[1:], start=1):
            # Skip if game is over
            if board.is_game_over():
                break
            
            svg_content = chess.svg.board(board, size=350)
    
            with open("chess_position_eval.svg", 'w') as f:
                f.write(svg_content)
                
            # Get 1400 ELO Stockfish move (true move)
            true_move = self.get_engine_best_move(board, self.true_engine)
            
            # Get 3640 ELO Stockfish move (best move)
            best_move = self.get_engine_best_move(board, self.best_engine)
            
            # Get my method's move
            pred_move = self.my_method(board)
            
            # Record results
            pred_equals_true = pred_move == true_move if pred_move and true_move else False
            pred_equals_best = pred_move == best_move if pred_move and best_move else False
            true_equals_best = true_move == best_move if true_move and best_move else False
            
            # Calculate the special cases
            true_positive = pred_equals_true and true_equals_best  # pred == true == best
            false_positive = pred_equals_true and not true_equals_best  # pred == true != best
            false_negative = pred_equals_best and not true_equals_best  # pred == best != true
            
            self.results['game_id'].append(game_id)
            self.results['move_number'].append(move_idx)
            self.results['true_move'].append(str(true_move) if true_move else None)
            self.results['best_move'].append(str(best_move) if best_move else None)
            self.results['pred_move'].append(str(pred_move) if pred_move else None)
            self.results['pred_equals_true'].append(pred_equals_true)
            self.results['pred_equals_best'].append(pred_equals_best)
            self.results['true_equals_best'].append(true_equals_best)
            self.results['true_positive'].append(true_positive)
            self.results['false_positive'].append(false_positive)
            self.results['false_negative'].append(false_negative)
            
            # Make the actual move from the game
            try:
                actual_move = chess.Move.from_uci(move_str)
                board.push(actual_move)
            except ValueError:
                try:
                    # Try as algebraic notation
                    actual_move = board.parse_san(move_str)
                    board.push(actual_move)
                except ValueError:
                    print(f"Could not parse move: {move_str}")
                    break
    
    def run_evaluation(self, csv_path):
        """
        Run the evaluation on all games.
        
        Args:
            csv_path: Path to the CSV file containing games
        """
        games = self.load_games(csv_path)
        
        for game_id, move_string in tqdm(enumerate(games), total=len(games)):
            self.evaluate_game(game_id, move_string)
            
        return pd.DataFrame(self.results)
    
    def calculate_metrics(self, results_df):
        """
        Calculate evaluation metrics from results using the specified definitions.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary of metrics
        """
        # Calculate accuracy as matches/total moves
        total_moves = len(results_df)
        matches = results_df['pred_equals_true'].sum()
        accuracy = matches / total_moves if total_moves > 0 else 0
        
        # Get counts for custom precision, recall, and F1 calculations
        true_positives = results_df['true_positive'].sum()
        false_positives = results_df['false_positive'].sum()
        false_negatives = results_df['false_negative'].sum()
        
        # Calculate precision, recall, and F1 using the specified definitions
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional useful stats
        pred_equals_best_count = results_df['pred_equals_best'].sum()
        true_equals_best_count = results_df['true_equals_best'].sum()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_moves': total_moves,
            'matching_true_moves': matches,
            'matching_best_moves': pred_equals_best_count,
            'true_equals_best_moves': true_equals_best_count,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def plot_metrics(self, metrics, output_dir='./plots'):
        """
        Generate and save plots for metrics.
        
        Args:
            metrics: Dictionary of metrics
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Bar chart for accuracy, precision, recall, f1
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics[m] for m in metrics_to_plot]
        
        plt.bar(metrics_to_plot, values, color=['#3498db', '#2ecc71', '#9b59b6', '#f39c12'])
        plt.ylim(0, 1)
        plt.title(f'Model Performance Metrics (True: {self.low_elo} ELO, Best: {self.high_elo} ELO)')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_metrics.png")
        
        # Confusion matrix-style visualization
        plt.figure(figsize=(10, 8))
        
        # Data for the confusion plot
        TP = metrics['true_positives']
        FP = metrics['false_positives']
        FN = metrics['false_negatives']
        
        # Calculate TN as the remaining moves
        # TN = When pred != true and true != best
        # Total - (TP + FP + FN) won't be correct here since we have other cases
        # We need to calculate this from the results DataFrame
        results_df = pd.DataFrame(self.results)
        TN = len(results_df) - (results_df['pred_equals_true'] | results_df['true_equals_best']).sum()
        
        # Create the confusion matrix visualization
        confusion_data = np.array([
            [TP, FP],
            [FN, TN]
        ])
        
        plt.imshow(confusion_data, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Custom Confusion Matrix')
        plt.colorbar()
        
        classes = ['Positive', 'Negative']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = confusion_data.max() / 2
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(confusion_data[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_data[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        
        # Move comparison visualization
        plt.figure(figsize=(10, 6))
        
        # Data for move comparison
        move_data = [
            metrics['matching_true_moves'],  # Matches with 1400 ELO
            metrics['matching_best_moves'],  # Matches with 3640 ELO
            metrics['true_equals_best_moves']  # 1400 ELO matches 3640 ELO
        ]
        
        move_labels = [
            f"Pred = True ({self.low_elo} ELO)",
            f"Pred = Best ({self.high_elo} ELO)",
            "True = Best"
        ]
        
        # Calculate percentages
        move_percentages = [count / metrics['total_moves'] * 100 for count in move_data]
        
        # Create bar chart with percentages
        bars = plt.bar(move_labels, move_percentages, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        plt.ylabel('Percentage of Moves (%)')
        plt.title('Move Comparison')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, move_percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{percentage:.1f}%', ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/move_comparison.png")
        
        # Venn-like diagram for move overlap
        plt.figure(figsize=(8, 8))
        
        # Calculate the proportions for our Venn-like visualization
        # Note: This is not a true Venn diagram but a simplified representation
        pred_true = metrics['matching_true_moves'] / metrics['total_moves']
        pred_best = metrics['matching_best_moves'] / metrics['total_moves']
        true_best = metrics['true_equals_best_moves'] / metrics['total_moves']
        tp_prop = metrics['true_positives'] / metrics['total_moves']
        
        # Create three circles to represent the overlaps
        circle1 = plt.Circle((0.3, 0.6), 0.3, alpha=0.5, color='#3498db', label='Pred = True')
        circle2 = plt.Circle((0.7, 0.6), 0.3, alpha=0.5, color='#e74c3c', label='Pred = Best')
        circle3 = plt.Circle((0.5, 0.3), 0.3, alpha=0.5, color='#2ecc71', label='True = Best')
        
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        
        # Annotate the diagram
        plt.text(0.3, 0.6, f"Pred = True\n{pred_true:.1%}", ha='center', va='center')
        plt.text(0.7, 0.6, f"Pred = Best\n{pred_best:.1%}", ha='center', va='center')
        plt.text(0.5, 0.3, f"True = Best\n{true_best:.1%}", ha='center', va='center')
        plt.text(0.5, 0.5, f"TP\n{tp_prop:.1%}", ha='center', va='center', weight='bold')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Move Overlap Visualization')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=15, label=f'Pred = True ({self.low_elo} ELO)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=15, label=f'Pred = Best ({self.high_elo} ELO)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=15, label=f'True = Best')
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/move_overlap.png")
        
        return

def main():
    # Configuration
    stockfish_path = "C:/Users/kaust/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update with actual path
    csv_path = "databases/Lichess(training)/games.csv"        # Update with actual path
    low_elo = 1400  # "true move" ELO
    high_elo = 3140  # "best move" ELO
    num_games = 100   # Adjust based on  needs
    start_row = 12000
    
    print(f"Starting evaluation of {num_games} games with Stockfish at ELO {low_elo} (true) and {high_elo} (best)")
    
    # Initialize evaluator
    evaluator = ChessEvaluator(
        stockfish_path=stockfish_path,
        low_elo=low_elo,
        high_elo=high_elo,
        num_games=num_games,
        start_row=start_row
    )
    
    # Run evaluation
    start_time = time.time()
    results_df = evaluator.run_evaluation(csv_path)
    end_time = time.time()
    
    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results_df)
    
    # Plot metrics
    evaluator.plot_metrics(metrics)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total games processed: {num_games}")
    print(f"Total moves analyzed: {metrics['total_moves']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")
    
    print("Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f} (matches with {low_elo} ELO / total moves)")
    print(f"Precision: {metrics['precision']:.4f} (TP / (TP + FP))")
    print(f"Recall: {metrics['recall']:.4f} (TP / (TP + FN))")
    print(f"F1-Score: {metrics['f1_score']:.4f} (2 * Precision * Recall / (Precision + Recall))")
    
    print("\nMove Statistics:")
    print(f"Matches with {low_elo} ELO (Accuracy): {metrics['matching_true_moves']} ({metrics['matching_true_moves']/metrics['total_moves']*100:.2f}%)")
    print(f"Matches with {high_elo} ELO: {metrics['matching_best_moves']} ({metrics['matching_best_moves']/metrics['total_moves']*100:.2f}%)")
    print(f"{low_elo} ELO matches {high_elo} ELO: {metrics['true_equals_best_moves']} ({metrics['true_equals_best_moves']/metrics['total_moves']*100:.2f}%)")
    
    print("\nCustom Metric Components:")
    print(f"True Positives (pred = true = best): {metrics['true_positives']}")
    print(f"False Positives (pred = true ≠ best): {metrics['false_positives']}")
    print(f"False Negatives (pred = best ≠ true): {metrics['false_negatives']}")

if __name__ == "__main__":
    main()