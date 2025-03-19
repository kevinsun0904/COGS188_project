import pygame
import chess
import chess.svg
import os
import numpy as np
import time
import random
from pygame.locals import *
from PIL import Image
import numpy as np
import io
import cairosvg
from ChessEnv import ChessEnv
from models.chessModels import ChessCNN  # Assuming this is your CNN class
from models.chessModels import MCTSNode  # Assuming this is your MCTS node class
from models.chessModels import MCTSCNNAgent 
from models.chessModels import ChessMCTSCNN  # Adjust this import based on your actual file structure

class ChessGUI:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chess AI - Neural Network with MCTS")
        
        # Initialize chess board
        self.board = chess.Board()
        self.board_size = min(width - 200, height)  # Leave space for info panel
        
        # Initialize AI
        self.chess_env = ChessEnv(stockfish_path = "/Users/kaust/stockfish/stockfish-windows-x86-64-avx2.exe")
        
        # Try to use the actual CNN evaluator, fall back to placeholder if not available
        try:
            self.cnn_evaluator = ChessCNN()
        except Exception as e:
            print(f"Error initializing CNN: {e}")
            self.cnn_evaluator = ChessCNN()  # Placeholder
        
        self.mcts = MCTSCNNAgent(model_path="chess_cnn_model.h5", simulation_limit=5000, time_limit=60.0)
        
        # Game state
        self.player_color = chess.WHITE  # Human plays white by default
        self.selected_square = None
        self.game_over = False
        self.message = "Your move (White)"
        self.ai_thinking = False
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 18)
        self.button_font = pygame.font.SysFont('Arial', 16)
        
        # Create buttons
        self.buttons = [
            {"rect": pygame.Rect(width - 190, 50, 180, 40), "text": "New Game", "action": self.new_game},
            {"rect": pygame.Rect(width - 190, 100, 180, 40), "text": "Flip Board", "action": self.flip_board},
            {"rect": pygame.Rect(width - 190, 150, 180, 40), "text": "AI Move", "action": self.request_ai_move},
            {"rect": pygame.Rect(width - 190, 200, 180, 40), "text": "Undo Move", "action": self.undo_move},
        ]
        
        # Load piece images
        self.piece_images = self.load_piece_images()
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Legal moves for highlighting
        self.legal_moves_for_selected = []
    
    def load_piece_images(self):
        """Load chess piece images from SVG"""
        piece_images = {}
        piece_size = self.board_size // 8 - 10  # Slightly smaller than square size
        
        pieces = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p']
        
        try:
            # First check if cairosvg is available
            import cairosvg
            
            for piece_symbol in pieces:
                # Generate SVG for the piece
                piece = chess.Piece.from_symbol(piece_symbol)
                svg_string = chess.svg.piece(piece, size=piece_size)
                
                # Convert SVG to PNG using cairosvg
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
                
                # Create a surface from the PNG data
                png_surface = pygame.image.load(io.BytesIO(png_data))
                
                # Store the image
                piece_images[piece_symbol] = png_surface
                
        except ImportError:
            # If cairosvg is not available, use a fallback method with colored rectangles
            print("Warning: cairosvg not available. Using placeholder piece images.")
            colors = {
                'K': (255, 255, 255), 'Q': (240, 240, 240), 'R': (220, 220, 220),
                'B': (200, 200, 200), 'N': (180, 180, 180), 'P': (160, 160, 160),
                'k': (0, 0, 0), 'q': (20, 20, 20), 'r': (40, 40, 40),
                'b': (60, 60, 60), 'n': (80, 80, 80), 'p': (100, 100, 100)
            }
            
            for piece_symbol, color in colors.items():
                # Create a surface for the piece
                surface = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                pygame.draw.rect(surface, color, (0, 0, piece_size, piece_size), border_radius=piece_size//4)
                
                # Add the piece letter
                text = self.font.render(piece_symbol, True, (255, 0, 0) if piece_symbol.isupper() else (0, 0, 255))
                text_rect = text.get_rect(center=(piece_size//2, piece_size//2))
                surface.blit(text, text_rect)
                
                # Store the image
                piece_images[piece_symbol] = surface
        
        return piece_images
    
    def draw_board(self):
        """Draw the chess board"""
        # Draw the board
        square_size = self.board_size // 8
        for row in range(8):
            for col in range(8):
                # Determine square color
                color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    pygame.Rect(col * square_size, row * square_size, square_size, square_size)
                )
                
                # Draw square coordinates
                if row == 7:
                    text = self.font.render(chess.FILE_NAMES[col], True, (0, 0, 0) if color == (240, 217, 181) else (255, 255, 255))
                    self.screen.blit(text, (col * square_size + 5, row * square_size + square_size - 20))
                if col == 0:
                    text = self.font.render(chess.RANK_NAMES[7-row], True, (0, 0, 0) if color == (240, 217, 181) else (255, 255, 255))
                    self.screen.blit(text, (5, row * square_size + 5))
        
        # Highlight legal moves for selected piece
        for move in self.legal_moves_for_selected:
            to_square = move.to_square
            row, col = 7 - (to_square // 8), to_square % 8
            # Draw circle in the center of the square
            pygame.draw.circle(
                self.screen,
                (100, 255, 100, 128),  # Green highlight
                (col * square_size + square_size // 2, row * square_size + square_size // 2),
                square_size // 6  # Circle radius
            )
        
        # Highlight selected square
        if self.selected_square is not None:
            row, col = 7 - (self.selected_square // 8), self.selected_square % 8
            pygame.draw.rect(
                self.screen,
                (255, 255, 0),  # Yellow highlight
                pygame.Rect(col * square_size, row * square_size, square_size, square_size),
                3  # Border width
            )
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Get piece position
                row, col = 7 - (square // 8), square % 8
                x = col * square_size + square_size // 2
                y = row * square_size + square_size // 2
                
                # Get piece image
                piece_img = self.piece_images.get(piece.symbol())
                if piece_img:
                    # Center the piece in the square
                    piece_rect = piece_img.get_rect(center=(x, y))
                    self.screen.blit(piece_img, piece_rect)
    
    def draw_info_panel(self):
        """Draw information panel"""
        # Draw panel background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            pygame.Rect(self.board_size, 0, self.width - self.board_size, self.height)
        )
        
        # Draw game status
        status_text = self.font.render(self.message, True, (0, 0, 0))
        self.screen.blit(status_text, (self.board_size + 10, 10))
        
        # Draw turn indicator
        turn_text = self.font.render(
            f"{'White' if self.board.turn == chess.WHITE else 'Black'} to move", 
            True, 
            (0, 0, 0)
        )
        self.screen.blit(turn_text, (self.board_size + 10, 30))
        
        # Draw buttons
        for button in self.buttons:
            # Draw button
            pygame.draw.rect(
                self.screen,
                (200, 200, 200),
                button["rect"],
                0,
                5  # Border radius
            )
            
            # Draw button text
            text = self.button_font.render(button["text"], True, (0, 0, 0))
            text_rect = text.get_rect(center=button["rect"].center)
            self.screen.blit(text, text_rect)
        
        # Draw move history
        history_text = self.font.render("Move History:", True, (0, 0, 0))
        self.screen.blit(history_text, (self.board_size + 10, 250))
        
        # Display last 10 moves
        moves = list(self.board.move_stack)[-10:]
        for i, move in enumerate(moves):
            move_text = self.font.render(f"{len(self.board.move_stack) - len(moves) + i + 1}. {move}", True, (0, 0, 0))
            self.screen.blit(move_text, (self.board_size + 10, 280 + i * 20))
        
        # If game is over, display result
        if self.game_over:
            result_text = self.font.render(f"Game Over: {self.get_game_result()}", True, (255, 0, 0))
            self.screen.blit(result_text, (self.board_size + 10, self.height - 50))

        # Add these methods to your ChessGUI class

    def update_legal_moves(self):
        """Update the list of legal moves for the selected square"""
        self.legal_moves_for_selected = []
        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    self.legal_moves_for_selected.append(move)

    def handle_click(self, pos):
        """Handle mouse click events"""
        x, y = pos
        
        # Check if click is on the board
        if x < self.board_size and y < self.board_size:
            # Convert click position to square
            square_size = self.board_size // 8
            col = x // square_size
            row = y // square_size
            square = chess.square(col, 7 - row)
            
            # Handle square selection
            if self.selected_square is None:
                # Select the square if it has a piece of the player's color
                piece = self.board.piece_at(square)
                if piece and piece.color == self.player_color and self.board.turn == self.player_color:
                    self.selected_square = square
                    self.update_legal_moves()
            else:
                # Check if the clicked square is a valid destination for the selected piece
                move = None
                for legal_move in self.legal_moves_for_selected:
                    if legal_move.to_square == square:
                        # Check for promotion
                        if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                            ((self.player_color == chess.WHITE and square // 8 == 7) or 
                            (self.player_color == chess.BLACK and square // 8 == 0))):
                            move = chess.Move(self.selected_square, square, chess.QUEEN)  # Default to queen promotion
                        else:
                            move = legal_move
                        break
                
                if move:
                    self.make_move(move)
                    self.selected_square = None
                    self.legal_moves_for_selected = []
                    
                    # If game not over, let AI make a move
                    if not self.game_over and self.board.turn != self.player_color:
                        self.ai_thinking = True
                        self.message = "AI is thinking..."
                else:
                    # If the move is not legal, check if the new square has a piece of the player's color
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.player_color:
                        self.selected_square = square
                        self.update_legal_moves()
                    else:
                        self.selected_square = None
                        self.legal_moves_for_selected = []
        
        # Check if click is on a button
        else:
            for button in self.buttons:
                if button["rect"].collidepoint(pos):
                    button["action"]()

    def make_move(self, move):
        """Make a move on the board"""
        self.board.push(move)
        
        # Check if the game is over
        if self.board.is_game_over():
            self.game_over = True
            self.message = f"Game over: {self.get_game_result()}"

    def get_game_result(self):
        """Get the game result as a string"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"{winner} wins by checkmate"
        elif self.board.is_stalemate():
            return "Draw by stalemate"
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material"
        elif self.board.is_fifty_moves():
            return "Draw by fifty-move rule"
        elif self.board.is_repetition():
            return "Draw by repetition"
        else:
            return "Game over"

    def new_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.game_over = False
        self.message = "Your move (White)"
        self.ai_thinking = False

    def flip_board(self):
        """Switch sides (player plays as black)"""
        self.player_color = chess.BLACK if self.player_color == chess.WHITE else chess.WHITE
        self.message = f"You are playing as {'White' if self.player_color == chess.WHITE else 'Black'}"
        
        # If AI's turn, make it move
        if not self.game_over and self.board.turn != self.player_color:
            self.ai_thinking = True
            self.message = "AI is thinking..."

    def request_ai_move(self):
        """Request AI to make a move"""
        if not self.game_over and not self.ai_thinking:
            self.ai_thinking = True
            self.message = "AI is thinking..."

    def undo_move(self):
        """Undo the last move"""
        if len(self.board.move_stack) >= 2:
            self.board.pop()  # Undo AI's move
            self.board.pop()  # Undo player's move
            self.game_over = False
            self.message = "Moves undone"
        elif len(self.board.move_stack) == 1:
            self.board.pop()  # Undo just one move
            self.game_over = False
            self.message = "Move undone"
        
        self.selected_square = None
        self.legal_moves_for_selected = []

    def ai_move(self):
        """Make AI move"""
        if not self.game_over and self.board.turn != self.player_color:
            try:
                # Get the best move from MCTS
                best_move = self.mcts.select_move(self.board)
                
                if best_move:
                    # Make the move
                    self.make_move(best_move)
                    self.message = f"AI moved: {best_move}"
                else:
                    self.message = "AI couldn't find a move"
            except Exception as e:
                self.message = f"AI error: {str(e)}"
                print(f"AI error: {str(e)}")
        
        self.ai_thinking = False

    def run(self):
        """Main game loop"""
        running = True
        ai_thinking_start = 0
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.ai_thinking:
                        self.handle_click(event.pos)
            
            # Clear the screen
            self.screen.fill((255, 255, 255))
            
            # Draw the board and pieces
            self.draw_board()
            
            # Draw info panel
            self.draw_info_panel()
            
            # AI thinking
            if self.ai_thinking:
                if ai_thinking_start == 0:
                    ai_thinking_start = pygame.time.get_ticks()
                    # Display thinking message
                    thinking_text = self.font.render("AI is thinking...", True, (0, 0, 0))
                    self.screen.blit(thinking_text, (self.board_size + 10, self.height - 30))
                
                # Give AI some time to visually show it's thinking (for UX)
                if pygame.time.get_ticks() - ai_thinking_start > 500:
                    self.ai_move()
                    ai_thinking_start = 0
            
            # If it's the AI's turn and not thinking, start thinking
            if not self.game_over and not self.ai_thinking and self.board.turn != self.player_color:
                self.ai_thinking = True
                self.message = "AI is thinking..."
            
            # Update the display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(30)
        
        # Quit pygame
        pygame.quit()

# Main function
if __name__ == "__main__":
    # Create and run the GUI
    gui = ChessGUI()
    gui.run()