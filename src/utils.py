import chess


def get_time_pressure_level(time_remaining: float) -> str:
    """Convert numeric time to pressure level."""
    if time_remaining > 30:
        return "relaxed"
    elif time_remaining > 10:
        return "moderate"
    elif time_remaining > 5:
        return "pressure"
    else:
        return "scramble"

def generate_all_possible_uci_moves():
        board = chess.Board()
        moves = set()
        squares = list(chess.SQUARES)
        promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        for from_square in squares:
            for to_square in squares:
                # Normal move
                move = chess.Move(from_square, to_square)
                moves.add(move.uci())
                # Promotion moves (only if move is from 7th rank for white or 2nd rank for black)
                if chess.square_rank(from_square) == 6:  # White 7th rank
                    for promo in promotion_pieces:
                        promo_move = chess.Move(from_square, to_square, promotion=promo)
                        moves.add(promo_move.uci())
                if chess.square_rank(from_square) == 1:  # Black 2nd rank
                    for promo in promotion_pieces:
                        promo_move = chess.Move(from_square, to_square, promotion=promo)
                        moves.add(promo_move.uci())
        return sorted(moves)