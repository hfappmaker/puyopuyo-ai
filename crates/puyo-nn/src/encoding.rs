//! NN encoding functions — re-exported from puyo-core for backward compatibility.
//!
//! The canonical implementations live in `puyo_core::puyo_game`.
//! This module re-exports them and provides fixed-size array wrappers
//! for existing callers that expect `[f32; N]` return types.

pub use puyo_core::puyo_game::{
    CONTEXT_TENSOR_SIZE, NUM_CHANNELS, PIECE_TENSOR_SIZE, TENSOR_SIZE,
};

use puyo_core::board::Board;
use puyo_core::piece::Piece;

/// Convert a Board to a fixed-size f32 array.
/// Wrapper around `puyo_core::puyo_game::board_to_tensor_data` for backward compatibility.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let vec = puyo_core::puyo_game::board_to_tensor_data(board);
    let mut arr = [0.0f32; TENSOR_SIZE];
    arr.copy_from_slice(&vec);
    arr
}

/// Convert three pieces to a fixed-size f32 array.
/// Wrapper around `puyo_core::puyo_game::pieces_to_tensor_data` for backward compatibility.
pub fn pieces_to_tensor_data(current: &Piece, next: &Piece, next_next: &Piece) -> [f32; PIECE_TENSOR_SIZE] {
    let vec = puyo_core::puyo_game::pieces_to_tensor_data(current, next, next_next);
    let mut arr = [0.0f32; PIECE_TENSOR_SIZE];
    arr.copy_from_slice(&vec);
    arr
}

/// Convert pieces to context tensor for FiLM conditioning.
/// Wrapper around `puyo_core::puyo_game::context_to_tensor_data` for backward compatibility.
pub fn context_to_tensor_data(
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
) -> [f32; CONTEXT_TENSOR_SIZE] {
    pieces_to_tensor_data(current, next, next_next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;
    use puyo_core::piece::Piece;

    #[test]
    fn test_empty_board_encoding() {
        let board = Board::new();
        let data = board_to_tensor_data(&board);
        for val in &data {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_tensor_size() {
        assert_eq!(TENSOR_SIZE, NUM_CHANNELS * puyo_core::board::ROWS * puyo_core::board::COLS);
    }

    #[test]
    fn test_color_channel_mapping() {
        let mut board = Board::new();
        let colors = PuyoColor::all_colors();
        for (col, &color) in colors.iter().enumerate() {
            board.drop_puyo(col, color);
        }
        let data = board_to_tensor_data(&board);

        for (ch, _) in colors.iter().enumerate() {
            assert_eq!(data[ch * puyo_core::board::ROWS * puyo_core::board::COLS + ch], 1.0);
        }
    }

    #[test]
    fn test_occupancy_channel() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(2 % puyo_core::board::COLS, PuyoColor::Blue);
        let data = board_to_tensor_data(&board);

        let occ = puyo_core::board::NUM_COLORS * puyo_core::board::ROWS * puyo_core::board::COLS;
        assert_eq!(data[occ], 1.0);
        assert_eq!(data[occ + 2 % puyo_core::board::COLS], 1.0);
        assert_eq!(data[occ + 1], 0.0);
    }

    #[test]
    fn test_adjacency_channel() {
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let data = board_to_tensor_data(&board);

        let adj = (puyo_core::board::NUM_COLORS + 1) * puyo_core::board::ROWS * puyo_core::board::COLS;
        assert!((data[adj] - 0.25).abs() < 1e-6);
        assert!((data[adj + puyo_core::board::COLS] - 0.5).abs() < 1e-6);
        assert!((data[adj + 2 * puyo_core::board::COLS] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_piece_tensor_size() {
        assert_eq!(PIECE_TENSOR_SIZE, 3 * 2 * puyo_core::board::NUM_COLORS);
    }

    #[test]
    fn test_context_tensor_size() {
        assert_eq!(CONTEXT_TENSOR_SIZE, PIECE_TENSOR_SIZE);
    }

    #[test]
    fn test_context_encoding() {
        let nc = puyo_core::board::NUM_COLORS;
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = context_to_tensor_data(&current, &next, &next_next);

        assert_eq!(data[0], 1.0);
        assert_eq!(data[nc + 2], 1.0);
        assert_eq!(data[2 * nc + 1], 1.0);
    }

    #[test]
    fn test_pieces_encoding() {
        let nc = puyo_core::board::NUM_COLORS;
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = pieces_to_tensor_data(&current, &next, &next_next);

        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[nc + 2], 1.0);
        assert_eq!(data[nc], 0.0);

        assert_eq!(data[2 * nc + 1], 1.0);
        assert_eq!(data[2 * nc + nc], 1.0);

        assert_eq!(data[4 * nc + 2], 1.0);
        assert_eq!(data[4 * nc + nc], 1.0);
    }

    #[test]
    fn test_adjacency_horizontal() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Red);
        let data = board_to_tensor_data(&board);

        let adj = (puyo_core::board::NUM_COLORS + 1) * puyo_core::board::ROWS * puyo_core::board::COLS;
        assert!((data[adj] - 0.25).abs() < 1e-6);
        assert!((data[adj + 1] - 0.25).abs() < 1e-6);
    }
}
