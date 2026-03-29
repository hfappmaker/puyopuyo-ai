//! NN encoding functions.
//!
//! Board と Piece を固定長 f32 配列にエンコードする。
//! テンソル定数は `puyo_core::config` で定義され、ここから再エクスポートする。

pub use puyo_core::state::{
    CONTEXT_TENSOR_SIZE, NUM_CHANNELS, PIECE_TENSOR_SIZE, TENSOR_SIZE,
};

use puyo_core::board::{Board, COLS, NUM_COLORS, ROWS};
use puyo_core::piece::Piece;

/// Board を [channel][row][col] の固定長 f32 配列に変換する。
///
/// チャンネル:
/// - 0..(NUM_COLORS-1): 色ごとの one-hot
/// - NUM_COLORS: occupancy map (ぷよの有無)
/// - NUM_COLORS+1: adjacency map (同色隣接数 / 4.0)
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

    let occ_offset = NUM_COLORS * ROWS * COLS;
    let adj_offset = (NUM_COLORS + 1) * ROWS * COLS;

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if color.is_color() {
                let ch = color as u8 as usize - 1;
                data[ch * ROWS * COLS + row * COLS + col] = 1.0;

                data[occ_offset + row * COLS + col] = 1.0;

                let count = [
                    col > 0 && board.get(col - 1, row) == color,
                    col + 1 < COLS && board.get(col + 1, row) == color,
                    row > 0 && board.get(col, row - 1) == color,
                    row + 1 < ROWS && board.get(col, row + 1) == color,
                ]
                .iter()
                .filter(|&&b| b)
                .count();
                data[adj_offset + row * COLS + col] = count as f32 / 4.0;
            }
        }
    }

    data
}

/// 3ピース（current, next, next_next）を固定長 f32 配列に変換する。
/// 各ピースの axis_color, satellite_color を NUM_COLORS 次元 one-hot でエンコード。
pub fn pieces_to_tensor_data(current: &Piece, next: &Piece, next_next: &Piece) -> [f32; PIECE_TENSOR_SIZE] {
    let mut data = [0.0f32; PIECE_TENSOR_SIZE];
    let pieces = [current, next, next_next];
    for (i, piece) in pieces.iter().enumerate() {
        let base = i * (2 * NUM_COLORS);
        let axis_ch = piece.axis_color as u8 as usize - 1;
        data[base + axis_ch] = 1.0;
        let sat_ch = piece.satellite_color as u8 as usize - 1;
        data[base + NUM_COLORS + sat_ch] = 1.0;
    }
    data
}

/// コンテキストテンソル（FiLM 条件付け用）に変換する。
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
        assert_eq!(TENSOR_SIZE, NUM_CHANNELS * ROWS * COLS);
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
            assert_eq!(data[ch * ROWS * COLS + ch], 1.0);
        }
    }

    #[test]
    fn test_occupancy_channel() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(2 % COLS, PuyoColor::Blue);
        let data = board_to_tensor_data(&board);

        let occ = NUM_COLORS * ROWS * COLS;
        assert_eq!(data[occ], 1.0);
        assert_eq!(data[occ + 2 % COLS], 1.0);
        assert_eq!(data[occ + 1], 0.0);
    }

    #[test]
    fn test_adjacency_channel() {
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let data = board_to_tensor_data(&board);

        let adj = (NUM_COLORS + 1) * ROWS * COLS;
        assert!((data[adj] - 0.25).abs() < 1e-6);
        assert!((data[adj + COLS] - 0.5).abs() < 1e-6);
        assert!((data[adj + 2 * COLS] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_piece_tensor_size() {
        assert_eq!(PIECE_TENSOR_SIZE, 3 * 2 * NUM_COLORS);
    }

    #[test]
    fn test_context_tensor_size() {
        assert_eq!(CONTEXT_TENSOR_SIZE, PIECE_TENSOR_SIZE);
    }

    #[test]
    fn test_context_encoding() {
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = context_to_tensor_data(&current, &next, &next_next);

        assert_eq!(data[0], 1.0);
        assert_eq!(data[NUM_COLORS + 2], 1.0);
        assert_eq!(data[2 * NUM_COLORS + 1], 1.0);
    }

    #[test]
    fn test_pieces_encoding() {
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = pieces_to_tensor_data(&current, &next, &next_next);

        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[NUM_COLORS + 2], 1.0);
        assert_eq!(data[NUM_COLORS], 0.0);

        assert_eq!(data[2 * NUM_COLORS + 1], 1.0);
        assert_eq!(data[2 * NUM_COLORS + NUM_COLORS], 1.0);

        assert_eq!(data[4 * NUM_COLORS + 2], 1.0);
        assert_eq!(data[4 * NUM_COLORS + NUM_COLORS], 1.0);
    }

    #[test]
    fn test_adjacency_horizontal() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Red);
        let data = board_to_tensor_data(&board);

        let adj = (NUM_COLORS + 1) * ROWS * COLS;
        assert!((data[adj] - 0.25).abs() < 1e-6);
        assert!((data[adj + 1] - 0.25).abs() < 1e-6);
    }
}
