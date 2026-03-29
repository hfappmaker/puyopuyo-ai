use puyo_core::board::{Board, COLS, NUM_COLORS, ROWS};
use puyo_core::piece::Piece;

/// Number of input channels:
/// 0..(NUM_COLORS-1): one-hot per color — Empty is implicit (all zero)
/// NUM_COLORS: occupancy map (1.0 if puyo present, 0.0 if empty)
/// NUM_COLORS+1: adjacency map (count of same-color neighbors / 4.0)
pub const NUM_CHANNELS: usize = NUM_COLORS + 2;

/// Total size of the flattened tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// Convert a Board to a flat f32 array.
/// Layout: [channel][row][col], total NUM_CHANNELS * ROWS * COLS floats.
/// Channels 0..(NUM_COLORS-1): one-hot encoding per color.
/// Channel NUM_COLORS: occupancy map.
/// Channel NUM_COLORS+1: adjacency map.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

    let occ_offset = NUM_COLORS * ROWS * COLS;
    let adj_offset = (NUM_COLORS + 1) * ROWS * COLS;

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if color.is_color() {
                // Channels 0..(NUM_COLORS-1): one-hot encoding
                let ch = color as u8 as usize - 1; // Red=0, Green=1, ...
                data[ch * ROWS * COLS + row * COLS + col] = 1.0;

                // Channel NUM_COLORS: occupancy
                data[occ_offset + row * COLS + col] = 1.0;

                // Channel NUM_COLORS+1: adjacency (same-color neighbor count / 4.0)
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

/// Number of floats for piece encoding: 3 pieces × 2 colors × NUM_COLORS one-hot.
pub const PIECE_TENSOR_SIZE: usize = 3 * 2 * NUM_COLORS;

/// Context tensor size: piece encoding only.
pub const CONTEXT_TENSOR_SIZE: usize = PIECE_TENSOR_SIZE;

/// Convert three pieces (current, next, next_next) to a flat f32 array.
/// Each piece encodes axis_color and satellite_color as NUM_COLORS-dim one-hot vectors.
/// Layout: [current_axis(NC), current_sat(NC), next_axis(NC), next_sat(NC), nn_axis(NC), nn_sat(NC)]
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

/// Convert pieces to context tensor for FiLM conditioning.
/// Layout: [pieces one-hot (PIECE_TENSOR_SIZE)] floats.
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
        assert_eq!(data[occ], 1.0); // col0, row0: occupied
        assert_eq!(data[occ + 2 % COLS], 1.0); // occupied
        assert_eq!(data[occ + 1], 0.0); // col1, row0: empty
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
        let nc = NUM_COLORS;
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = context_to_tensor_data(&current, &next, &next_next);

        assert_eq!(data[0], 1.0); // current axis=Red (ch0)
        assert_eq!(data[nc + 2], 1.0); // current sat=Blue (ch2)
        assert_eq!(data[2 * nc + 1], 1.0); // next axis=Green (ch1)
    }

    #[test]
    fn test_pieces_encoding() {
        let nc = NUM_COLORS;
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Red);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = pieces_to_tensor_data(&current, &next, &next_next);

        // current: axis=Red(0), sat=Blue(2)
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[nc + 2], 1.0);
        assert_eq!(data[nc], 0.0);

        // next: axis=Green(1), sat=Red(0)
        assert_eq!(data[2 * nc + 1], 1.0);
        assert_eq!(data[2 * nc + nc], 1.0); // sat=Red

        // next_next: axis=Blue(2), sat=Red(0)
        assert_eq!(data[4 * nc + 2], 1.0);
        assert_eq!(data[4 * nc + nc], 1.0);
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
