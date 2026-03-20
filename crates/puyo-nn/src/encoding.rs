use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::Piece;

/// Number of input channels:
/// 0-3: one-hot per color (Red, Green, Blue, Yellow) — Empty is implicit (all zero)
/// 4: occupancy map (1.0 if puyo present, 0.0 if empty)
/// 5: adjacency map (count of same-color neighbors / 4.0)
pub const NUM_CHANNELS: usize = 6;

/// Total size of the flattened tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// Convert a Board to a flat f32 array.
/// Layout: [channel][row][col] = [6][14][6], total 504 floats.
/// Channels 0-3: one-hot encoding (Red, Green, Blue, Yellow).
/// Channel 4: occupancy map.
/// Channel 5: adjacency map.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

    let ch4_offset = 4 * ROWS * COLS;
    let ch5_offset = 5 * ROWS * COLS;

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if color.is_color() {
                // Channels 0-3: one-hot encoding
                let ch = color as u8 as usize - 1; // Red=0, Green=1, Blue=2, Yellow=3
                data[ch * ROWS * COLS + row * COLS + col] = 1.0;

                // Channel 4: occupancy
                data[ch4_offset + row * COLS + col] = 1.0;

                // Channel 5: adjacency (same-color neighbor count / 4.0)
                let mut count = 0u8;
                if col > 0 && board.get(col - 1, row) == color {
                    count += 1;
                }
                if col + 1 < COLS && board.get(col + 1, row) == color {
                    count += 1;
                }
                if row > 0 && board.get(col, row - 1) == color {
                    count += 1;
                }
                if row + 1 < ROWS && board.get(col, row + 1) == color {
                    count += 1;
                }
                data[ch5_offset + row * COLS + col] = count as f32 / 4.0;
            }
        }
    }

    data
}

/// Number of floats for piece encoding: 3 pieces × 2 colors × 4 one-hot = 24.
pub const PIECE_TENSOR_SIZE: usize = 24;

/// Context tensor size: piece encoding (24).
pub const CONTEXT_TENSOR_SIZE: usize = PIECE_TENSOR_SIZE;

/// Convert three pieces (current, next, next_next) to a flat f32 array.
/// Each piece encodes axis_color and satellite_color as 4-dim one-hot vectors.
/// Layout: [current_axis(4), current_sat(4), next_axis(4), next_sat(4), nn_axis(4), nn_sat(4)]
pub fn pieces_to_tensor_data(current: &Piece, next: &Piece, next_next: &Piece) -> [f32; PIECE_TENSOR_SIZE] {
    let mut data = [0.0f32; PIECE_TENSOR_SIZE];
    let pieces = [current, next, next_next];
    for (i, piece) in pieces.iter().enumerate() {
        let base = i * 8;
        // axis_color one-hot (Red=0, Green=1, Blue=2, Yellow=3)
        let axis_ch = piece.axis_color as u8 as usize - 1;
        data[base + axis_ch] = 1.0;
        // satellite_color one-hot
        let sat_ch = piece.satellite_color as u8 as usize - 1;
        data[base + 4 + sat_ch] = 1.0;
    }
    data
}

/// Convert pieces to context tensor for FiLM conditioning.
/// Layout: [pieces one-hot (24)] = 24 floats.
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
        assert_eq!(TENSOR_SIZE, 504); // 6 * 14 * 6
    }

    #[test]
    fn test_color_channel_mapping() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Green);
        board.drop_puyo(2, PuyoColor::Blue);
        board.drop_puyo(3, PuyoColor::Yellow);
        let data = board_to_tensor_data(&board);

        assert_eq!(data[0 * ROWS * COLS + 0 * COLS + 0], 1.0); // Red at col 0
        assert_eq!(data[1 * ROWS * COLS + 0 * COLS + 1], 1.0); // Green at col 1
        assert_eq!(data[2 * ROWS * COLS + 0 * COLS + 2], 1.0); // Blue at col 2
        assert_eq!(data[3 * ROWS * COLS + 0 * COLS + 3], 1.0); // Yellow at col 3
    }

    #[test]
    fn test_occupancy_channel() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(2, PuyoColor::Blue);
        let data = board_to_tensor_data(&board);

        let ch4 = 4 * ROWS * COLS;
        assert_eq!(data[ch4 + 0 * COLS + 0], 1.0); // col 0, row 0: occupied
        assert_eq!(data[ch4 + 0 * COLS + 2], 1.0); // col 2, row 0: occupied
        assert_eq!(data[ch4 + 0 * COLS + 1], 0.0); // col 1, row 0: empty
    }

    #[test]
    fn test_adjacency_channel() {
        let mut board = Board::new();
        // Stack 3 reds vertically in col 0
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let data = board_to_tensor_data(&board);

        let ch5 = 5 * ROWS * COLS;
        // row 0 (bottom): 1 same-color neighbor above → 1/4 = 0.25
        assert!((data[ch5 + 0 * COLS + 0] - 0.25).abs() < 1e-6);
        // row 1 (middle): 2 same-color neighbors (above + below) → 2/4 = 0.5
        assert!((data[ch5 + 1 * COLS + 0] - 0.5).abs() < 1e-6);
        // row 2 (top): 1 same-color neighbor below → 1/4 = 0.25
        assert!((data[ch5 + 2 * COLS + 0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_piece_tensor_size() {
        assert_eq!(PIECE_TENSOR_SIZE, 24);
    }

    #[test]
    fn test_context_tensor_size() {
        assert_eq!(CONTEXT_TENSOR_SIZE, 24);
    }

    #[test]
    fn test_context_encoding() {
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = context_to_tensor_data(&current, &next, &next_next);

        // Pieces one-hot (same as pieces_to_tensor_data)
        assert_eq!(data[0], 1.0); // current axis=Red
        assert_eq!(data[4 + 2], 1.0); // current sat=Blue
        assert_eq!(data[8 + 1], 1.0); // next axis=Green
    }

    #[test]
    fn test_pieces_encoding() {
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);
        let data = pieces_to_tensor_data(&current, &next, &next_next);

        // current: axis=Red(0), sat=Blue(2)
        assert_eq!(data[0], 1.0); // Red
        assert_eq!(data[1], 0.0);
        assert_eq!(data[4 + 2], 1.0); // Blue
        assert_eq!(data[4 + 0], 0.0);

        // next: axis=Green(1), sat=Yellow(3)
        assert_eq!(data[8 + 1], 1.0); // Green
        assert_eq!(data[8 + 4 + 3], 1.0); // Yellow

        // next_next: axis=Blue(2), sat=Red(0)
        assert_eq!(data[16 + 2], 1.0); // Blue
        assert_eq!(data[16 + 4 + 0], 1.0); // Red
    }

    #[test]
    fn test_adjacency_horizontal() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Red);
        let data = board_to_tensor_data(&board);

        let ch5 = 5 * ROWS * COLS;
        // col 0, row 0: 1 same-color neighbor to the right → 0.25
        assert!((data[ch5 + 0 * COLS + 0] - 0.25).abs() < 1e-6);
        // col 1, row 0: 1 same-color neighbor to the left → 0.25
        assert!((data[ch5 + 0 * COLS + 1] - 0.25).abs() < 1e-6);
    }
}
