use puyo_core::board::{Board, COLS, ROWS};

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

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;

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
