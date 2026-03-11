use puyo_core::board::{Board, COLS, ROWS};

/// Number of input channels:
/// 0-4: one-hot per PuyoColor (Empty, Red, Green, Blue, Yellow)
/// 5: column heights (normalized by ROWS)
pub const NUM_CHANNELS: usize = 6;

/// Total size of the flattened tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// Convert a Board to a flat f32 array.
/// Layout: [channel][row][col] = [6][14][6], total 504 floats.
/// Channels 0-4: one-hot encoding (Empty, Red, Green, Blue, Yellow).
/// Channel 5: column height / ROWS for all cells in that column.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

    // Channels 0-4: one-hot encoding
    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row) as u8 as usize;
            let index = color * ROWS * COLS + row * COLS + col;
            data[index] = 1.0;
        }
    }

    // Channel 5: column heights (normalized)
    let ch5_offset = 5 * ROWS * COLS;
    for col in 0..COLS {
        let height_norm = board.column_height(col) as f32 / ROWS as f32;
        for row in 0..ROWS {
            data[ch5_offset + row * COLS + col] = height_norm;
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
        // All cells are Empty (channel 0), so channel 0 should be all 1.0
        for row in 0..ROWS {
            for col in 0..COLS {
                assert_eq!(data[0 * ROWS * COLS + row * COLS + col], 1.0);
            }
        }
        // Color channels 1-4 should be 0.0
        for ch in 1..5 {
            for row in 0..ROWS {
                for col in 0..COLS {
                    assert_eq!(data[ch * ROWS * COLS + row * COLS + col], 0.0);
                }
            }
        }
        // Channel 5 (column heights): all 0.0 for empty board
        let ch5_offset = 5 * ROWS * COLS;
        for i in 0..(ROWS * COLS) {
            assert_eq!(data[ch5_offset + i], 0.0);
        }
    }

    #[test]
    fn test_single_puyo_encoding() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        let data = board_to_tensor_data(&board);
        // Red = channel 1, row 0, col 0
        assert_eq!(data[1 * ROWS * COLS + 0 * COLS + 0], 1.0);
        // Empty channel at (0, 0) should now be 0
        assert_eq!(data[0 * ROWS * COLS + 0 * COLS + 0], 0.0);

        // Channel 5: column 0 height = 1/14
        let ch5_offset = 5 * ROWS * COLS;
        assert_eq!(data[ch5_offset + 0 * COLS + 0], 1.0 / ROWS as f32);
    }

    #[test]
    fn test_tensor_size() {
        // 6 channels * 14 rows * 6 cols = 504
        assert_eq!(TENSOR_SIZE, 504);
    }

    #[test]
    fn test_column_height_channel() {
        let mut board = Board::new();
        // Stack 3 puyos in column 2
        board.drop_puyo(2, PuyoColor::Red);
        board.drop_puyo(2, PuyoColor::Green);
        board.drop_puyo(2, PuyoColor::Blue);
        let data = board_to_tensor_data(&board);

        let ch5_offset = 5 * ROWS * COLS;
        let expected = 3.0 / ROWS as f32;
        // All rows in column 2 should have the same height value
        for row in 0..ROWS {
            assert_eq!(data[ch5_offset + row * COLS + 2], expected);
        }
        // Column 0 should be 0
        assert_eq!(data[ch5_offset + 0 * COLS + 0], 0.0);
    }
}
