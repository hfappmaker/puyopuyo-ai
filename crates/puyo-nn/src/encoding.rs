use puyo_core::board::{Board, COLS, ROWS};

/// Number of input channels (one per PuyoColor variant: Empty, Red, Green, Blue, Yellow).
pub const NUM_CHANNELS: usize = 5;

/// Total size of the flattened one-hot tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// Convert a Board to a flat f32 array in one-hot format.
/// Layout: [channel][row][col] = [5][13][6], total 390 floats.
/// Channel 0 = Empty, 1 = Red, 2 = Green, 3 = Blue, 4 = Yellow.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];
    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row) as u8 as usize;
            let index = color * ROWS * COLS + row * COLS + col;
            data[index] = 1.0;
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
        // Other channels should be 0.0
        for ch in 1..NUM_CHANNELS {
            for row in 0..ROWS {
                for col in 0..COLS {
                    assert_eq!(data[ch * ROWS * COLS + row * COLS + col], 0.0);
                }
            }
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
    }

    #[test]
    fn test_tensor_size() {
        // 5 channels * 14 rows * 6 cols = 420
        assert_eq!(TENSOR_SIZE, 420);
    }
}
