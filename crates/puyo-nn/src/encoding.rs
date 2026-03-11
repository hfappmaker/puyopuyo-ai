use puyo_core::board::{Board, PuyoColor, COLS, ROWS, VISIBLE_ROWS};
use puyo_core::chain::flood_fill;

/// Number of input channels:
/// 0-4: one-hot per PuyoColor (Empty, Red, Green, Blue, Yellow)
/// 5: column heights (normalized by ROWS)
/// 6: adjacent empty count (normalized by 4)
/// 7: connected group size (normalized by 4)
pub const NUM_CHANNELS: usize = 8;

/// Total size of the flattened tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// Convert a Board to a flat f32 array.
/// Layout: [channel][row][col] = [8][14][6], total 672 floats.
/// Channels 0-4: one-hot encoding (Empty, Red, Green, Blue, Yellow).
/// Channel 5: column height / ROWS for all cells in that column.
/// Channel 6: adjacent empty count / 4.0 for non-empty cells.
/// Channel 7: connected group size / 4.0 for non-empty cells.
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

    // Channel 6: adjacent empty count (normalized by 4)
    let ch6_offset = 6 * ROWS * COLS;
    for col in 0..COLS {
        for row in 0..ROWS {
            if !board.get(col, row).is_color() {
                continue;
            }
            let mut empty_count = 0u32;
            for (dc, dr) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nc = col as i32 + dc;
                let nr = row as i32 + dr;
                if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= ROWS as i32 {
                    continue;
                }
                if board.get(nc as usize, nr as usize) == PuyoColor::Empty {
                    empty_count += 1;
                }
            }
            data[ch6_offset + row * COLS + col] = empty_count as f32 / 4.0;
        }
    }

    // Channel 7: connected group size (normalized by 4)
    // flood_fill is bounded by VISIBLE_ROWS, so hidden rows naturally get 0.0
    let ch7_offset = 7 * ROWS * COLS;
    let mut visited = [[false; ROWS]; COLS];
    for col in 0..COLS {
        for row in 0..VISIBLE_ROWS {
            if !board.get(col, row).is_color() || visited[col][row] {
                continue;
            }
            let cells = flood_fill(board, col, row, &mut visited);
            let group_size_norm = cells.len() as f32 / 4.0;
            for &(c, r) in &cells {
                data[ch7_offset + r * COLS + c] = group_size_norm;
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Channels 6, 7: all 0.0 for empty board (no colored cells)
        for ch in 6..8 {
            let offset = ch * ROWS * COLS;
            for i in 0..(ROWS * COLS) {
                assert_eq!(data[offset + i], 0.0);
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

        // Channel 5: column 0 height = 1/14
        let ch5_offset = 5 * ROWS * COLS;
        assert_eq!(data[ch5_offset + 0 * COLS + 0], 1.0 / ROWS as f32);

        // Channel 6: single puyo at (0,0), check adjacent empties
        // left: out of bounds, right: (1,0) empty, down: out of bounds, up: (0,1) empty
        // = 2 adjacent empties → 2/4 = 0.5
        let ch6_offset = 6 * ROWS * COLS;
        assert_eq!(data[ch6_offset + 0 * COLS + 0], 0.5);

        // Channel 7: group size = 1 → 1/4 = 0.25
        let ch7_offset = 7 * ROWS * COLS;
        assert_eq!(data[ch7_offset + 0 * COLS + 0], 0.25);
    }

    #[test]
    fn test_tensor_size() {
        // 8 channels * 14 rows * 6 cols = 672
        assert_eq!(TENSOR_SIZE, 672);
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

    #[test]
    fn test_group_size_channel() {
        let mut board = Board::new();
        // Place 3 connected reds in an L shape
        board.set(0, 0, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        let data = board_to_tensor_data(&board);

        let ch7_offset = 7 * ROWS * COLS;
        let expected = 3.0 / 4.0; // group size 3, normalized by 4
        assert_eq!(data[ch7_offset + 0 * COLS + 0], expected);
        assert_eq!(data[ch7_offset + 0 * COLS + 1], expected);
        assert_eq!(data[ch7_offset + 1 * COLS + 0], expected);
    }
}
