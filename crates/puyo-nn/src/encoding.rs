use puyo_core::board::{Board, PuyoColor, COLS, ROWS};
use puyo_core::chain;

/// Number of input channels:
/// 0-3: one-hot per color (Red, Green, Blue, Yellow) — Empty is implicit (all zero)
/// 4-9: chain step maps per trigger column (col 0..5)
pub const NUM_CHANNELS: usize = 10;

/// Total size of the flattened tensor.
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

const COLORS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];
const VIRTUAL_PUYO_COUNT: usize = 3;

/// Convert a Board to a flat f32 array.
/// Layout: [channel][row][col] = [10][14][6], total 840 floats.
/// Channels 0-3: one-hot encoding (Red, Green, Blue, Yellow).
/// Channels 4-9: chain step maps for trigger columns 0-5.
pub fn board_to_tensor_data(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

    // Channels 0-3: one-hot encoding (skip Empty)
    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if color.is_color() {
                let ch = color as u8 as usize - 1; // Red=0, Green=1, Blue=2, Yellow=3
                let index = ch * ROWS * COLS + row * COLS + col;
                data[index] = 1.0;
            }
        }
    }

    // Channels 4-9: chain step maps
    let maps = compute_chain_step_maps(board);
    for trigger_col in 0..COLS {
        let ch_offset = (4 + trigger_col) * ROWS * COLS;
        for col in 0..COLS {
            for row in 0..ROWS {
                data[ch_offset + row * COLS + col] = maps[trigger_col][col][row];
            }
        }
    }

    data
}

/// Compute chain step maps for all 6 trigger columns.
/// Returns `[trigger_col][cell_col][cell_row]` with values `step / 10.0` clamped to [0, 1].
fn compute_chain_step_maps(board: &Board) -> [[[f32; ROWS]; COLS]; COLS] {
    let mut maps = [[[0.0f32; ROWS]; COLS]; COLS];

    for trigger_col in 0..COLS {
        let mut best_chain = 0u32;
        let mut best_map = [[0.0f32; ROWS]; COLS];

        // Check current board without adding any puyos
        {
            let (step_map, chain_count) = simulate_with_tracking(board);
            if chain_count > best_chain {
                best_chain = chain_count;
                best_map = step_map;
            }
        }

        // Try adding virtual puyos of each color
        for &color in &COLORS {
            let h = board.column_height(trigger_col);
            let available = if board.has_isolated_top_puyo(trigger_col) {
                ROWS - 1 - h
            } else {
                ROWS - h
            };
            if available == 0 {
                continue;
            }
            let count = VIRTUAL_PUYO_COUNT.min(available);
            let mut sim = board.clone();
            for _ in 0..count {
                sim.drop_puyo(trigger_col, color);
            }
            let (step_map, chain_count) = simulate_with_tracking(&sim);
            if chain_count > best_chain {
                best_chain = chain_count;
                best_map = step_map;
            }
        }

        maps[trigger_col] = best_map;
    }
    maps
}

/// Simulate chain resolution while tracking which step each cell was cleared in.
/// Returns `(step_map[col][row], chain_count)` where step_map values are `step / 10.0` clamped to [0, 1].
fn simulate_with_tracking(board: &Board) -> ([[f32; ROWS]; COLS], u32) {
    let mut sim = board.clone();
    let mut step_map = [[0.0f32; ROWS]; COLS];
    let mut chain_num = 0u32;

    loop {
        let groups = chain::find_groups(&sim);
        if groups.is_empty() {
            break;
        }
        chain_num += 1;

        for group in &groups {
            for &(col, row) in &group.cells {
                step_map[col][row] = (chain_num as f32 / 10.0).min(1.0);
                sim.set(col, row, PuyoColor::Empty);
            }
        }
        sim.apply_gravity();
    }
    (step_map, chain_num)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_board_encoding() {
        let board = Board::new();
        let data = board_to_tensor_data(&board);
        // All cells are Empty, so color channels 0-3 should be all 0.0
        for ch in 0..4 {
            for row in 0..ROWS {
                for col in 0..COLS {
                    assert_eq!(data[ch * ROWS * COLS + row * COLS + col], 0.0);
                }
            }
        }
        // Channels 4-9 (chain step maps): all 0.0 for empty board
        for ch in 4..10 {
            for i in 0..(ROWS * COLS) {
                assert_eq!(data[ch * ROWS * COLS + i], 0.0);
            }
        }
    }

    #[test]
    fn test_single_puyo_encoding() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        let data = board_to_tensor_data(&board);
        // Red = channel 0, row 0, col 0
        assert_eq!(data[0 * ROWS * COLS + 0 * COLS + 0], 1.0);
        // Other color channels at (0, 0) should be 0
        for ch in 1..4 {
            assert_eq!(data[ch * ROWS * COLS + 0 * COLS + 0], 0.0);
        }
        // ch4 (trigger col 0): adding 3 virtual reds makes 4 → 1-chain detected
        assert_eq!(data[4 * ROWS * COLS + 0 * COLS + 0], 0.1); // step 1
        // Columns without relevant trigger should have no chain
        // (other trigger cols can't trigger a chain with just 1 red in col 0)
    }

    #[test]
    fn test_tensor_size() {
        // 10 channels * 14 rows * 6 cols = 840
        assert_eq!(TENSOR_SIZE, 840);
    }

    #[test]
    fn test_chain_step_map_single_chain() {
        // Place 4 reds vertically in col 0 → 1 chain when resolved
        let mut board = Board::new();
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let maps = compute_chain_step_maps(&board);

        // The current board already has a chain, so all trigger columns should detect it
        // ch4 (trigger col 0): step_map should show step 1 (0.1) at col 0 rows 0-3
        for row in 0..4 {
            assert!(
                maps[0][0][row] > 0.0,
                "col 0, row {} should be cleared",
                row
            );
            assert_eq!(maps[0][0][row], 0.1); // step 1 / 10.0
        }
        // Cells not cleared should be 0
        assert_eq!(maps[0][0][4], 0.0);
    }

    #[test]
    fn test_chain_step_map_two_chain() {
        // Known 2-chain pattern:
        // Col 0: B B B (rows 0-2)
        // Col 1: R R R R (rows 0-3) + B on top (row 4)
        // Chain 1: 4 reds clear → B falls to row 0
        // Chain 2: 4 blues clear (3 from col 0 + 1 from col 1)
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        board.drop_puyo(1, PuyoColor::Blue);

        let maps = compute_chain_step_maps(&board);

        // This board already has chains, so all trigger columns see it.
        // Step 1: reds at col 1 rows 0-3 cleared, but after gravity blue falls
        //   to col 1 row 0 and gets cleared in step 2, overwriting step_map[1][0].
        // So col 1 rows 1-3 = 0.1 (step 1 only), col 1 row 0 = 0.2 (overwritten by step 2)
        for row in 1..4 {
            assert_eq!(maps[0][1][row], 0.1, "col 1, row {} should be step 1", row);
        }
        assert_eq!(maps[0][1][0], 0.2, "col 1, row 0 overwritten by step 2 blue");

        // Step 2: blues at col 0 rows 0-2 → 0.2
        for row in 0..3 {
            assert_eq!(maps[0][0][row], 0.2, "col 0, row {} should be step 2", row);
        }
    }

    #[test]
    fn test_chain_step_map_virtual_trigger() {
        // Place 3 reds in col 0 (not enough to chain by itself)
        // Adding virtual puyos should trigger a chain
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Red);
        }

        let maps = compute_chain_step_maps(&board);

        // Trigger col 0: adding 3 reds makes 6 reds → chain
        // The best trigger should find the chain
        assert!(
            maps[0][0][0] > 0.0,
            "Virtual trigger should detect chain at col 0"
        );
    }

    #[test]
    fn test_color_channel_mapping() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Green);
        board.drop_puyo(2, PuyoColor::Blue);
        board.drop_puyo(3, PuyoColor::Yellow);
        let data = board_to_tensor_data(&board);

        // Red → ch0, Green → ch1, Blue → ch2, Yellow → ch3
        assert_eq!(data[0 * ROWS * COLS + 0 * COLS + 0], 1.0); // Red at col 0
        assert_eq!(data[1 * ROWS * COLS + 0 * COLS + 1], 1.0); // Green at col 1
        assert_eq!(data[2 * ROWS * COLS + 0 * COLS + 2], 1.0); // Blue at col 2
        assert_eq!(data[3 * ROWS * COLS + 0 * COLS + 3], 1.0); // Yellow at col 3
    }
}
