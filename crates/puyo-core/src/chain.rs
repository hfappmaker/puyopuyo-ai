use crate::board::{Board, PuyoColor, COLS, ROWS, VISIBLE_ROWS};
use std::collections::VecDeque;

/// Result of resolving all chains on a board.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub chain_count: u32,
    pub score: u32,
    /// Details per chain step.
    pub steps: Vec<ChainStep>,
}

/// A single chain step (one round of simultaneous clears).
#[derive(Debug, Clone)]
pub struct ChainStep {
    pub chain_num: u32,     // 1-indexed chain number
    pub groups: Vec<Group>, // groups cleared in this step
    pub score: u32,         // score for this step
}

/// A connected group of same-color puyos.
#[derive(Debug, Clone)]
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>, // (col, row)
}

/// BFS flood-fill from a starting cell. Returns all connected cells of the same color.
pub fn flood_fill(
    board: &Board,
    start_col: usize,
    start_row: usize,
    visited: &mut [[bool; ROWS]; COLS],
) -> Vec<(usize, usize)> {
    let color = board.get(start_col, start_row);
    let mut queue = VecDeque::new();
    let mut cells = Vec::new();
    queue.push_back((start_col, start_row));
    visited[start_col][start_row] = true;

    while let Some((c, r)) = queue.pop_front() {
        cells.push((c, r));
        for (dc, dr) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let nc = c as i32 + dc;
            let nr = r as i32 + dr;
            if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= VISIBLE_ROWS as i32 {
                continue;
            }
            let (nc, nr) = (nc as usize, nr as usize);
            if !visited[nc][nr] && board.get(nc, nr) == color {
                visited[nc][nr] = true;
                queue.push_back((nc, nr));
            }
        }
    }
    cells
}

/// Minimum number of connected same-color puyos required to clear.
pub const MIN_GROUP_SIZE: usize = 4;

/// Find all connected groups of 4+ same-color puyos using BFS flood-fill.
pub fn find_groups(board: &Board) -> Vec<Group> {
    let mut visited = [[false; ROWS]; COLS];
    let mut groups = Vec::new();

    for col in 0..COLS {
        for row in 0..VISIBLE_ROWS {
            if !board.get(col, row).is_color() || visited[col][row] {
                continue;
            }
            let color = board.get(col, row);
            let cells = flood_fill(board, col, row, &mut visited);
            if cells.len() >= MIN_GROUP_SIZE {
                groups.push(Group { color, cells });
            }
        }
    }

    groups
}

/// Resolve one chain step on the board. Modifies board in-place.
/// Returns Some(ChainStep) if groups were found and cleared, None if no groups exist.
/// After returning Some, gravity has been applied and the board is ready for the next step.
/// The caller is responsible for tracking chain_num (1-indexed).
pub fn resolve_one_step(board: &mut Board, chain_num: u32) -> Option<ChainStep> {
    let groups = find_groups(board);
    if groups.is_empty() {
        return None;
    }

    // Remove groups from board
    for group in &groups {
        for &(col, row) in &group.cells {
            board.set(col, row, PuyoColor::Empty);
        }
    }

    // Calculate score for this step
    let step_score = crate::score::calculate_step_score(chain_num, &groups);

    // Apply gravity
    board.apply_gravity();

    Some(ChainStep {
        chain_num,
        groups,
        score: step_score,
    })
}

/// Resolve all chains on the board. Modifies board in-place.
/// Returns the chain result with score details.
pub fn resolve_chains(board: &mut Board) -> ChainResult {
    let steps: Vec<ChainStep> = (1..)
        .map_while(|chain_num| resolve_one_step(board, chain_num))
        .collect();
    let total_score = steps.iter().map(|s| s.score).sum();
    ChainResult {
        chain_count: steps.len() as u32,
        score: total_score,
        steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_chain() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Blue);
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 0);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_single_group_clear() {
        let mut board = Board::new();
        // Place 4 red puyos in a column
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 1);
        assert!(result.score > 0);
        // Board should be empty after clear
        assert_eq!(board.column_height(0), 0);
    }

    #[test]
    fn test_horizontal_group() {
        let mut board = Board::new();
        // Place 4 red puyos in a row (bottom of columns 0-3)
        for col in 0..4 {
            board.drop_puyo(col, PuyoColor::Red);
        }
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 1);
    }

    #[test]
    fn test_two_chain() {
        // 2-chain: first clear triggers gravity which creates second clear.
        // Col 0: B B B  (rows 0,1,2) — only 3 blues, not clearable yet
        // Col 1: R R R R (rows 0,1,2,3) — 4 reds, clearable
        // Col 1: B       (row 4, on top of reds) — after reds clear, B falls to row 0
        // After chain 1: Col 0 has B at (0,0),(0,1),(0,2) and Col 1 has B at (1,0)
        // = 4 connected blues → chain 2!
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        board.drop_puyo(1, PuyoColor::Blue);
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 2);
        assert_eq!(board.column_height(0), 0);
        assert_eq!(board.column_height(1), 0);
    }

    #[test]
    fn test_gravity_after_clear() {
        let mut board = Board::new();
        // Col 0: R R R R (will clear)
        // Col 0, on top: G (row 4) - should fall to row 0 after clear
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        board.drop_puyo(0, PuyoColor::Green);
        resolve_chains(&mut board);
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_l_shape_group() {
        let mut board = Board::new();
        // L-shape of 4 red:
        // Col 0 row 0: R
        // Col 0 row 1: R
        // Col 1 row 0: R
        // Col 2 row 0: R
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(2, 0, PuyoColor::Red);
        let groups = find_groups(&board);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].cells.len(), 4);
    }

    #[test]
    fn test_hidden_row_puyo_not_cleared() {
        use crate::board::VISIBLE_ROWS;
        let mut board = Board::new();
        // 4 red in col 0 rows 0-3 (visible, should clear)
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        // 1 red in hidden row (row 12) — should NOT be cleared
        board.set(0, VISIBLE_ROWS, PuyoColor::Red);
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 1);
        // Hidden row puyo survives and falls down via gravity
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_hidden_row_puyo_falls_after_clear() {
        use crate::board::VISIBLE_ROWS;
        let mut board = Board::new();
        // Col 0: 4 red (rows 0-3), then green on top (row 4)
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        board.set(0, 4, PuyoColor::Green);
        // Place a green in hidden row
        board.set(0, VISIBLE_ROWS, PuyoColor::Green);
        resolve_chains(&mut board);
        // After red clears, green from row 4 and hidden row fall down
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.get(0, 1), PuyoColor::Green);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    fn test_resolve_one_step_two_chain() {
        // Same 2-chain setup as test_two_chain
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        board.drop_puyo(1, PuyoColor::Blue);

        // Step 1: should clear 4 reds
        let step1 = resolve_one_step(&mut board, 1);
        assert!(step1.is_some());
        let step1 = step1.unwrap();
        assert_eq!(step1.chain_num, 1);

        // Step 2: after gravity, blues connect → clear
        let step2 = resolve_one_step(&mut board, 2);
        assert!(step2.is_some());
        let step2 = step2.unwrap();
        assert_eq!(step2.chain_num, 2);

        // Step 3: no more groups
        let step3 = resolve_one_step(&mut board, 3);
        assert!(step3.is_none());

        // Board should be empty
        assert_eq!(board.column_height(0), 0);
        assert_eq!(board.column_height(1), 0);
    }

    #[test]
    fn test_resolve_one_step_no_chain() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Blue);
        let result = resolve_one_step(&mut board, 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_hidden_row_only_does_not_clear() {
        use crate::board::VISIBLE_ROWS;
        let mut board = Board::new();
        // Place 4 red in hidden row across cols 0-3
        // (Only 1 hidden row per column, so place them vertically is impossible.
        //  Instead, place 4 in row VISIBLE_ROWS across 4 columns.)
        for col in 0..4 {
            board.set(col, VISIBLE_ROWS, PuyoColor::Red);
        }
        let groups = find_groups(&board);
        assert!(
            groups.is_empty(),
            "Hidden-row-only puyos should not form clearable groups"
        );
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 0);
    }
}
