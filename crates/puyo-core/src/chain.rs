use crate::board::{Board, PuyoColor, COLS, ROWS};
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
    pub chain_num: u32,       // 1-indexed chain number
    pub groups: Vec<Group>,   // groups cleared in this step
    pub score: u32,           // score for this step
}

/// A connected group of same-color puyos.
#[derive(Debug, Clone)]
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>, // (col, row)
}

/// Find all connected groups of 4+ same-color puyos using BFS flood-fill.
pub fn find_groups(board: &Board) -> Vec<Group> {
    let mut visited = [[false; ROWS]; COLS];
    let mut groups = Vec::new();

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if !color.is_color() || visited[col][row] {
                continue;
            }

            // BFS flood-fill
            let mut queue = VecDeque::new();
            let mut cells = Vec::new();
            queue.push_back((col, row));
            visited[col][row] = true;

            while let Some((c, r)) = queue.pop_front() {
                cells.push((c, r));

                // Check 4 neighbors
                let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dc, dr) in neighbors {
                    let nc = c as i32 + dc;
                    let nr = r as i32 + dr;
                    if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= ROWS as i32 {
                        continue;
                    }
                    let nc = nc as usize;
                    let nr = nr as usize;
                    if !visited[nc][nr] && board.get(nc, nr) == color {
                        visited[nc][nr] = true;
                        queue.push_back((nc, nr));
                    }
                }
            }

            if cells.len() >= 4 {
                groups.push(Group { color, cells });
            }
        }
    }

    groups
}

/// Resolve all chains on the board. Modifies board in-place.
/// Returns the chain result with score details.
pub fn resolve_chains(board: &mut Board) -> ChainResult {
    let mut chain_count = 0u32;
    let mut total_score = 0u32;
    let mut steps = Vec::new();

    loop {
        let groups = find_groups(board);
        if groups.is_empty() {
            break;
        }

        chain_count += 1;

        // Remove groups from board
        for group in &groups {
            for &(col, row) in &group.cells {
                board.set(col, row, PuyoColor::Empty);
            }
        }

        // Calculate score for this step
        let step_score = crate::score::calculate_step_score(chain_count, &groups);
        total_score += step_score;

        steps.push(ChainStep {
            chain_num: chain_count,
            groups: groups,
            score: step_score,
        });

        // Apply gravity
        board.apply_gravity();
    }

    ChainResult {
        chain_count,
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
        let mut board = Board::new();
        // Set up a 2-chain:
        // Col 0: Blue Blue Blue Blue (will be exposed after Red clears)
        // Col 0-3: Red at row 4 (on top of blues)
        // Actually let's do it properly:
        // Bottom layer: 4 blue in col 0
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        // Red on top spanning cols 0-3 at the bottom
        // Let me redo: simpler 2-chain
        let mut board = Board::new();
        // Col 0: R R R B
        // Col 1: B B B B  <- will clear when gravity drops B from col 0
        // No, let me think more carefully.

        // 2-chain: first clear triggers gravity which creates second clear.
        // Col 0: G G G
        // Col 1: G R R R
        // Col 2: R
        // Col 3: R
        // When 4 R clears (row 0 of cols 1,2,3 + ?), then G falls...
        // Simpler approach:
        // Col 0 bottom-up: B B B R R R R
        // Clear: 4 R clears (rows 3-6), then B B B falls, but only 3 B's - not enough.
        // Need to set up properly:

        // Vertical 2-chain:
        // Col 0: B B B   (rows 0,1,2)
        // Col 1: B R R R R (rows 0,1,2,3,4) - wait cols don't work like that
        // Let me just use a known working pattern:
        // Col 0 bottom: R R R, on top: G G G G  <- when G clears, nothing new
        // That's just 1 chain.

        // Real 2-chain: after first clear, gravity makes new group of 4.
        // Col 0: B B B (rows 0,1,2)
        // Col 1: B     (row 0)
        // On top of col 0's B B B, put R R R R vertically (rows 3,4,5,6)
        // After R clears, nothing new happens. Not a 2-chain.

        // Proper 2-chain:
        // Col 0: R R R   (rows 0,1,2)
        // Col 1: G R     (rows 0,1)
        // Col 2: G       (row 0)
        // Col 3: G       (row 0)
        // Col 4: G       (row 0) - now 4 G's? No, (1,0),(2,0),(3,0),(4,0) = 4 G's horizontal
        // But also need R to chain. Let me just use:

        // Set up: 4 Red in a column clears, then 4 Blue are exposed
        // Col 0: B B B B R R R R  (blue at bottom, red on top)
        // Red (rows 4,5,6,7) clears -> board has B B B B which is 4 -> clears = 2 chain!
        let mut board = Board::new();
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let result = resolve_chains(&mut board);
        assert_eq!(result.chain_count, 2);
        assert_eq!(board.column_height(0), 0);
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
}
