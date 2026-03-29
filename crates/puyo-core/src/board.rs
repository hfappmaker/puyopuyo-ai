use std::collections::VecDeque;

/// Puyo colors. Empty = no puyo in that cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PuyoColor {
    Empty = 0,
    Red = 1,
    Green = 2,
    Blue = 3,
    Yellow = 4,
}

/// All color variants in order (excluding Empty). NUM_COLORS selects the active subset.
const ALL_COLOR_VARIANTS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];

impl PuyoColor {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => PuyoColor::Red,
            2 => PuyoColor::Green,
            3 => PuyoColor::Blue,
            4 => PuyoColor::Yellow,
            _ => PuyoColor::Empty,
        }
    }

    pub fn is_color(self) -> bool {
        self != PuyoColor::Empty
    }

    /// Returns the active color variants based on NUM_COLORS.
    pub fn all_colors() -> &'static [PuyoColor] {
        &ALL_COLOR_VARIANTS[..NUM_COLORS]
    }
}

pub const COLS: usize = 3;
pub const ROWS: usize = 8; // VISIBLE_ROWS + 2 hidden top rows
pub const VISIBLE_ROWS: usize = 6;
/// Spawn column: center of the board (0-indexed). Derived from COLS.
pub const SPAWN_COL: usize = (COLS - 1) / 2;
pub const NUM_COLORS: usize = 3;

/// Minimum number of connected same-color puyos required to clear.
pub const MIN_GROUP_SIZE: usize = 4;

// Compile-time validation of board parameters.
const _: () = assert!(NUM_COLORS >= 1 && NUM_COLORS <= 4, "NUM_COLORS must be 1..=4");
const _: () = assert!(COLS >= 1, "COLS must be >= 1");
const _: () = assert!(ROWS == VISIBLE_ROWS + 2, "ROWS must be VISIBLE_ROWS + 2");

// ---- Chain types ----

/// A connected group of same-color puyos.
#[derive(Debug, Clone)]
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>, // (col, row)
}

/// Result of resolving all chains on a board.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub chain_count: u32,
    pub score: u32,
}

// ---- Board ----

/// Board stored in column-major order: columns[col][row].
/// Row 0 is the bottom, the top hidden row (ROWS-1) is the top (hidden).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    pub columns: [[PuyoColor; ROWS]; COLS],
}

impl Board {
    pub fn new() -> Self {
        Board {
            columns: [[PuyoColor::Empty; ROWS]; COLS],
        }
    }

    /// Returns the height of a column (number of contiguous non-empty cells from bottom).
    /// Bottom-up scan: finds the first empty cell from row 0 upward.
    /// This correctly handles isolated puyos in the top hidden row (ROWS-1) (top hidden row)
    /// that may remain after chain elimination clears cells below them.
    pub fn column_height(&self, col: usize) -> usize {
        self.column_info(col).0
    }

    /// 列の高さと the top hidden row (ROWS-1) 孤立ぷよの有無を同時に返す。
    pub fn column_info(&self, col: usize) -> (usize, bool) {
        let mut height = ROWS;
        for row in 0..ROWS {
            if !self.columns[col][row].is_color() {
                height = row;
                break;
            }
        }
        let isolated = self.columns[col][ROWS - 1].is_color() && !self.columns[col][ROWS - 2].is_color();
        (height, isolated)
    }

    /// Get the color at (col, row).
    pub fn get(&self, col: usize, row: usize) -> PuyoColor {
        self.columns[col][row]
    }

    /// Set the color at (col, row).
    pub fn set(&mut self, col: usize, row: usize, color: PuyoColor) {
        self.columns[col][row] = color;
    }

    /// Drop a puyo into a column. Returns the row it landed on.
    ///
    /// # Panics
    /// 列が満杯（高さ >= ROWS）の場合パニックする。
    /// 呼び出し側が事前に空きを確認すること。
    pub fn drop_puyo(&mut self, col: usize, color: PuyoColor) -> usize {
        let h = self.column_height(col);
        assert!(h < ROWS, "drop_puyo: column {col} is full (height={h})");
        self.columns[col][h] = color;
        h
    }

    /// Apply gravity: make all puyos fall down to fill gaps.
    /// The top hidden row (ROWS-1) (top hidden row) is excluded — puyos there stay until game over.
    pub fn apply_gravity(&mut self) {
        for col in 0..COLS {
            let mut write = 0;
            for read in 0..(ROWS - 1) {
                if self.columns[col][read].is_color() {
                    self.columns[col][write] = self.columns[col][read];
                    if write != read {
                        self.columns[col][read] = PuyoColor::Empty;
                    }
                    write += 1;
                }
            }
            // The top hidden row (ROWS-1) (top hidden row) is not touched by gravity
        }
    }

    /// Check if game is over (spawn column has puyo above visible area).
    pub fn is_game_over(&self) -> bool {
        self.column_height(SPAWN_COL) >= VISIBLE_ROWS
    }

    /// Flatten board to a Vec<u8> for WASM transfer. Column-major, bottom to top.
    pub fn to_flat(&self) -> Vec<u8> {
        self.columns
            .iter()
            .flat_map(|col| col.iter().map(|&c| c as u8))
            .collect()
    }

    // ---- Chain detection & resolution ----

    /// Find all connected same-color groups on the visible board.
    pub fn find_connected_groups(&self) -> Vec<Group> {
        let mut visited = [[false; ROWS]; COLS];
        let mut groups = Vec::new();

        for col in 0..COLS {
            for row in 0..VISIBLE_ROWS {
                if !self.get(col, row).is_color() || visited[col][row] {
                    continue;
                }
                let color = self.get(col, row);

                // BFS flood fill
                let mut queue = VecDeque::new();
                let mut cells = Vec::new();
                queue.push_back((col, row));
                visited[col][row] = true;

                while let Some((c, r)) = queue.pop_front() {
                    cells.push((c, r));
                    for (dc, dr) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let nc = c as i32 + dc;
                        let nr = r as i32 + dr;
                        if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= VISIBLE_ROWS as i32 {
                            continue;
                        }
                        let (nc, nr) = (nc as usize, nr as usize);
                        if !visited[nc][nr] && self.get(nc, nr) == color {
                            visited[nc][nr] = true;
                            queue.push_back((nc, nr));
                        }
                    }
                }

                groups.push(Group { color, cells });
            }
        }

        groups
    }

    /// Find groups of MIN_GROUP_SIZE or larger (clearable groups).
    pub fn find_clearable_groups(&self) -> Vec<Group> {
        self.find_connected_groups()
            .into_iter()
            .filter(|g| g.cells.len() >= MIN_GROUP_SIZE)
            .collect()
    }

    /// Resolve one chain step. Modifies board in-place.
    /// Returns Some(score) if groups were found and cleared, None if no groups exist.
    fn resolve_one_step(&mut self, chain_num: u32) -> Option<u32> {
        let groups = self.find_clearable_groups();
        if groups.is_empty() {
            return None;
        }

        // Remove groups from board
        for group in &groups {
            for &(col, row) in &group.cells {
                self.set(col, row, PuyoColor::Empty);
            }
        }

        let step_score = crate::score::calculate_step_score(chain_num, &groups);
        self.apply_gravity();

        Some(step_score)
    }

    /// Resolve all chains on the board. Modifies board in-place.
    pub fn resolve_chains(&mut self) -> ChainResult {
        let mut chain_count = 0u32;
        let mut score = 0u32;
        for chain_num in 1.. {
            match self.resolve_one_step(chain_num) {
                Some(step_score) => {
                    chain_count += 1;
                    score += step_score;
                }
                None => break,
            }
        }
        ChainResult { chain_count, score }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board_is_empty() {
        let board = Board::new();
        for col in 0..COLS {
            assert_eq!(board.column_height(col), 0);
        }
    }

    #[test]
    fn test_drop_puyo() {
        let mut board = Board::new();
        let row = board.drop_puyo(0, PuyoColor::Red);
        assert_eq!(row, 0);
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);

        let row = board.drop_puyo(0, PuyoColor::Blue);
        assert_eq!(row, 1);
        assert_eq!(board.get(0, 1), PuyoColor::Blue);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    #[should_panic(expected = "column 0 is full")]
    fn test_column_full() {
        let mut board = Board::new();
        for _ in 0..ROWS {
            board.drop_puyo(0, PuyoColor::Red);
        }
        board.drop_puyo(0, PuyoColor::Red);
    }

    #[test]
    fn test_apply_gravity() {
        let mut board = Board::new();
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 2, PuyoColor::Blue); // gap at row 1
        board.set(0, 5, PuyoColor::Green); // bigger gap
        board.apply_gravity();
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.get(0, 1), PuyoColor::Blue);
        assert_eq!(board.get(0, 2), PuyoColor::Green);
        assert_eq!(board.get(0, 3), PuyoColor::Empty);
    }

    #[test]
    fn test_game_over() {
        let mut board = Board::new();
        assert!(!board.is_game_over());
        for _ in 0..VISIBLE_ROWS-1 {
            board.drop_puyo(SPAWN_COL, PuyoColor::Red);
        }
        assert!(!board.is_game_over());
        board.drop_puyo(SPAWN_COL, PuyoColor::Red);
        assert!(board.is_game_over());
    }

    #[test]
    fn test_top_hidden_row_not_affected_by_gravity() {
        let mut board = Board::new();
        board.set(0, ROWS - 1, PuyoColor::Red);
        board.apply_gravity();
        assert_eq!(board.get(0, ROWS - 1), PuyoColor::Red);
        assert_eq!(board.get(0, 0), PuyoColor::Empty);
    }

    #[test]
    fn test_column_height_with_isolated_top_hidden_row() {
        let mut board = Board::new();
        board.set(0, ROWS - 1, PuyoColor::Red);
        assert_eq!(board.column_height(0), 0);
    }

    #[test]
    fn test_column_height_with_stack_and_top_hidden_row() {
        let mut board = Board::new();
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Blue);
        board.set(0, 2, PuyoColor::Green);
        board.set(0, ROWS - 1, PuyoColor::Blue);
        assert_eq!(board.column_height(0), 3);
    }

    #[test]
    fn test_drop_puyo_with_isolated_top_hidden_row() {
        let mut board = Board::new();
        board.set(0, ROWS - 1, PuyoColor::Red);
        let row = board.drop_puyo(0, PuyoColor::Blue);
        assert_eq!(row, 0);
        assert_eq!(board.get(0, 0), PuyoColor::Blue);
    }

    #[test]
    fn test_column_info_isolated() {
        let mut board = Board::new();
        assert_eq!(board.column_info(0), (0, false));
        board.set(0, ROWS - 1, PuyoColor::Red);
        assert_eq!(board.column_info(0), (0, true));
        board.set(0, ROWS - 2, PuyoColor::Blue);
        assert_eq!(board.column_info(0), (0, false));
        assert_eq!(board.column_info(1), (0, false));
    }

    #[test]
    fn test_to_flat() {
        let board = Board::new();
        let flat = board.to_flat();
        assert_eq!(flat.len(), COLS * ROWS);
        assert!(flat.iter().all(|&v| v == 0));
    }

    // ---- Chain tests ----

    #[test]
    fn test_no_chain() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(1, PuyoColor::Blue);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 0);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_single_group_clear() {
        let mut board = Board::new();
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
        assert!(result.score > 0);
        assert_eq!(board.column_height(0), 0);
    }

    #[test]
    fn test_l_shape_group_clear() {
        // With only 3 columns, can't do 4 horizontal. Use L-shape instead.
        let mut board = Board::new();
        board.set(0, 0, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(2, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
    }

    #[test]
    fn test_two_chain() {
        let mut board = Board::new();
        for _ in 0..3 {
            board.drop_puyo(0, PuyoColor::Blue);
        }
        for _ in 0..4 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        board.drop_puyo(1, PuyoColor::Blue);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 2);
        assert_eq!(board.column_height(0), 0);
        assert_eq!(board.column_height(1), 0);
    }

    #[test]
    fn test_gravity_after_clear() {
        let mut board = Board::new();
        for _ in 0..4 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        board.drop_puyo(0, PuyoColor::Green);
        board.resolve_chains();
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_l_shape_group() {
        let mut board = Board::new();
        board.set(0, 0, PuyoColor::Red);
        board.set(0, 1, PuyoColor::Red);
        board.set(1, 0, PuyoColor::Red);
        board.set(2, 0, PuyoColor::Red);
        let groups = board.find_clearable_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].cells.len(), 4);
    }

    #[test]
    fn test_hidden_row_puyo_not_cleared() {
        let mut board = Board::new();
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        board.set(0, VISIBLE_ROWS, PuyoColor::Red);
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 1);
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);
    }

    #[test]
    fn test_hidden_row_puyo_falls_after_clear() {
        let mut board = Board::new();
        for row in 0..4 {
            board.set(0, row, PuyoColor::Red);
        }
        board.set(0, 4, PuyoColor::Green);
        board.set(0, VISIBLE_ROWS, PuyoColor::Green);
        board.resolve_chains();
        assert_eq!(board.get(0, 0), PuyoColor::Green);
        assert_eq!(board.get(0, 1), PuyoColor::Green);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    fn test_hidden_row_only_does_not_clear() {
        let mut board = Board::new();
        for col in 0..COLS {
            board.set(col, VISIBLE_ROWS, PuyoColor::Red);
        }
        // Also place one in the next hidden row to get 4 puyos
        board.set(0, VISIBLE_ROWS + 1, PuyoColor::Red);
        let groups = board.find_clearable_groups();
        assert!(
            groups.is_empty(),
            "Hidden-row-only puyos should not form clearable groups"
        );
        let result = board.resolve_chains();
        assert_eq!(result.chain_count, 0);
    }
}
