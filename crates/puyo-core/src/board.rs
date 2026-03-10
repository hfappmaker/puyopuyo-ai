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
}

pub const COLS: usize = 6;
pub const ROWS: usize = 14; // 12 visible + 2 hidden top rows
pub const VISIBLE_ROWS: usize = 12;
pub const SPAWN_COL: usize = 2;

/// Board stored in column-major order: columns[col][row].
/// Row 0 is the bottom, row 13 is the top (hidden).
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

    /// Returns the height of a column (number of non-empty cells from bottom).
    pub fn column_height(&self, col: usize) -> usize {
        for row in (0..ROWS).rev() {
            if self.columns[col][row].is_color() {
                return row + 1;
            }
        }
        0
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
    /// Row 13 (top hidden row) is excluded — puyos there stay until game over.
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
            // Row 13 (top hidden row) is not touched by gravity
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
        // Fill column 2 to height 12 (visible rows full, not yet game over)
        for _ in 0..VISIBLE_ROWS {
            board.drop_puyo(2, PuyoColor::Red);
        }
        assert!(!board.is_game_over());
        // One more puyo reaches row 12 (13th row, 1st hidden row) -> game over
        board.drop_puyo(2, PuyoColor::Red);
        assert!(board.is_game_over());
    }

    #[test]
    fn test_top_hidden_row_not_affected_by_gravity() {
        let mut board = Board::new();
        // Place a puyo directly at row 13 (top hidden row)
        board.set(0, ROWS - 1, PuyoColor::Red);
        board.apply_gravity();
        // The puyo in row 13 must remain there (does not fall)
        assert_eq!(board.get(0, ROWS - 1), PuyoColor::Red);
        assert_eq!(board.get(0, 0), PuyoColor::Empty);
    }

    #[test]
    fn test_to_flat() {
        let board = Board::new();
        let flat = board.to_flat();
        assert_eq!(flat.len(), COLS * ROWS);
        assert!(flat.iter().all(|&v| v == 0));
    }
}
