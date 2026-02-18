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
pub const ROWS: usize = 13; // 12 visible + 1 hidden top row
pub const VISIBLE_ROWS: usize = 12;

/// Board stored in column-major order: columns[col][row].
/// Row 0 is the bottom, row 12 is the top.
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

    /// Drop a puyo into a column. Returns the row it landed on, or None if full.
    pub fn drop_puyo(&mut self, col: usize, color: PuyoColor) -> Option<usize> {
        let h = self.column_height(col);
        if h >= ROWS {
            return None;
        }
        self.columns[col][h] = color;
        Some(h)
    }

    /// Apply gravity: make all puyos fall down to fill gaps.
    pub fn apply_gravity(&mut self) {
        for col in 0..COLS {
            let mut write = 0;
            for read in 0..ROWS {
                if self.columns[col][read].is_color() {
                    self.columns[col][write] = self.columns[col][read];
                    if write != read {
                        self.columns[col][read] = PuyoColor::Empty;
                    }
                    write += 1;
                }
            }
        }
    }

    /// Check if game is over (column 2, the 3rd column, has puyo at row 11, filling all visible rows).
    pub fn is_game_over(&self) -> bool {
        self.column_height(2) >= VISIBLE_ROWS
    }

    /// Flatten board to a Vec<u8> for WASM transfer. Column-major, bottom to top.
    pub fn to_flat(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(COLS * ROWS);
        for col in 0..COLS {
            for row in 0..ROWS {
                data.push(self.columns[col][row] as u8);
            }
        }
        data
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
        assert_eq!(row, Some(0));
        assert_eq!(board.get(0, 0), PuyoColor::Red);
        assert_eq!(board.column_height(0), 1);

        let row = board.drop_puyo(0, PuyoColor::Blue);
        assert_eq!(row, Some(1));
        assert_eq!(board.get(0, 1), PuyoColor::Blue);
        assert_eq!(board.column_height(0), 2);
    }

    #[test]
    fn test_column_full() {
        let mut board = Board::new();
        for _ in 0..ROWS {
            board.drop_puyo(0, PuyoColor::Red);
        }
        assert_eq!(board.drop_puyo(0, PuyoColor::Red), None);
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
        // Fill column 2 to height 11 (not yet game over)
        for _ in 0..VISIBLE_ROWS - 1 {
            board.drop_puyo(2, PuyoColor::Red);
        }
        assert!(!board.is_game_over());
        // One more puyo fills to height 12 -> game over
        board.drop_puyo(2, PuyoColor::Red);
        assert!(board.is_game_over());
    }

    #[test]
    fn test_to_flat() {
        let board = Board::new();
        let flat = board.to_flat();
        assert_eq!(flat.len(), COLS * ROWS);
        assert!(flat.iter().all(|&v| v == 0));
    }
}
