use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Orientation, Piece, Placement};

/// Compute which columns are reachable from the spawn column (col 2).
/// A column with height >= ROWS - 1 blocks entry and further traversal.
fn compute_reachable_columns(board: &Board) -> [bool; COLS] {
    const SPAWN_COL: usize = 2;
    let mut reachable = [false; COLS];

    if board.column_height(SPAWN_COL) >= ROWS - 1 {
        return reachable;
    }
    reachable[SPAWN_COL] = true;

    // Expand left from spawn
    for col in (0..SPAWN_COL).rev() {
        if board.column_height(col) >= ROWS - 1 {
            break;
        }
        reachable[col] = true;
    }

    // Expand right from spawn
    for col in (SPAWN_COL + 1)..COLS {
        if board.column_height(col) >= ROWS - 1 {
            break;
        }
        reachable[col] = true;
    }

    reachable
}

/// Enumerate all legal placements for a piece on the given board.
/// Maximum 22 placements: North/South × 6 cols + East × 5 cols + West × 5 cols.
///
/// Placement rules:
/// 1. The axis puyo must not land at row ROWS-1 (the topmost hidden row).
/// 2. Each column involved must be reachable from the spawn column (col 2).
///    A column with height >= ROWS-1 blocks traversal.
pub fn enumerate_placements(board: &Board, piece: &Piece) -> Vec<Placement> {
    let mut placements = Vec::with_capacity(22);
    let reachable = compute_reachable_columns(board);

    // North: axis on bottom (row h), satellite above (row h+1). Axis in cols 0-5.
    // h + 2 <= ROWS ensures both fit; this also guarantees axis row h <= ROWS-2.
    for col in 0..COLS {
        if reachable[col] && board.column_height(col) + 2 <= ROWS {
            placements.push(Placement::new(col, Orientation::North));
        }
    }

    // South: satellite on bottom (row h), axis above (row h+1). Axis in cols 0-5.
    // Axis at row h+1 must be < ROWS-1, so h + 2 <= ROWS - 1.
    for col in 0..COLS {
        if reachable[col] && board.column_height(col) + 2 <= ROWS - 1 {
            placements.push(Placement::new(col, Orientation::South));
        }
    }

    // East: axis at col, satellite at col+1. Axis in cols 0-4.
    // Axis at row h1 must be < ROWS-1 (not on topmost hidden row).
    for col in 0..(COLS - 1) {
        let h1 = board.column_height(col);
        let h2 = board.column_height(col + 1);
        if reachable[col] && reachable[col + 1] && h1 < ROWS - 1 && h2 < ROWS {
            placements.push(Placement::new(col, Orientation::East));
        }
    }

    // West: axis at col, satellite at col-1. Axis in cols 1-5.
    // Axis at row h1 must be < ROWS-1 (not on topmost hidden row).
    for col in 1..COLS {
        let h1 = board.column_height(col);
        let h2 = board.column_height(col - 1);
        if reachable[col] && reachable[col - 1] && h1 < ROWS - 1 && h2 < ROWS {
            placements.push(Placement::new(col, Orientation::West));
        }
    }

    // Deduplicate: if both colors are the same, North==South and East(col)==West(col+1).
    if piece.axis_color == piece.satellite_color {
        let mut deduped = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for p in &placements {
            let key = normalize_placement(p, true);
            if seen.insert(key) {
                deduped.push(*p);
            }
        }
        return deduped;
    }

    placements
}

/// Normalize a placement for deduplication when both colors are the same.
/// Returns (min_col, max_col, is_vertical).
fn normalize_placement(p: &Placement, _same_color: bool) -> (usize, usize, bool) {
    let (dc, _dr) = p.orientation.offset();
    let sat_col = (p.col as i32 + dc) as usize;
    let is_vertical = matches!(p.orientation, Orientation::North | Orientation::South);

    if is_vertical {
        (p.col, p.col, true)
    } else {
        let min_c = p.col.min(sat_col);
        let max_c = p.col.max(sat_col);
        (min_c, max_c, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;

    #[test]
    fn test_empty_board_placements() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);
        // North: 6, South: 6, East: 5, West: 5 = 22
        assert_eq!(placements.len(), 22);
    }

    #[test]
    fn test_same_color_dedup() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let placements = enumerate_placements(&board, &piece);
        // North==South for each col (6 deduped to 6), East(c)==West(c+1) (5 deduped to 5)
        // Total: 6 + 5 = 11
        assert_eq!(placements.len(), 11);
    }

    #[test]
    fn test_full_column_reduces_placements() {
        let mut board = Board::new();
        // Fill column 0 completely
        for _ in 0..ROWS {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);
        // Col 0 is full (height 14 >= 13), unreachable and blocked by height checks.
        // Remaining: North 5 + South 5 + East 4 + West 4 = 18
        assert_eq!(placements.len(), 18);
    }

    #[test]
    fn test_south_blocked_at_height_12() {
        let mut board = Board::new();
        // Fill column 3 to height 12
        for _ in 0..12 {
            board.drop_puyo(3, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 3: h=12, h+2=14 > 13=ROWS-1 → blocked (axis would be at row 13)
        assert!(!placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::South));

        // North at col 3: h=12, h+2=14 <= 14=ROWS → allowed (axis at row 12)
        assert!(placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_at_height_11() {
        let mut board = Board::new();
        // Fill column 3 to height 11
        for _ in 0..11 {
            board.drop_puyo(3, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 3: h=11, h+2=13 <= 13=ROWS-1 → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_high_column_blocks_traversal_left() {
        let mut board = Board::new();
        // Fill column 1 to height 13 → blocks access to column 0
        for _ in 0..13 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // Column 0 is unreachable (blocked by col 1)
        assert!(!placements.iter().any(|p| p.col == 0));

        // Column 1 is also unreachable (height >= 13)
        assert!(!placements.iter().any(|p| p.col == 1
            && matches!(p.orientation, Orientation::North | Orientation::South)));

        // Columns 2-5 are reachable
        assert!(placements
            .iter()
            .any(|p| p.col == 2 && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == 5 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_high_column_blocks_traversal_right() {
        let mut board = Board::new();
        // Fill column 4 to height 13 → blocks access to column 5
        for _ in 0..13 {
            board.drop_puyo(4, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // Column 5 is unreachable (blocked by col 4)
        assert!(!placements.iter().any(|p| p.col == 5));

        // Column 4 is also unreachable (height >= 13)
        assert!(!placements.iter().any(|p| p.col == 4
            && matches!(p.orientation, Orientation::North | Orientation::South)));

        // Columns 0-3 are reachable
        assert!(placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_east_west_satellite_reachability() {
        let mut board = Board::new();
        // Fill column 4 to height 13 → col 4 and 5 unreachable
        for _ in 0..13 {
            board.drop_puyo(4, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // East at col 3: satellite at col 4 (unreachable) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::East));

        // West at col 4: axis col 4 (unreachable) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == 4 && p.orientation == Orientation::West));

        // East at col 2: axis col 2, satellite col 3 (both reachable) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 2 && p.orientation == Orientation::East));

        // West at col 3: axis col 3, satellite col 2 (both reachable) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::West));
    }
}
