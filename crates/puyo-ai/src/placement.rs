use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Orientation, Piece, Placement};

/// Enumerate all legal placements for a piece on the given board.
/// Maximum 22 placements: North/South × 6 cols + East × 5 cols + West × 5 cols.
pub fn enumerate_placements(board: &Board, piece: &Piece) -> Vec<Placement> {
    let mut placements = Vec::with_capacity(22);

    // North: axis on bottom, satellite above. Axis can be in cols 0-5.
    for col in 0..COLS {
        if board.column_height(col) + 2 <= ROWS {
            placements.push(Placement::new(col, Orientation::North));
        }
    }

    // South: satellite on bottom, axis above. Axis can be in cols 0-5.
    for col in 0..COLS {
        if board.column_height(col) + 2 <= ROWS {
            placements.push(Placement::new(col, Orientation::South));
        }
    }

    // East: satellite to the right. Axis in cols 0-4.
    for col in 0..(COLS - 1) {
        let h1 = board.column_height(col);
        let h2 = board.column_height(col + 1);
        if h1 < ROWS && h2 < ROWS {
            placements.push(Placement::new(col, Orientation::East));
        }
    }

    // West: satellite to the left. Axis in cols 1-5.
    for col in 1..COLS {
        let h1 = board.column_height(col);
        let h2 = board.column_height(col - 1);
        if h1 < ROWS && h2 < ROWS {
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
        // Col 0 is full, so North/South at col 0 are excluded.
        // East at col 0 excluded (col 0 full). West at col 1 excluded (col 0 full).
        // Remaining: North 5 + South 5 + East 4 + West 4 = 18
        assert_eq!(placements.len(), 18);
    }
}
