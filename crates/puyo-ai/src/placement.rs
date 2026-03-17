use puyo_core::board::{Board, ChainResult, COLS, ROWS, SPAWN_COL};
use puyo_core::game::GameState;
use puyo_core::piece::{Orientation, Piece, Placement};

/// Simulate placing a piece on a board clone. Returns the resulting board and chain result.
pub fn simulate_placement(board: &Board, piece: &Piece, placement: &Placement) -> (Board, ChainResult) {
    let mut sim = GameState::new(0);
    sim.board = board.clone();
    sim.place_piece(piece, placement);
    let chain_result = sim.board.resolve_chains();
    (sim.board, chain_result)
}

/// Compute which columns are reachable from the spawn column.
/// A column with height >= ROWS - 1 blocks entry and further traversal.
fn compute_reachable_columns(board: &Board) -> [bool; COLS] {
    let mut reachable = [false; COLS];

    if board.column_height(SPAWN_COL) >= ROWS - 1 {
        return reachable;
    }

    let is_passable = |col: &usize| board.column_height(*col) < ROWS - 1;

    for col in std::iter::once(SPAWN_COL)
        .chain((0..SPAWN_COL).rev().take_while(is_passable))
        .chain((SPAWN_COL + 1..COLS).take_while(is_passable))
    {
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
    let reachable = compute_reachable_columns(board);

    // North: axis on bottom, satellite above. h + 2 <= max_rows ensures both fit.
    // If row 13 has an isolated puyo, satellite cannot go there.
    let north = (0..COLS)
        .filter(|&col| {
            let (h, isolated) = board.column_info(col);
            let max_rows = if isolated { ROWS - 1 } else { ROWS };
            reachable[col] && h + 2 <= max_rows
        })
        .map(|col| Placement::new(col, Orientation::North));

    // South: satellite on bottom, axis above. Axis at h+1 must be < ROWS-1.
    // Additionally, South requires rotating through East or West from spawn (North).
    // If both adjacent columns have height >= ROWS-1 (or are walls), rotation is blocked.
    let south = (0..COLS)
        .filter(|&col| {
            if !reachable[col] || board.column_height(col) + 2 > ROWS - 1 {
                return false;
            }
            let left_blocked = col == 0 || board.column_height(col - 1) >= ROWS - 1;
            let right_blocked = col == COLS - 1 || board.column_height(col + 1) >= ROWS - 1;
            !(left_blocked && right_blocked)
        })
        .map(|col| Placement::new(col, Orientation::South));

    // East: axis at col, satellite at col+1. Axis must be < ROWS-1.
    // Satellite column with isolated row 13 puyo has reduced max height.
    let east = (0..COLS - 1)
        .filter(|&col| {
            let (sat_h, sat_isolated) = board.column_info(col + 1);
            let sat_max = if sat_isolated { ROWS - 1 } else { ROWS };
            reachable[col]
                && reachable[col + 1]
                && board.column_height(col) < ROWS - 1
                && sat_h < sat_max
        })
        .map(|col| Placement::new(col, Orientation::East));

    // West: axis at col, satellite at col-1. Axis must be < ROWS-1.
    // Satellite column with isolated row 13 puyo has reduced max height.
    let west = (1..COLS)
        .filter(|&col| {
            let (sat_h, sat_isolated) = board.column_info(col - 1);
            let sat_max = if sat_isolated { ROWS - 1 } else { ROWS };
            reachable[col]
                && reachable[col - 1]
                && board.column_height(col) < ROWS - 1
                && sat_h < sat_max
        })
        .map(|col| Placement::new(col, Orientation::West));

    let placements: Vec<Placement> = north.chain(south).chain(east).chain(west).collect();

    // Deduplicate: if both colors are the same, North==South and East(col)==West(col+1).
    if piece.axis_color == piece.satellite_color {
        let mut seen = std::collections::HashSet::new();
        return placements
            .into_iter()
            .filter(|p| seen.insert(normalize_placement(p)))
            .collect();
    }

    placements
}

/// Total number of possible placement indices (6 cols × 4 orientations).
pub const NUM_ACTIONS: usize = 24;

/// Convert a Placement to a flat index: col * 4 + orientation.as_u8().
pub fn placement_to_index(p: &Placement) -> usize {
    p.col * 4 + p.orientation.as_u8() as usize
}

/// Convert a flat index back to a Placement.
/// Panics if index >= 24.
pub fn index_to_placement(index: usize) -> Placement {
    assert!(index < NUM_ACTIONS, "index out of range: {}", index);
    let col = index / 4;
    let ori = match index % 4 {
        0 => Orientation::North,
        1 => Orientation::East,
        2 => Orientation::South,
        3 => Orientation::West,
        _ => unreachable!(),
    };
    Placement::new(col, ori)
}

/// Compute a valid-action mask from enumerate_placements.
/// Returns [bool; 24] where true = valid placement.
pub fn compute_valid_mask(board: &Board, piece: &Piece) -> [bool; NUM_ACTIONS] {
    let mut mask = [false; NUM_ACTIONS];
    for p in enumerate_placements(board, piece) {
        mask[placement_to_index(&p)] = true;
    }
    mask
}

/// Normalize a placement for deduplication when both colors are the same.
/// Returns (min_col, max_col, is_vertical).
fn normalize_placement(p: &Placement) -> (usize, usize, bool) {
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
        assert!(!placements.iter().any(
            |p| p.col == 1 && matches!(p.orientation, Orientation::North | Orientation::South)
        ));

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
        assert!(!placements.iter().any(
            |p| p.col == 4 && matches!(p.orientation, Orientation::North | Orientation::South)
        ));

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

    #[test]
    fn test_north_blocked_by_isolated_row13() {
        let mut board = Board::new();
        // Fill col 3 to height 12
        for _ in 0..12 {
            board.drop_puyo(3, PuyoColor::Red);
        }
        // Isolated puyo at row 13
        board.set(3, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // North at col 3: h=12, satellite would go to row 13 (occupied) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_north_allowed_without_isolated_row13() {
        let mut board = Board::new();
        // Fill col 3 to height 10 + isolated puyo at row 13
        for _ in 0..10 {
            board.drop_puyo(3, PuyoColor::Red);
        }
        board.set(3, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // North at col 3: h=10, satellite at row 11 (not row 13) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_blocked_when_both_neighbors_height_13() {
        let mut board = Board::new();
        // Fill col 1 and col 3 to height 13
        for _ in 0..13 {
            board.drop_puyo(1, PuyoColor::Red);
            board.drop_puyo(3, PuyoColor::Blue);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 2: both neighbors (col 1, col 3) have height 13
        // → rotation through East or West is blocked → South not reachable
        assert!(!placements
            .iter()
            .any(|p| p.col == 2 && p.orientation == Orientation::South));

        // North at col 2 should still be allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 2 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_when_one_neighbor_height_13() {
        let mut board = Board::new();
        // Fill only col 1 to height 13, col 3 is low
        for _ in 0..13 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 2: col 1 blocked but col 3 is open → can rotate via East
        assert!(placements
            .iter()
            .any(|p| p.col == 2 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_south_blocked_at_boundary_col0() {
        let mut board = Board::new();
        // Fill col 1 to height 13 → col 0 has wall on left, col 1 blocked on right
        for _ in 0..13 {
            board.drop_puyo(1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 0: left is wall (blocked), right is col 1 height 13 (blocked)
        // But col 0 is unreachable anyway (blocked by col 1)
        assert!(!placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_east_west_blocked_by_isolated_row13_satellite() {
        let mut board = Board::new();
        // Fill col 4 to height 13 (row 0-12 filled)
        for _ in 0..13 {
            board.drop_puyo(4, PuyoColor::Red);
        }
        // Isolated puyo at row 13 on col 4
        board.set(4, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // East at col 3: satellite at col 4 (height 13, isolated row 13)
        // sat_max = ROWS - 1 = 13, height 13 < 13 is false → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == 3 && p.orientation == Orientation::East));
    }

    #[test]
    fn test_placement_to_index_roundtrip() {
        for col in 0..6 {
            for ori in [
                Orientation::North,
                Orientation::East,
                Orientation::South,
                Orientation::West,
            ] {
                let p = Placement::new(col, ori);
                let idx = placement_to_index(&p);
                let p2 = index_to_placement(idx);
                assert_eq!(p, p2);
            }
        }
    }

    #[test]
    fn test_compute_valid_mask_empty_board() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mask = compute_valid_mask(&board, &piece);
        let count = mask.iter().filter(|&&v| v).count();
        assert_eq!(count, 22); // same as enumerate_placements
    }

    #[test]
    fn test_compute_valid_mask_same_color() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let mask = compute_valid_mask(&board, &piece);
        let count = mask.iter().filter(|&&v| v).count();
        assert_eq!(count, 11); // deduped
    }
}
