use puyo_core::board::{Board, ChainResult, COLS, ROWS, SPAWN_COL};
use puyo_core::piece::{Orientation, Piece, Placement};

/// Place a piece directly onto a board (no GameState overhead).
fn place_piece_on_board(board: &mut Board, piece: &Piece, placement: &Placement) {
    let (dc, _dr) = placement.orientation.offset();

    match placement.orientation {
        Orientation::North => {
            board.drop_puyo(placement.col, piece.axis_color);
            let sat_col = (placement.col as i32 + dc) as usize;
            let sat_h = board.column_height(sat_col);
            if sat_h < ROWS && !board.get(sat_col, sat_h).is_color() {
                board.drop_puyo(sat_col, piece.satellite_color);
            }
        }
        Orientation::South => {
            let sat_col = (placement.col as i32 + dc) as usize;
            board.drop_puyo(sat_col, piece.satellite_color);
            board.drop_puyo(placement.col, piece.axis_color);
        }
        Orientation::East | Orientation::West => {
            board.drop_puyo(placement.col, piece.axis_color);
            let sat_col = (placement.col as i32 + dc) as usize;
            board.drop_puyo(sat_col, piece.satellite_color);
        }
    }
}

/// Simulate placing a piece on a board clone. Returns the resulting board and chain result.
pub fn simulate_placement(board: &Board, piece: &Piece, placement: &Placement) -> (Board, ChainResult) {
    let mut b = board.clone();
    place_piece_on_board(&mut b, piece, placement);
    let chain_result = b.resolve_chains();
    (b, chain_result)
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
/// Maximum COLS*2 + (COLS-1)*2 placements: North/South × COLS + East/West × (COLS-1).
///
/// Placement rules:
/// 1. The axis puyo must not land at row ROWS-1 (the topmost hidden row).
/// 2. Each column involved must be reachable from the spawn column (SPAWN_COL).
///    A column with height >= ROWS-1 blocks traversal.
pub fn enumerate_placements(board: &Board, piece: &Piece) -> Vec<Placement> {
    let reachable = compute_reachable_columns(board);

    // North: axis on bottom, satellite above. h + 2 <= max_rows ensures both fit.
    // If the top hidden row (ROWS-1) has an isolated puyo, satellite cannot go there.
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
    // If satellite column has an isolated puyo at the top hidden row, max height is reduced.
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
    // If satellite column has an isolated puyo at the top hidden row, max height is reduced.
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

pub use puyo_core::config::NUM_ACTIONS;

/// Convert a Placement to a flat index: col * 4 + orientation.as_u8().
pub fn placement_to_index(p: &Placement) -> usize {
    p.col * 4 + p.orientation.as_u8() as usize
}

/// Convert a flat index back to a Placement.
/// Panics if index >= NUM_ACTIONS.
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
/// Returns [bool; NUM_ACTIONS] where true = valid placement.
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
    use puyo_core::board::{PuyoColor, VISIBLE_ROWS};

    #[test]
    fn test_empty_board_placements() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);
        // North: COLS, South: COLS, East: COLS-1, West: COLS-1
        let expected = COLS * 2 + (COLS - 1) * 2;
        assert_eq!(placements.len(), expected);
    }

    #[test]
    fn test_same_color_dedup() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let placements = enumerate_placements(&board, &piece);
        // North==South deduped to COLS, East(c)==West(c+1) deduped to COLS-1
        let expected = COLS + (COLS - 1);
        assert_eq!(placements.len(), expected);
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
        // Col 0 is full (height >= ROWS-1), blocks traversal left of spawn.
        // Col 0 unreachable. Remaining cols: COLS-1
        // North (COLS-1) + South (COLS-1) + East (COLS-2) + West (COLS-2, but col 0 blocked)
        let remaining = COLS - 1;
        let expected = remaining * 2 + (remaining - 1).max(0) + (remaining - 1).max(0);
        assert_eq!(placements.len(), expected);
    }

    #[test]
    fn test_south_blocked_at_visible_height() {
        let mut board = Board::new();
        // Fill column to VISIBLE_ROWS height
        for _ in 0..VISIBLE_ROWS {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South: h=VISIBLE_ROWS, h+2 > ROWS-1 → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::South));

        // North: h=VISIBLE_ROWS, h+2 <= ROWS → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_at_height_below_visible() {
        let mut board = Board::new();
        // Fill column to VISIBLE_ROWS - 1
        for _ in 0..VISIBLE_ROWS - 1 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South: h=VISIBLE_ROWS-1, h+2 <= ROWS-1 → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_high_column_blocks_traversal_left() {
        let mut board = Board::new();
        // Fill column 0 to height ROWS-1 → col 0 is impassable and unreachable
        for _ in 0..ROWS - 1 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // Column 0 is unreachable (height >= ROWS-1)
        assert!(!placements.iter().any(
            |p| p.col == 0 && matches!(p.orientation, Orientation::North | Orientation::South)
        ));

        // Columns right of col 0 are still reachable
        assert!(placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_high_column_blocks_traversal_right() {
        let mut board = Board::new();
        // Fill rightmost column to ROWS-1 → blocks access
        for _ in 0..ROWS - 1 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // Rightmost column is unreachable (height >= ROWS-1)
        assert!(!placements.iter().any(
            |p| p.col == COLS - 1 && matches!(p.orientation, Orientation::North | Orientation::South)
        ));

        // Columns left of rightmost are reachable
        assert!(placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::North));
        assert!(placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::North));
    }

    #[test]
    fn test_east_west_satellite_reachability() {
        let mut board = Board::new();
        // Fill rightmost column to ROWS-1 → unreachable
        for _ in 0..ROWS - 1 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // East at spawn col: satellite at COLS-1 (unreachable) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::East));

        // West at spawn col: satellite col 0 (reachable) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::West));

        // East at col 0: satellite at spawn col (reachable) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::East));
    }

    #[test]
    fn test_north_blocked_by_isolated_top_row() {
        let mut board = Board::new();
        // Fill rightmost col to height ROWS-2
        for _ in 0..ROWS - 2 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        // Isolated puyo at row ROWS-1
        board.set(COLS - 1, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // North: satellite would go to ROWS-1 (occupied) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_north_allowed_without_isolated_top_row() {
        let mut board = Board::new();
        // Fill rightmost col to half height + isolated puyo at ROWS-1
        for _ in 0..VISIBLE_ROWS / 2 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        board.set(COLS - 1, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // North: satellite at h+1 (not ROWS-1) → allowed
        assert!(placements
            .iter()
            .any(|p| p.col == COLS - 1 && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_blocked_when_both_neighbors_high() {
        let mut board = Board::new();
        // Fill col 0 and rightmost col to height ROWS-1
        for _ in 0..ROWS - 1 {
            board.drop_puyo(0, PuyoColor::Red);
            board.drop_puyo(COLS - 1, PuyoColor::Blue);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at spawn col: both neighbors high → rotation blocked → South not reachable
        assert!(!placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::South));

        // North at spawn col should still be allowed
        assert!(placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::North));
    }

    #[test]
    fn test_south_allowed_when_one_neighbor_high() {
        let mut board = Board::new();
        // Fill only col 0 to height ROWS-1, col 2 is low
        for _ in 0..ROWS - 1 {
            board.drop_puyo(0, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at spawn col: col 0 blocked but rightmost is open → can rotate via East
        assert!(placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::South));
    }

    #[test]
    fn test_south_blocked_at_boundary_col0() {
        let mut board = Board::new();
        // Fill spawn col to ROWS-1 → col 0 unreachable (blocked by spawn)
        for _ in 0..ROWS - 1 {
            board.drop_puyo(SPAWN_COL, PuyoColor::Red);
        }
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // South at col 0: col 0 is unreachable (spawn col blocked)
        assert!(!placements
            .iter()
            .any(|p| p.col == 0 && p.orientation == Orientation::South));
    }

    #[test]
    fn test_east_west_blocked_by_isolated_top_row_satellite() {
        let mut board = Board::new();
        // Fill rightmost col to ROWS-1 + isolated puyo at ROWS-1
        for _ in 0..ROWS - 1 {
            board.drop_puyo(COLS - 1, PuyoColor::Red);
        }
        board.set(COLS - 1, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let placements = enumerate_placements(&board, &piece);

        // East at spawn col: satellite at COLS-1 (blocked) → blocked
        assert!(!placements
            .iter()
            .any(|p| p.col == SPAWN_COL && p.orientation == Orientation::East));
    }

    #[test]
    fn test_placement_to_index_roundtrip() {
        for col in 0..COLS {
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
        let expected = COLS * 2 + (COLS - 1) * 2;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_compute_valid_mask_same_color() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Red);
        let mask = compute_valid_mask(&board, &piece);
        let count = mask.iter().filter(|&&v| v).count();
        let expected = COLS + (COLS - 1);
        assert_eq!(count, expected);
    }
}
