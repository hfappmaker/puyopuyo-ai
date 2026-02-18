use crate::board::PuyoColor;

/// Orientation of the satellite puyo relative to the axis puyo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    North, // satellite above axis
    East,  // satellite right of axis
    South, // satellite below axis
    West,  // satellite left of axis
}

impl Orientation {
    /// Rotate clockwise.
    pub fn rotate_cw(self) -> Self {
        match self {
            Orientation::North => Orientation::East,
            Orientation::East => Orientation::South,
            Orientation::South => Orientation::West,
            Orientation::West => Orientation::North,
        }
    }

    /// Rotate counter-clockwise.
    pub fn rotate_ccw(self) -> Self {
        match self {
            Orientation::North => Orientation::West,
            Orientation::West => Orientation::South,
            Orientation::South => Orientation::East,
            Orientation::East => Orientation::North,
        }
    }

    /// Get the (dcol, drow) offset for the satellite relative to axis.
    pub fn offset(self) -> (i32, i32) {
        match self {
            Orientation::North => (0, 1),
            Orientation::East => (1, 0),
            Orientation::South => (0, -1),
            Orientation::West => (-1, 0),
        }
    }
}

/// A two-puyo piece (tsumo). axis_color is the pivot, satellite_color orbits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub axis_color: PuyoColor,
    pub satellite_color: PuyoColor,
}

impl Piece {
    pub fn new(axis: PuyoColor, satellite: PuyoColor) -> Self {
        Piece {
            axis_color: axis,
            satellite_color: satellite,
        }
    }
}

/// A specific placement: which column the axis lands in and the orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    pub col: usize,        // axis column (0-5)
    pub orientation: Orientation,
}

impl Placement {
    pub fn new(col: usize, orientation: Orientation) -> Self {
        Placement { col, orientation }
    }

    /// Get the satellite column given this placement.
    /// Returns None if out of bounds.
    pub fn satellite_col(&self) -> Option<usize> {
        let (dc, _) = self.orientation.offset();
        let sc = self.col as i32 + dc;
        if sc < 0 || sc >= 6 {
            None
        } else {
            Some(sc as usize)
        }
    }
}

/// The falling piece state during gameplay.
#[derive(Debug, Clone)]
pub struct FallingPiece {
    pub piece: Piece,
    pub col: usize,          // axis column
    pub row: f32,            // axis row (fractional for smooth fall)
    pub orientation: Orientation,
}

impl FallingPiece {
    pub fn spawn(piece: Piece) -> Self {
        FallingPiece {
            piece,
            col: 2,       // spawn at column 2 (3rd from left)
            row: 12.0,    // spawn above visible area
            orientation: Orientation::North,
        }
    }

    /// Get the satellite position (col, row_offset).
    pub fn satellite_pos(&self) -> (i32, i32) {
        let (dc, dr) = self.orientation.offset();
        (self.col as i32 + dc, self.row as i32 + dr)
    }

    /// Try to move left. Returns true if successful.
    pub fn try_move_left(&mut self, col_heights: &[usize; 6]) -> bool {
        let new_col = self.col as i32 - 1;
        if self.can_occupy(new_col, self.row as i32, self.orientation, col_heights) {
            self.col = new_col as usize;
            true
        } else {
            false
        }
    }

    /// Try to move right. Returns true if successful.
    pub fn try_move_right(&mut self, col_heights: &[usize; 6]) -> bool {
        let new_col = self.col as i32 + 1;
        if self.can_occupy(new_col, self.row as i32, self.orientation, col_heights) {
            self.col = new_col as usize;
            true
        } else {
            false
        }
    }

    /// Try to rotate clockwise with wall kick.
    pub fn try_rotate_cw(&mut self, col_heights: &[usize; 6]) -> bool {
        let new_ori = self.orientation.rotate_cw();
        // Try normal rotation
        if self.can_occupy(self.col as i32, self.row as i32, new_ori, col_heights) {
            self.orientation = new_ori;
            return true;
        }
        // Wall kick: try shifting opposite to satellite direction
        let (dc, _) = new_ori.offset();
        let kick_col = self.col as i32 - dc;
        if self.can_occupy(kick_col, self.row as i32, new_ori, col_heights) {
            self.col = kick_col as usize;
            self.orientation = new_ori;
            return true;
        }
        false
    }

    /// Try to rotate counter-clockwise with wall kick.
    pub fn try_rotate_ccw(&mut self, col_heights: &[usize; 6]) -> bool {
        let new_ori = self.orientation.rotate_ccw();
        if self.can_occupy(self.col as i32, self.row as i32, new_ori, col_heights) {
            self.orientation = new_ori;
            return true;
        }
        let (dc, _) = new_ori.offset();
        let kick_col = self.col as i32 - dc;
        if self.can_occupy(kick_col, self.row as i32, new_ori, col_heights) {
            self.col = kick_col as usize;
            self.orientation = new_ori;
            return true;
        }
        false
    }

    /// Check if a piece can occupy the given position.
    fn can_occupy(&self, col: i32, row: i32, ori: Orientation, col_heights: &[usize; 6]) -> bool {
        let (dc, dr) = ori.offset();
        let sc = col + dc;
        let sr = row + dr;

        // Bounds check
        if col < 0 || col >= 6 || sc < 0 || sc >= 6 {
            return false;
        }
        if row < 0 || sr < 0 {
            return false;
        }

        // Collision check: axis puyo must not overlap existing puyos
        if (row as usize) < col_heights[col as usize] {
            return false;
        }
        // Collision check: satellite puyo must not overlap existing puyos
        if (sr as usize) < col_heights[sc as usize] {
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_rotation() {
        assert_eq!(Orientation::North.rotate_cw(), Orientation::East);
        assert_eq!(Orientation::East.rotate_cw(), Orientation::South);
        assert_eq!(Orientation::South.rotate_cw(), Orientation::West);
        assert_eq!(Orientation::West.rotate_cw(), Orientation::North);

        assert_eq!(Orientation::North.rotate_ccw(), Orientation::West);
        assert_eq!(Orientation::West.rotate_ccw(), Orientation::South);
    }

    #[test]
    fn test_placement_satellite_col() {
        let p = Placement::new(2, Orientation::East);
        assert_eq!(p.satellite_col(), Some(3));

        let p = Placement::new(0, Orientation::West);
        assert_eq!(p.satellite_col(), None); // out of bounds

        let p = Placement::new(5, Orientation::East);
        assert_eq!(p.satellite_col(), None); // out of bounds

        let p = Placement::new(3, Orientation::North);
        assert_eq!(p.satellite_col(), Some(3)); // same column
    }

    #[test]
    fn test_falling_piece_spawn() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let fp = FallingPiece::spawn(piece);
        assert_eq!(fp.col, 2);
        assert_eq!(fp.orientation, Orientation::North);
    }

    #[test]
    fn test_move_left_right() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece::spawn(piece);
        let heights = [0; 6];

        assert!(fp.try_move_left(&heights));
        assert_eq!(fp.col, 1);
        assert!(fp.try_move_left(&heights));
        assert_eq!(fp.col, 0);
        assert!(!fp.try_move_left(&heights)); // can't go further left
        assert_eq!(fp.col, 0);

        fp.col = 4;
        fp.orientation = Orientation::North;
        assert!(fp.try_move_right(&heights));
        assert_eq!(fp.col, 5);
        assert!(!fp.try_move_right(&heights)); // at right edge with North, satellite is same col, should be ok
        // Actually with North orientation, satellite is above, so col 5 should still allow right... let me check
        // col=5, orientation=North: axis at 5, satellite at 5. Move right would put axis at 6 = out of bounds
    }

    #[test]
    fn test_rotate_wall_kick() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece::spawn(piece);
        let heights = [0; 6];

        // At col 0, orientation North, rotate CW -> East would put satellite at col 1 (ok)
        fp.col = 0;
        fp.orientation = Orientation::North;
        assert!(fp.try_rotate_cw(&heights));
        assert_eq!(fp.orientation, Orientation::East);

        // At col 0, orientation East, rotate CW -> South (ok, satellite below)
        assert!(fp.try_rotate_cw(&heights));
        assert_eq!(fp.orientation, Orientation::South);

        // At col 0, orientation South, rotate CW -> West would put satellite at col -1
        // Wall kick should shift axis to col 1
        assert!(fp.try_rotate_cw(&heights));
        assert_eq!(fp.orientation, Orientation::West);
    }

    #[test]
    fn test_move_blocked_by_existing_puyos() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 3,
            row: 2.0,
            orientation: Orientation::North,
        };

        // Column 2 has height 5 — blocks leftward move at row 2
        let mut heights = [0; 6];
        heights[2] = 5;

        assert!(!fp.try_move_left(&heights)); // col 2 has puyos at row 2
        assert_eq!(fp.col, 3); // didn't move

        // Column 4 has height 5 — blocks rightward move at row 2
        heights[4] = 5;
        assert!(!fp.try_move_right(&heights));
        assert_eq!(fp.col, 3);

        // If piece is above the height, movement is allowed
        fp.row = 6.0;
        assert!(fp.try_move_left(&heights));
        assert_eq!(fp.col, 2);
    }

    #[test]
    fn test_move_blocked_by_satellite_collision() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 2,
            row: 2.0,
            orientation: Orientation::East, // satellite at col 3
        };

        // Column 1 is empty, but column 2 (where satellite would be) is fine
        // Moving left: axis -> col 1, satellite -> col 2. Both clear.
        let heights = [0; 6];
        assert!(fp.try_move_left(&heights));
        assert_eq!(fp.col, 1);

        // Reset and block the satellite's target column
        fp.col = 2;
        let mut heights = [0; 6];
        heights[4] = 5; // col 4 blocked

        // Moving right: axis -> col 3, satellite -> col 4. Col 4 is blocked at row 2.
        assert!(!fp.try_move_right(&heights));
        assert_eq!(fp.col, 2);
    }

    #[test]
    fn test_rotate_blocked_by_existing_puyos() {
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let mut fp = FallingPiece {
            piece,
            col: 3,
            row: 2.0,
            orientation: Orientation::North,
        };

        // Block col 4 so rotating CW (North->East, satellite to col 4) fails
        // Also block col 2 so wall kick (axis to col 2) fails too
        let mut heights = [0; 6];
        heights[4] = 5;
        heights[2] = 5;

        assert!(!fp.try_rotate_cw(&heights));
        assert_eq!(fp.orientation, Orientation::North); // unchanged
    }
}
