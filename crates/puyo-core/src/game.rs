use crate::board::{Board, ChainResult, PuyoColor, COLS, NUM_COLORS, ROWS};
use crate::piece::{FallingPiece, Orientation, Piece, Placement};
use crate::rng::Rng;

/// Phase of the game state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    /// Player is controlling the falling piece.
    Falling,
    /// Piece has landed, resolving chains.
    Resolving,
    /// Game is over.
    GameOver,
}

impl GamePhase {
    /// Convert to integer representation (0=Falling, 1=Resolving, 2=GameOver).
    pub fn as_u8(&self) -> u8 {
        match self {
            GamePhase::Falling => 0,
            GamePhase::Resolving => 1,
            GamePhase::GameOver => 2,
        }
    }
}

/// Main game state.
#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Board,
    pub current_piece: Option<FallingPiece>,
    pub next_piece: Piece,
    pub next_next_piece: Piece,
    pub score: u32,
    pub max_chain: u32,
    pub phase: GamePhase,
    pub rng: Rng,
    pub total_pieces: u32,
}

impl GameState {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let current = Self::generate_piece(&mut rng);
        let next = Self::generate_piece(&mut rng);
        let next_next = Self::generate_piece(&mut rng);

        let mut state = GameState {
            board: Board::new(),
            current_piece: None,
            next_piece: next,
            next_next_piece: next_next,
            score: 0,
            max_chain: 0,
            phase: GamePhase::Falling,
            rng,
            total_pieces: 0,
        };

        state.spawn_piece(current);
        state
    }

    fn generate_piece(rng: &mut Rng) -> Piece {
        let axis = PuyoColor::from_u8(rng.next_range(NUM_COLORS as u32) as u8 + 1);
        let satellite = PuyoColor::from_u8(rng.next_range(NUM_COLORS as u32) as u8 + 1);
        Piece::new(axis, satellite)
    }

    /// Spawn a new piece at the top.
    fn spawn_piece(&mut self, piece: Piece) {
        self.current_piece = Some(FallingPiece::spawn(piece));
    }

    /// Advance to the next piece.
    fn advance_piece(&mut self) {
        self.total_pieces += 1;
        let next = self.next_piece;
        self.next_piece = self.next_next_piece;
        self.next_next_piece = Self::generate_piece(&mut self.rng);
        self.spawn_piece(next);
    }

    /// Move current piece left.
    pub fn move_left(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_move_left(&self.board)
        } else {
            false
        }
    }

    /// Move current piece right.
    pub fn move_right(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_move_right(&self.board)
        } else {
            false
        }
    }

    /// Rotate current piece clockwise.
    pub fn rotate_cw(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_rotate_cw(&self.board)
        } else {
            false
        }
    }

    /// Rotate current piece counter-clockwise.
    pub fn rotate_ccw(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            fp.try_rotate_ccw(&self.board)
        } else {
            false
        }
    }

    /// Hard drop the current piece. Returns chain result if chains occurred.
    pub fn hard_drop(&mut self) -> Option<ChainResult> {
        if self.phase != GamePhase::Falling {
            return None;
        }

        let fp = self.current_piece.take()?;
        let placement = Placement::new(fp.col, fp.orientation);

        self.place_piece(&fp.piece, &placement);
        let result = self.resolve();
        Some(result)
    }

    /// Place a piece on the board at the given placement.
    pub fn place_piece(&mut self, piece: &Piece, placement: &Placement) {
        let (dc, _dr) = placement.orientation.offset();

        match placement.orientation {
            Orientation::North => {
                // Axis first (bottom), then satellite on top
                self.board.drop_puyo(placement.col, piece.axis_color);
                let sat_col = (placement.col as i32 + dc) as usize;
                let sat_h = self.board.column_height(sat_col);
                // Defend against overwriting an isolated puyo at the top hidden row (ROWS-1)
                if sat_h < ROWS && !self.board.get(sat_col, sat_h).is_color() {
                    self.board.drop_puyo(sat_col, piece.satellite_color);
                }
            }
            Orientation::South => {
                // Satellite first (bottom), then axis on top
                let sat_col = (placement.col as i32 + dc) as usize;
                self.board.drop_puyo(sat_col, piece.satellite_color);
                self.board.drop_puyo(placement.col, piece.axis_color);
            }
            Orientation::East | Orientation::West => {
                // Side by side - drop both independently
                self.board.drop_puyo(placement.col, piece.axis_color);
                let sat_col = (placement.col as i32 + dc) as usize;
                self.board.drop_puyo(sat_col, piece.satellite_color);
            }
        }
    }

    /// Resolve chains after placing a piece.
    fn resolve(&mut self) -> ChainResult {
        self.phase = GamePhase::Resolving;

        let result = self.board.resolve_chains();
        self.score += result.score;
        if result.chain_count > self.max_chain {
            self.max_chain = result.chain_count;
        }

        // Check game over
        if self.board.is_game_over() {
            self.phase = GamePhase::GameOver;
        } else {
            self.phase = GamePhase::Falling;
            self.advance_piece();
        }

        result
    }

    /// Place a piece on the board without resolving chains or advancing to next piece.
    /// Used by the training loop to control chain resolution step by step.
    /// Returns true if a piece was placed, false if no current piece.
    pub fn place_piece_only(&mut self, placement: &Placement) -> bool {
        if let Some(fp) = self.current_piece.take() {
            self.place_piece(&fp.piece, placement);
            self.phase = GamePhase::Resolving;
            true
        } else {
            false
        }
    }

    /// Finalize game state after manual chain resolution.
    /// Updates score/max_chain, checks game over, advances to next piece.
    pub fn finalize_after_chains(&mut self, total_score: u32, max_chain: u32) {
        self.score += total_score;
        if max_chain > self.max_chain {
            self.max_chain = max_chain;
        }
        if self.board.is_game_over() {
            self.phase = GamePhase::GameOver;
        } else {
            self.phase = GamePhase::Falling;
            self.advance_piece();
        }
    }

    /// Apply a placement directly (used by AI). Returns chain result.
    pub fn apply_placement(&mut self, placement: &Placement) -> ChainResult {
        if let Some(fp) = self.current_piece.take() {
            self.place_piece(&fp.piece, placement);
            self.resolve()
        } else {
            ChainResult {
                chain_count: 0,
                score: 0,
            }
        }
    }

    /// Soft drop: move piece down by one step. Returns true if moved, false if landed.
    pub fn soft_drop(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            let axis_col = fp.col;
            let (dc, _dr) = fp.orientation.offset();
            let sat_col = (axis_col as i32 + dc).max(0) as usize;

            let axis_height = self.board.column_height(axis_col);
            let sat_height = self.board.column_height(sat_col);

            // The piece can't go below the highest column it occupies
            let min_row = match fp.orientation {
                Orientation::North => axis_height as f32, // axis on bottom
                Orientation::South => (axis_height + 1) as f32, // satellite lands at column_height, axis one above
                _ => axis_height.max(sat_height) as f32,        // side by side
            };

            if fp.row - 1.0 < min_row {
                // Would land
                false
            } else {
                fp.row -= 1.0;
                true
            }
        } else {
            false
        }
    }

    /// Tick: advance the game by one frame's worth of gravity.
    pub fn tick(&mut self, gravity: f32) -> Option<ChainResult> {
        if self.phase != GamePhase::Falling {
            return None;
        }

        let fp = self.current_piece.as_ref()?;
        let landing_row = self.landing_row_for(fp);
        let new_row = fp.row - gravity;

        if new_row <= landing_row {
            self.hard_drop()
        } else {
            self.current_piece.as_mut()?.row = new_row;
            None
        }
    }

    /// Compute the landing row for a falling piece based on current board state.
    fn landing_row_for(&self, fp: &FallingPiece) -> f32 {
        let axis_col = fp.col;
        let (dc, _) = fp.orientation.offset();
        let sat_col = (axis_col as i32 + dc).clamp(0, (COLS - 1) as i32) as usize;

        let axis_height = self.board.column_height(axis_col);
        let sat_height = self.board.column_height(sat_col);

        match fp.orientation {
            Orientation::North => axis_height as f32,
            Orientation::South => (sat_height + 1) as f32,
            Orientation::East | Orientation::West => axis_height.max(sat_height) as f32,
        }
    }

    /// Restart the game with a new seed.
    pub fn restart(&mut self, seed: u64) {
        *self = GameState::new(seed);
    }

    /// Get current piece info for rendering: (axis_color, sat_color, col, row, orientation_index)
    pub fn get_current_piece_info(&self) -> Option<(u8, u8, u8, f32, u8)> {
        self.current_piece.as_ref().map(|fp| {
            (
                fp.piece.axis_color as u8,
                fp.piece.satellite_color as u8,
                fp.col as u8,
                fp.row,
                fp.orientation.as_u8(),
            )
        })
    }

    /// Get next piece info: (axis_color, satellite_color)
    pub fn get_next_piece_info(&self) -> (u8, u8) {
        (
            self.next_piece.axis_color as u8,
            self.next_piece.satellite_color as u8,
        )
    }

    /// Get next-next piece info: (axis_color, satellite_color)
    pub fn get_next_next_piece_info(&self) -> (u8, u8) {
        (
            self.next_next_piece.axis_color as u8,
            self.next_next_piece.satellite_color as u8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::SPAWN_COL;

    #[test]
    fn test_new_game() {
        let game = GameState::new(42);
        assert_eq!(game.phase, GamePhase::Falling);
        assert!(game.current_piece.is_some());
        assert_eq!(game.score, 0);
        assert_eq!(game.max_chain, 0);
    }

    #[test]
    fn test_hard_drop_and_next() {
        let mut game = GameState::new(42);
        game.hard_drop();
        // After hard drop, current piece should be what was next
        if game.phase == GamePhase::Falling {
            assert!(game.current_piece.is_some());
        }
    }

    #[test]
    fn test_deterministic_game() {
        let mut game1 = GameState::new(42);
        let mut game2 = GameState::new(42);

        for _ in 0..10 {
            if game1.phase != GamePhase::Falling || game2.phase != GamePhase::Falling {
                break;
            }
            game1.hard_drop();
            game2.hard_drop();
            assert_eq!(game1.score, game2.score);
            assert_eq!(game1.board, game2.board);
        }
    }

    #[test]
    fn test_move_operations() {
        let mut game = GameState::new(42);
        assert_eq!(game.current_piece.as_ref().unwrap().col, SPAWN_COL);

        game.move_left();
        assert_eq!(game.current_piece.as_ref().unwrap().col, SPAWN_COL - 1);

        game.move_right();
        assert_eq!(game.current_piece.as_ref().unwrap().col, SPAWN_COL);

        game.rotate_cw();
        assert_eq!(
            game.current_piece.as_ref().unwrap().orientation,
            Orientation::East
        );
    }

    #[test]
    fn test_restart() {
        let mut game = GameState::new(42);
        game.hard_drop();
        game.hard_drop();
        game.restart(42);
        assert_eq!(game.score, 0);
        assert_eq!(game.max_chain, 0);
        assert_eq!(game.phase, GamePhase::Falling);
    }

    #[test]
    fn test_soft_drop_south_orientation_landing() {
        let mut game = GameState::new(42);
        // Pre-fill column 2 with 3 puyos
        game.board.drop_puyo(2, PuyoColor::Red);
        game.board.drop_puyo(2, PuyoColor::Blue);
        game.board.drop_puyo(2, PuyoColor::Green);
        assert_eq!(game.board.column_height(2), 3);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Blue, PuyoColor::Red),
            col: 2,
            row: 7.0,
            orientation: Orientation::South,
        });

        while game.soft_drop() {}

        // axis should stop at column_height + 1 = 4
        // satellite at fp.row - 1 = 3 (first free row)
        let fp = game.current_piece.as_ref().unwrap();
        assert_eq!(fp.row, 4.0);
    }

    #[test]
    fn test_tick_south_orientation_landing() {
        let mut game = GameState::new(42);
        game.board.drop_puyo(2, PuyoColor::Red);
        game.board.drop_puyo(2, PuyoColor::Blue);
        assert_eq!(game.board.column_height(2), 2);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Blue, PuyoColor::Green),
            col: 2,
            row: 4.0,
            orientation: Orientation::South,
        });
        game.phase = GamePhase::Falling;

        // Tick with gravity that would bring axis below landing row
        let _result = game.tick(1.5);

        // After landing: satellite (Green) at row 2, axis (Blue) at row 3
        assert_eq!(game.board.column_height(2), 4);
        assert_eq!(game.board.get(2, 2), PuyoColor::Green);
        assert_eq!(game.board.get(2, 3), PuyoColor::Blue);
    }

    #[test]
    fn test_south_orientation_empty_column() {
        let mut game = GameState::new(42);
        assert_eq!(game.board.column_height(2), 0);

        game.current_piece = Some(FallingPiece {
            piece: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            col: 2,
            row: 6.0,
            orientation: Orientation::South,
        });

        while game.soft_drop() {}

        // Empty column: satellite at row 0, axis at row 1
        let fp = game.current_piece.as_ref().unwrap();
        assert_eq!(fp.row, 1.0);
    }

    #[test]
    fn test_place_piece_north_defends_top_hidden_row_isolated() {
        use crate::board::{ROWS, VISIBLE_ROWS};
        let mut game = GameState::new(42);
        // Fill col 0 to full visible height (6)
        for _ in 0..VISIBLE_ROWS {
            game.board.drop_puyo(0, PuyoColor::Red);
        }
        // Place isolated puyo at top hidden row
        game.board.set(0, ROWS - 1, PuyoColor::Green);

        let piece = Piece::new(PuyoColor::Blue, PuyoColor::Blue);
        let placement = Placement::new(0, Orientation::North);
        game.place_piece(&piece, &placement);

        // Axis (Blue) should be placed at row VISIBLE_ROWS
        assert_eq!(game.board.get(0, VISIBLE_ROWS), PuyoColor::Blue);
        // Top hidden row should still be the original isolated Green (satellite skipped)
        assert_eq!(game.board.get(0, ROWS - 1), PuyoColor::Green);
    }
}
