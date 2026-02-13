use crate::board::{Board, PuyoColor, COLS};
use crate::chain::{self, ChainResult};
use crate::piece::{FallingPiece, Orientation, Piece, Placement};
use crate::rng::Rng;

const NUM_COLORS: u32 = 4;

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

/// Main game state.
#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Board,
    pub current_piece: Option<FallingPiece>,
    pub next_piece: Piece,
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

        let mut state = GameState {
            board: Board::new(),
            current_piece: None,
            next_piece: next,
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
        let axis = PuyoColor::from_u8(rng.next_range(NUM_COLORS) as u8 + 1);
        let satellite = PuyoColor::from_u8(rng.next_range(NUM_COLORS) as u8 + 1);
        Piece::new(axis, satellite)
    }

    /// Spawn a new piece at the top.
    fn spawn_piece(&mut self, piece: Piece) {
        self.current_piece = Some(FallingPiece::spawn(piece));
        self.total_pieces += 1;
    }

    /// Advance to the next piece.
    fn advance_piece(&mut self) {
        let next = self.next_piece;
        self.next_piece = Self::generate_piece(&mut self.rng);
        self.spawn_piece(next);
    }

    /// Move current piece left.
    pub fn move_left(&mut self) -> bool {
        if self.phase != GamePhase::Falling {
            return false;
        }
        if let Some(ref mut fp) = self.current_piece {
            let heights = self.get_column_heights();
            fp.try_move_left(&heights)
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
            let heights = self.get_column_heights();
            fp.try_move_right(&heights)
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
            let heights = self.get_column_heights();
            fp.try_rotate_cw(&heights)
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
            let heights = self.get_column_heights();
            fp.try_rotate_ccw(&heights)
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
        let (dc, dr) = placement.orientation.offset();

        match placement.orientation {
            Orientation::North => {
                // Axis first (bottom), then satellite on top
                self.board.drop_puyo(placement.col, piece.axis_color);
                let sat_col = (placement.col as i32 + dc) as usize;
                self.board.drop_puyo(sat_col, piece.satellite_color);
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

        let result = chain::resolve_chains(&mut self.board);
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

    /// Apply a placement directly (used by AI). Returns chain result.
    pub fn apply_placement(&mut self, placement: &Placement) -> ChainResult {
        if let Some(fp) = self.current_piece.take() {
            self.place_piece(&fp.piece, placement);
            self.resolve()
        } else {
            ChainResult {
                chain_count: 0,
                score: 0,
                steps: vec![],
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
                Orientation::North => axis_height as f32,       // axis on bottom
                Orientation::South => (axis_height) as f32,     // axis on top, satellite on bottom
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

        if let Some(ref mut fp) = self.current_piece.clone() {
            let axis_col = fp.col;
            let (dc, _) = fp.orientation.offset();
            let sat_col = (axis_col as i32 + dc).max(0).min(5) as usize;

            let axis_height = self.board.column_height(axis_col);
            let sat_height = self.board.column_height(sat_col);

            let landing_row = match fp.orientation {
                Orientation::North => axis_height as f32,
                Orientation::South => {
                    // satellite below: both land based on column heights
                    let sat_land = sat_height;
                    let axis_land = sat_height + 1; // axis is above satellite
                    // But they're in the same column, so satellite lands first
                    sat_land as f32
                }
                Orientation::East | Orientation::West => {
                    axis_height.max(sat_height) as f32
                }
            };

            let new_row = fp.row - gravity;
            if new_row <= landing_row {
                // Piece has landed
                return self.hard_drop();
            } else {
                // Update position
                if let Some(ref mut real_fp) = self.current_piece {
                    real_fp.row = new_row;
                }
                None
            }
        } else {
            None
        }
    }

    /// Restart the game with a new seed.
    pub fn restart(&mut self, seed: u64) {
        *self = GameState::new(seed);
    }

    fn get_column_heights(&self) -> [usize; 6] {
        let mut heights = [0usize; 6];
        for col in 0..COLS {
            heights[col] = self.board.column_height(col);
        }
        heights
    }

    /// Get current piece info for rendering: (axis_color, sat_color, col, row, orientation_index)
    pub fn get_current_piece_info(&self) -> Option<(u8, u8, u8, f32, u8)> {
        self.current_piece.as_ref().map(|fp| {
            let ori = match fp.orientation {
                Orientation::North => 0u8,
                Orientation::East => 1,
                Orientation::South => 2,
                Orientation::West => 3,
            };
            (
                fp.piece.axis_color as u8,
                fp.piece.satellite_color as u8,
                fp.col as u8,
                fp.row,
                ori,
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let first_next = game.next_piece;
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
        // Default spawn column is 2
        assert_eq!(game.current_piece.as_ref().unwrap().col, 2);

        game.move_left();
        assert_eq!(game.current_piece.as_ref().unwrap().col, 1);

        game.move_right();
        assert_eq!(game.current_piece.as_ref().unwrap().col, 2);

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
}
