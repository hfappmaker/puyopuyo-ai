use wasm_bindgen::prelude::*;

use puyo_core::game::{GamePhase, GameState};
use puyo_core::piece::Orientation;
use puyo_ai::search;

#[wasm_bindgen]
pub struct WasmGame {
    state: GameState,
}

#[wasm_bindgen]
impl WasmGame {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> WasmGame {
        WasmGame {
            state: GameState::new(seed),
        }
    }

    /// Get the board as a flat Vec<u8>, column-major, bottom to top.
    /// Length = 6 * 14 = 84. Each byte is a PuyoColor (0=empty, 1-4=colors).
    #[wasm_bindgen]
    pub fn get_board(&self) -> Vec<u8> {
        self.state.board.to_flat()
    }

    /// Get current piece info: [axis_color, sat_color, col, row_int, row_frac_x100, orientation]
    /// Returns empty vec if no current piece.
    #[wasm_bindgen]
    pub fn get_current_piece(&self) -> Vec<u8> {
        match self.state.get_current_piece_info() {
            Some((axis, sat, col, row, ori)) => {
                vec![axis, sat, col, row as u8, ((row.fract()) * 100.0) as u8, ori]
            }
            None => vec![],
        }
    }

    /// Get next piece info: [axis_color, sat_color]
    #[wasm_bindgen]
    pub fn get_next_piece(&self) -> Vec<u8> {
        let (axis, sat) = self.state.get_next_piece_info();
        vec![axis, sat]
    }

    /// Get current score.
    #[wasm_bindgen]
    pub fn get_score(&self) -> u32 {
        self.state.score
    }

    /// Get max chain achieved.
    #[wasm_bindgen]
    pub fn get_max_chain(&self) -> u32 {
        self.state.max_chain
    }

    /// Get game phase: 0=Falling, 1=Resolving, 2=GameOver
    #[wasm_bindgen]
    pub fn get_phase(&self) -> u8 {
        match self.state.phase {
            GamePhase::Falling => 0,
            GamePhase::Resolving => 1,
            GamePhase::GameOver => 2,
        }
    }

    /// Get total pieces placed.
    #[wasm_bindgen]
    pub fn get_total_pieces(&self) -> u32 {
        self.state.total_pieces
    }

    /// Move left.
    #[wasm_bindgen]
    pub fn move_left(&mut self) -> bool {
        self.state.move_left()
    }

    /// Move right.
    #[wasm_bindgen]
    pub fn move_right(&mut self) -> bool {
        self.state.move_right()
    }

    /// Rotate clockwise.
    #[wasm_bindgen]
    pub fn rotate_cw(&mut self) -> bool {
        self.state.rotate_cw()
    }

    /// Rotate counter-clockwise.
    #[wasm_bindgen]
    pub fn rotate_ccw(&mut self) -> bool {
        self.state.rotate_ccw()
    }

    /// Hard drop. Returns the chain count (0 if no chain).
    #[wasm_bindgen]
    pub fn hard_drop(&mut self) -> u32 {
        match self.state.hard_drop() {
            Some(result) => result.chain_count,
            None => 0,
        }
    }

    /// Tick game with gravity. Returns chain count if piece landed and chains occurred.
    #[wasm_bindgen]
    pub fn tick(&mut self, gravity: f32) -> u32 {
        match self.state.tick(gravity) {
            Some(result) => result.chain_count,
            None => 0,
        }
    }

    /// AI: compute best move. Returns [col, orientation] or empty if no move.
    /// orientation: 0=North, 1=East, 2=South, 3=West
    #[wasm_bindgen]
    pub fn ai_best_move(&self) -> Vec<u8> {
        if self.state.phase != GamePhase::Falling {
            return vec![];
        }

        let current_piece = match &self.state.current_piece {
            Some(fp) => fp.piece,
            None => return vec![],
        };

        let result = search::find_best_move(
            &self.state.board,
            &current_piece,
            &self.state.next_piece,
        );

        match result {
            Some(r) => {
                let ori = match r.best_placement.orientation {
                    Orientation::North => 0u8,
                    Orientation::East => 1,
                    Orientation::South => 2,
                    Orientation::West => 3,
                };
                vec![r.best_placement.col as u8, ori]
            }
            None => vec![],
        }
    }

    /// AI: compute and immediately apply the best move.
    /// Returns chain count from the placement.
    #[wasm_bindgen]
    pub fn ai_play_move(&mut self) -> u32 {
        if self.state.phase != GamePhase::Falling {
            return 0;
        }

        let current_piece = match &self.state.current_piece {
            Some(fp) => fp.piece,
            None => return 0,
        };

        let result = search::find_best_move(
            &self.state.board,
            &current_piece,
            &self.state.next_piece,
        );

        match result {
            Some(r) => {
                let chain_result = self.state.apply_placement(&r.best_placement);
                chain_result.chain_count
            }
            None => 0,
        }
    }

    /// Restart the game.
    #[wasm_bindgen]
    pub fn restart(&mut self, seed: u64) {
        self.state.restart(seed);
    }
}
