use game_core::Game;
use puyo_core::board::{Board, ChainResult, COLS, NUM_COLORS, ROWS};
use puyo_core::piece::{Piece, Placement};
use puyo_core::puyo_game::{
    board_to_tensor_data, context_to_tensor_data, PuyoState, CONTEXT_TENSOR_SIZE, NUM_CHANNELS,
};

use crate::hash_util::splitmix64;
use crate::placement::{
    compute_valid_mask, enumerate_placements, index_to_placement, placement_to_index,
    simulate_placement, NUM_ACTIONS,
};

/// ぷよぷよゲームの Game trait 実装。
#[derive(Clone)]
pub struct PuyoGame;

impl Game for PuyoGame {
    type State = PuyoState;
    type Action = Placement;
    type ActionResult = ChainResult;

    fn num_actions() -> usize {
        NUM_ACTIONS
    }

    fn board_tensor_shape() -> (usize, usize, usize) {
        (NUM_CHANNELS, ROWS, COLS)
    }

    fn context_tensor_size() -> usize {
        CONTEXT_TENSOR_SIZE
    }

    fn is_terminal(state: &PuyoState) -> bool {
        state.board.is_game_over()
    }

    fn legal_actions(state: &PuyoState) -> Vec<Placement> {
        enumerate_placements(&state.board, &state.current)
    }

    fn valid_action_mask(state: &PuyoState) -> Vec<bool> {
        compute_valid_mask(&state.board, &state.current).to_vec()
    }

    fn action_to_index(action: &Placement) -> usize {
        placement_to_index(action)
    }

    fn index_to_action(index: usize) -> Placement {
        index_to_placement(index)
    }

    fn apply_action(state: &PuyoState, action: &Placement) -> (PuyoState, ChainResult) {
        let (new_board, chain_result) = simulate_placement(&state.board, &state.current, action);
        let new_state = PuyoState {
            board: new_board,
            current: state.next,
            next: state.next_next,
            // next_next は advance_turn で設定される。暫定的に next_next をコピー。
            next_next: state.next_next,
        };
        (new_state, chain_result)
    }

    fn reward(result: &ChainResult) -> f32 {
        result.score as f32
    }

    fn encode_board(state: &PuyoState) -> Vec<f32> {
        board_to_tensor_data(&state.board)
    }

    fn encode_context(state: &PuyoState) -> Vec<f32> {
        context_to_tensor_data(&state.current, &state.next, &state.next_next)
    }

    fn advance_turn(state: &mut PuyoState, seed1: u64, seed2: u64, seed3: u64) {
        let piece = sample_piece(seed1, seed2, seed3);
        state.next_next = piece;
    }

    fn state_hash(state: &PuyoState) -> u64 {
        board_hash(&state.board)
    }
}

/// Board のFNV-1aハッシュを計算する。
fn board_hash(board: &Board) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for col in 0..COLS {
        for row in 0..ROWS {
            h ^= board.columns[col][row] as u8 as u64;
            h = h.wrapping_mul(0x00000100000001B3);
        }
    }
    h
}

/// 決定的にランダムなピースを生成する。
fn sample_piece(seed1: u64, seed2: u64, seed3: u64) -> Piece {
    let mut x = splitmix64(
        seed1
            .wrapping_mul(6364136223846793005)
            .wrapping_add(seed2)
            .wrapping_add(1)
            .wrapping_add(seed3.wrapping_mul(0x9e3779b97f4a7c15)),
    );

    let axis = ((x % NUM_COLORS as u64) as u8) + 1;
    x = (x ^ (x >> 30)).wrapping_mul(0x517cc1b727220a95);
    x = x ^ (x >> 27);
    let sat = ((x % NUM_COLORS as u64) as u8) + 1;
    Piece::new(
        puyo_core::board::PuyoColor::from_u8(axis),
        puyo_core::board::PuyoColor::from_u8(sat),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;
    use puyo_core::piece::Orientation;

    #[test]
    fn test_puyo_game_num_actions() {
        assert_eq!(PuyoGame::num_actions(), COLS * 4);
    }

    #[test]
    fn test_puyo_game_apply_action() {
        let state = PuyoState {
            board: Board::new(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let action = Placement::new(0, Orientation::North);
        let (new_state, result) = PuyoGame::apply_action(&state, &action);
        assert!(!PuyoGame::is_terminal(&new_state));
        assert_eq!(result.score, 0);
        // current が next に進んでいること
        assert_eq!(new_state.current, state.next);
    }

    #[test]
    fn test_puyo_game_encode_board_size() {
        let state = PuyoState {
            board: Board::new(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let tensor = PuyoGame::encode_board(&state);
        let (ch, h, w) = PuyoGame::board_tensor_shape();
        assert_eq!(tensor.len(), ch * h * w);
    }

    #[test]
    fn test_puyo_game_encode_context_size() {
        let state = PuyoState {
            board: Board::new(),
            current: Piece::new(PuyoColor::Red, PuyoColor::Blue),
            next: Piece::new(PuyoColor::Green, PuyoColor::Blue),
            next_next: Piece::new(PuyoColor::Blue, PuyoColor::Red),
        };
        let tensor = PuyoGame::encode_context(&state);
        assert_eq!(tensor.len(), PuyoGame::context_tensor_size());
    }

    #[test]
    fn test_sample_piece_deterministic() {
        let p1 = sample_piece(42, 7, 0);
        let p2 = sample_piece(42, 7, 0);
        assert_eq!(p1.axis_color, p2.axis_color);
        assert_eq!(p1.satellite_color, p2.satellite_color);
    }
}
