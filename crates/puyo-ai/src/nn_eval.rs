use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Piece, Placement};
use puyo_nn::encoding::{board_to_tensor_data, NUM_CHANNELS};
use puyo_nn::model::PuyoValueNet;

use crate::eval::{Evaluator, W_GAME_OVER};
use crate::placement::{enumerate_placements, simulate_placement};

type InferBackend = NdArray;

/// Neural network based board evaluator.
pub struct NnEvaluator {
    model: PuyoValueNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
    mean: f32,
    std_dev: f32,
}

impl NnEvaluator {
    pub fn new(
        model: PuyoValueNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
        mean: f32,
        std_dev: f32,
    ) -> Self {
        Self {
            model,
            device,
            mean,
            std_dev,
        }
    }

    fn nn_evaluate(&self, board: &Board) -> f64 {
        if board.is_game_over() {
            return W_GAME_OVER;
        }

        let data = board_to_tensor_data(board);
        let tensor = Tensor::<InferBackend, 1>::from_floats(data.as_slice(), &self.device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let output = self.model.forward(tensor);
        let normalized = match output.into_data().to_vec::<f32>() {
            Ok(v) => v[0],
            Err(_) => return W_GAME_OVER,
        };
        // Denormalize
        (normalized * self.std_dev + self.mean) as f64
    }
}

impl Evaluator for NnEvaluator {
    /// depth-3 → depth-2 → depth-1 フォールバック、連鎖オーバーライドなし。
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        next_next: Option<&Piece>,
    ) -> Option<Placement> {
        let placements = enumerate_placements(board, current);
        if placements.is_empty() {
            return None;
        }

        // depth-3
        if let Some(nn) = next_next {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_placement = placements[0];

            for placement in &placements {
                let (board_after, _) = simulate_placement(board, current, placement);
                if board_after.is_game_over() {
                    continue;
                }

                let next_placements = enumerate_placements(&board_after, next);
                let mut score = f64::NEG_INFINITY;
                for next_placement in &next_placements {
                    let (next_board, _) =
                        simulate_placement(&board_after, next, next_placement);
                    if next_board.is_game_over() {
                        continue;
                    }
                    let nn_placements = enumerate_placements(&next_board, nn);
                    let mut inner_best = f64::NEG_INFINITY;
                    for nn_placement in &nn_placements {
                        let (nn_board, _) =
                            simulate_placement(&next_board, nn, nn_placement);
                        if nn_board.is_game_over() {
                            continue;
                        }
                        let s = self.nn_evaluate(&nn_board);
                        if s > inner_best {
                            inner_best = s;
                        }
                    }
                    if inner_best > score {
                        score = inner_best;
                    }
                }
                let score = score;
                if score > best_score {
                    best_score = score;
                    best_placement = *placement;
                }
            }

            if best_score > f64::NEG_INFINITY {
                return Some(best_placement);
            }
        }

        // depth-2 フォールバック
        {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_placement = placements[0];

            for placement in &placements {
                let (board_after, _) = simulate_placement(board, current, placement);
                if board_after.is_game_over() {
                    continue;
                }

                let next_placements = enumerate_placements(&board_after, next);
                let mut score = f64::NEG_INFINITY;
                for next_placement in &next_placements {
                    let (next_board, _) =
                        simulate_placement(&board_after, next, next_placement);
                    if next_board.is_game_over() {
                        continue;
                    }
                    let s = self.nn_evaluate(&next_board);
                    if s > score {
                        score = s;
                    }
                }
                let score = score;
                if score > best_score {
                    best_score = score;
                    best_placement = *placement;
                }
            }

            if best_score > f64::NEG_INFINITY {
                return Some(best_placement);
            }
        }

        // depth-1 フォールバック
        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for placement in &placements {
            let (board_after, _) = simulate_placement(board, current, placement);
            if board_after.is_game_over() {
                continue;
            }

            let score = self.nn_evaluate(&board_after);
            if score > best_score {
                best_score = score;
                best_placement = *placement;
            }
        }

        Some(best_placement)
    }
}
