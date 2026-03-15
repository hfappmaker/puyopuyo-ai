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
    /// BFS順で全深度の盤面を評価し、最高スコアの1手目を返す。
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        next_next: &Piece,
    ) -> Option<(Placement, f64)> {
        let placements = enumerate_placements(board, current);
        if placements.is_empty() {
            return None;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for p1 in &placements {
            let (board1, _) = simulate_placement(board, current, p1);
            if board1.is_game_over() {
                continue;
            }

            let s = self.nn_evaluate(&board1);
            if s > best_score {
                best_score = s;
                best_placement = *p1;
            }

            for p2 in &enumerate_placements(&board1, next) {
                let (board2, _) = simulate_placement(&board1, next, p2);
                if board2.is_game_over() {
                    continue;
                }

                let s = self.nn_evaluate(&board2);
                if s > best_score {
                    best_score = s;
                    best_placement = *p1;
                }

                // for p3 in &enumerate_placements(&board2, next_next) {
                //     let (board3, _) = simulate_placement(&board2, next_next, p3);
                //     if board3.is_game_over() {
                //         continue;
                //     }

                //     let s = self.nn_evaluate(&board3);
                //     if s > best_score {
                //         best_score = s;
                //         best_placement = *p1;
                //     }
                // }
            }
        }

        Some((best_placement, best_score))
    }
}
