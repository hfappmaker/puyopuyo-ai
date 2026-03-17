use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Piece, Placement};
use puyo_nn::encoding::{board_to_tensor_data, pieces_to_tensor_data, NUM_CHANNELS, PIECE_TENSOR_SIZE};
use puyo_nn::model::PuyoPolicyNet;

use crate::eval::Evaluator;
use crate::placement::{compute_valid_mask, index_to_placement};

type InferBackend = NdArray;

/// Neural network policy evaluator.
/// Outputs placement probabilities directly — no search loop needed.
pub struct NnEvaluator {
    model: PuyoPolicyNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
}

impl NnEvaluator {
    pub fn new(
        model: PuyoPolicyNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
    ) -> Self {
        Self { model, device }
    }
}

impl Evaluator for NnEvaluator {
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        next_next: &Piece,
    ) -> Option<(Placement, f64)> {
        let mask = compute_valid_mask(board, current);
        if !mask.iter().any(|&v| v) {
            return None;
        }

        // Encode inputs
        let board_data = board_to_tensor_data(board);
        let piece_data = pieces_to_tensor_data(current, next, next_next);

        let board_tensor =
            Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), &self.device)
                .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let piece_tensor =
            Tensor::<InferBackend, 1>::from_floats(piece_data.as_slice(), &self.device)
                .reshape([1, PIECE_TENSOR_SIZE]);

        // Forward pass → logits [1, 24]
        let logits = self.model.forward(board_tensor, piece_tensor);
        let logits_vec = match logits.into_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => return None,
        };

        // Masked argmax
        let mut best_index = 0;
        let mut best_logit = f64::NEG_INFINITY;
        for (i, &logit) in logits_vec.iter().enumerate() {
            if mask[i] && (logit as f64) > best_logit {
                best_logit = logit as f64;
                best_index = i;
            }
        }

        Some((index_to_placement(best_index), best_logit))
    }
}
