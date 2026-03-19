use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Piece, Placement};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data, CONTEXT_TENSOR_SIZE, NUM_CHANNELS};
use puyo_nn::model::PuyoNet;

use crate::eval::Evaluator;
use crate::mcts::mcts_search;
use crate::placement::{compute_valid_mask, index_to_placement, NUM_ACTIONS};

type InferBackend = NdArray;

/// MCTS configuration.
pub struct MctsConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 200,
            c_puct: 1.5,
            temperature: 0.1,
        }
    }
}

/// Neural network evaluator using the dual-head PuyoNet.
/// Supports two modes: Policy-only (fast, for WASM) and MCTS (for training).
pub struct NnEvaluator {
    model: PuyoNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
    mcts_config: Option<MctsConfig>,
}

impl NnEvaluator {
    pub fn new(
        model: PuyoNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
    ) -> Self {
        Self {
            model,
            device,
            mcts_config: None,
        }
    }

    /// Enable MCTS mode for training.
    pub fn with_mcts(mut self, config: MctsConfig) -> Self {
        self.mcts_config = Some(config);
        self
    }

    /// Get access to the model (for MCTS in self-play).
    pub fn model(&self) -> &PuyoNet<InferBackend> {
        &self.model
    }

    pub fn device(&self) -> &<InferBackend as Backend>::Device {
        &self.device
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

        // MCTS mode: use tree search
        if let Some(ref mcts_config) = self.mcts_config {
            // MCTS mode uses max_turns=0 to indicate no turn limit (remaining_ratio=1.0)
            let policy = mcts_search(
                board,
                current,
                next,
                next_next,
                &self.model,
                &self.device,
                mcts_config.num_simulations,
                mcts_config.c_puct,
                mcts_config.temperature,
                0,
                0,
            );

            let mut best_index = 0;
            let mut best_prob = f64::NEG_INFINITY;
            for i in 0..NUM_ACTIONS {
                if mask[i] && (policy[i] as f64) > best_prob {
                    best_prob = policy[i] as f64;
                    best_index = i;
                }
            }
            return Some((index_to_placement(best_index), best_prob));
        }

        // Policy-only mode (fast, for WASM)
        let board_data = board_to_tensor_data(board);
        // Policy-only mode has no turn limit; use 1.0 (full remaining turns)
        let context_data = context_to_tensor_data(current, next, next_next, 1.0);

        let board_tensor =
            Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), &self.device)
                .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let context_tensor =
            Tensor::<InferBackend, 1>::from_floats(context_data.as_slice(), &self.device)
                .reshape([1, CONTEXT_TENSOR_SIZE]);

        let (logits, _value) = self.model.forward(board_tensor, context_tensor);
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
