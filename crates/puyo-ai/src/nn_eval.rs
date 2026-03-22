use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::{Piece, Placement};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data, CONTEXT_TENSOR_SIZE, NUM_CHANNELS};
use puyo_nn::model::PuyoNet;

use crate::eval::Evaluator;
use crate::mcts::{board_hash, mcts_search};
use crate::placement::{compute_valid_mask, index_to_placement, NUM_ACTIONS};

type InferBackend = NdArray;

/// MCTS configuration (Gumbel AlphaZero).
pub struct MctsConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
    /// Number of initial actions to sample via Gumbel-Top-k.
    pub m: usize,
    /// Q-value scaling for advantage computation.
    pub c_visit: f32,
    /// Scale parameter for advantage computation.
    pub c_scale: f32,
    /// Discount factor for future rewards.
    pub gamma: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            c_puct: 1.5,
            m: 16,
            c_visit: 1.0,
            c_scale: 1.0,
            gamma: 0.95,
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

        // MCTS mode: use Gumbel tree search
        if let Some(ref mcts_config) = self.mcts_config {
            let seed = board_hash(board);

            let (policy, q_values) = mcts_search(
                board,
                current,
                next,
                next_next,
                &self.model,
                &self.device,
                mcts_config,
                seed,
            );

            // Select the action with highest improved policy probability
            let mut best_index = 0;
            let mut best_prob = f64::NEG_INFINITY;
            for i in 0..NUM_ACTIONS {
                if mask[i] && (policy[i] as f64) > best_prob {
                    best_prob = policy[i] as f64;
                    best_index = i;
                }
            }
            // Return Q value (average cumulative reward) as the score
            return Some((index_to_placement(best_index), q_values[best_index] as f64));
        }

        // Policy-only mode (fast, for WASM)
        let board_data = board_to_tensor_data(board);
        let context_data = context_to_tensor_data(current, next, next_next);

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

    fn set_num_simulations(&mut self, num_simulations: usize) {
        if let Some(ref mut config) = self.mcts_config {
            config.num_simulations = num_simulations;
        }
    }
}
