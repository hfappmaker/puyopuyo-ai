use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{COLS, ROWS};
use puyo_core::piece::Placement;
use puyo_core::state::{
    board_to_tensor_data, context_to_tensor_data, PuyoState, CONTEXT_TENSOR_SIZE, NUM_CHANNELS,
};
use puyo_nn::model::PuyoNet;

use az_framework::eval::Evaluator;
use az_framework::mcts::{mcts_search, InferenceProvider};
use az_framework::model::GameModel;
use az_framework::nn_eval::DirectInference;
use az_framework::value_transform::value_inverse_transform;
pub use az_framework::nn_eval::MctsConfig;

const VALUE_SCALE: f32 = 15.0;

use puyo_core::placement::{compute_valid_mask, index_to_placement, NUM_ACTIONS};
use crate::puyo_game::PuyoGame;

type InferBackend = NdArray;

/// GameModel implementation wrapping PuyoNet for use with generic game-ai infrastructure.
pub struct PuyoGameModel<B: Backend> {
    pub net: PuyoNet<B>,
}

impl<B: Backend> PuyoGameModel<B> {
    pub fn new(net: PuyoNet<B>) -> Self {
        Self { net }
    }
}

impl<B: Backend> Clone for PuyoGameModel<B> {
    fn clone(&self) -> Self {
        Self {
            net: self.net.clone(),
        }
    }
}

impl<B: Backend> GameModel<B> for PuyoGameModel<B> {
    fn board_shape(&self) -> (usize, usize, usize) {
        (NUM_CHANNELS, ROWS, COLS)
    }

    fn context_size(&self) -> usize {
        CONTEXT_TENSOR_SIZE
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn forward(&self, board: Tensor<B, 4>, context: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.net.forward(board, context)
    }

    fn postprocess_value(&self, raw: f32) -> f32 {
        value_inverse_transform(raw, VALUE_SCALE)
    }
}

/// Neural network evaluator using the dual-head PuyoNet.
/// Supports two modes: Policy-only (fast, for WASM) and MCTS (for training).
pub struct NnEvaluator {
    provider: DirectInference<InferBackend, PuyoGameModel<InferBackend>>,
    mcts_config: Option<MctsConfig>,
}

impl NnEvaluator {
    pub fn new(
        model: PuyoNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
    ) -> Self {
        Self {
            provider: DirectInference::new(PuyoGameModel::new(model), device),
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
        &self.provider.model().net
    }

    pub fn device(&self) -> &<InferBackend as Backend>::Device {
        self.provider.device()
    }
}


impl Evaluator<PuyoGame> for NnEvaluator {
    fn find_best_move(
        &self,
        state: &PuyoState,
    ) -> Option<(Placement, f64)> {
        let mask = compute_valid_mask(&state.board, &state.current);
        if !mask.iter().any(|&v| v) {
            return None;
        }

        // MCTS mode: use Gumbel tree search
        if let Some(ref mcts_config) = self.mcts_config {
            let seed = puyo_core::rand::time_seed();

            let (policy, q_values) = mcts_search::<PuyoGame>(
                state,
                &self.provider,
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
        let board_data = board_to_tensor_data(&state.board);
        let context_data = context_to_tensor_data(&state.current, &state.next, &state.next_next);

        let (logits_vec, _value) = self.provider.infer(&board_data, &context_data);

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
