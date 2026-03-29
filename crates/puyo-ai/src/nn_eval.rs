use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use game_core::Game;
use puyo_core::board::{COLS, ROWS};
use puyo_core::piece::Placement;
use puyo_core::puyo_game::{
    board_to_tensor_data, context_to_tensor_data, PuyoState, CONTEXT_TENSOR_SIZE, NUM_CHANNELS,
};
use puyo_nn::model::PuyoNet;
use puyo_nn::value_transform::value_inverse_transform;

use crate::eval::Evaluator;
use crate::mcts::{mcts_search, InferenceProvider};
use crate::placement::{compute_valid_mask, index_to_placement, NUM_ACTIONS};
use crate::puyo_game::PuyoGame;

type InferBackend = NdArray;

/// MCTS configuration (Gumbel AlphaZero).
pub struct MctsConfig {
    pub num_simulations: usize,
    pub c_puct: f32,
    /// Number of initial actions to sample via Gumbel-Top-k.
    pub m: usize,
    /// Q-value scaling for advantage computation.
    pub c_visit: f32,
    /// Discount factor for future rewards.
    pub gamma: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 64,
            c_puct: 1.5,
            m: 16,
            c_visit: 5.0,
            gamma: 0.95,
        }
    }
}

/// Direct (single-sample) inference using a burn backend.
/// Used for CPU inference (NdArray) in WASM and single-threaded scenarios.
pub struct DirectInference<B: Backend> {
    model: PuyoNet<B>,
    device: B::Device,
}

impl<B: Backend> DirectInference<B> {
    pub fn new(model: PuyoNet<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    pub fn model(&self) -> &PuyoNet<B> {
        &self.model
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

impl<B: Backend> Clone for DirectInference<B> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            device: self.device.clone(),
        }
    }
}

impl<B: Backend> InferenceProvider for DirectInference<B> {
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32) {
        let board_tensor = Tensor::<B, 1>::from_floats(board_data, &self.device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let context_tensor = Tensor::<B, 1>::from_floats(context_data, &self.device)
            .reshape([1, CONTEXT_TENSOR_SIZE]);

        let (logits, value) = self.model.forward(board_tensor, context_tensor);

        let logits_vec = logits.into_data().to_vec::<f32>().expect("Failed to extract logits tensor");
        let value_scalar = value.into_data().to_vec::<f32>().expect("Failed to extract value tensor");
        let v_raw = if value_scalar.is_empty() { 0.0 } else { value_scalar[0] };
        let v = value_inverse_transform(v_raw);

        (logits_vec, v)
    }
}

/// Neural network evaluator using the dual-head PuyoNet.
/// Supports two modes: Policy-only (fast, for WASM) and MCTS (for training).
pub struct NnEvaluator {
    provider: DirectInference<InferBackend>,
    mcts_config: Option<MctsConfig>,
}

impl NnEvaluator {
    pub fn new(
        model: PuyoNet<InferBackend>,
        device: <InferBackend as Backend>::Device,
    ) -> Self {
        Self {
            provider: DirectInference::new(model, device),
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
        self.provider.model()
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
            let seed = PuyoGame::state_hash(state);

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
