use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

use puyo_core::config::{COLS, CONTEXT_TENSOR_SIZE, NUM_ACTIONS, NUM_CHANNELS, ROWS};

const RESIDUAL_CHANNELS: usize = 64;
const NUM_RESIDUAL_BLOCKS: usize = 6;
const HEAD_DROPOUT: f64 = 0.2;

// Shared FC trunk (backbone flatten + context → shared hidden → policy/value)
const BACKBONE_FLAT: usize = RESIDUAL_CHANNELS * ROWS * COLS;
const SHARED_HIDDEN: usize = 256;

/// FiLM parameters generated from context (pieces).
const FILM_HIDDEN: usize = 128;
/// FiLM output size: per-block (gamma + beta) for each residual block.
const FILM_OUTPUT: usize = RESIDUAL_CHANNELS * 2 * NUM_RESIDUAL_BLOCKS; // 768

/// Residual block with GroupNorm and FiLM conditioning.
///
/// Conv → GroupNorm → ReLU → Conv → GroupNorm → Residual FiLM(gamma, beta) → add skip → ReLU.
/// Residual FiLM: y = x * (1 + gamma) + beta (identity when gamma=0, beta=0).
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: GroupNorm<B>,
    conv2: Conv2d<B>,
    norm2: GroupNorm<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    channels: usize,
}

impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        ResidualBlock {
            conv1: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            norm1: GroupNormConfig::new(1, self.channels).init(device),
            conv2: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            norm2: GroupNormConfig::new(1, self.channels).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    /// Forward pass with residual FiLM conditioning.
    /// gamma, beta: [batch, channels] — broadcast over spatial dims.
    /// Uses residual FiLM: y = x * (1 + gamma) + beta so that gamma=0 → identity.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        gamma: Tensor<B, 2>,
        beta: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let residual = x.clone();
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);

        // Residual FiLM: reshape [batch, channels] → [batch, channels, 1, 1] for broadcast
        let gamma_dims = gamma.dims();
        let gamma = gamma.reshape([gamma_dims[0], gamma_dims[1], 1, 1]);
        let beta_dims = beta.dims();
        let beta = beta.reshape([beta_dims[0], beta_dims[1], 1, 1]);
        // y = x * (1 + gamma) + beta: when gamma=0, beta=0 this is identity
        let x = x * (gamma + 1.0) + beta;

        self.activation.forward(x + residual)
    }
}

/// Dual-head CNN for Puyo Puyo with per-block FiLM conditioning (AlphaZero-style).
///
/// Architecture:
///   FiLM generator: context(CONTEXT_TENSOR_SIZE) → Linear → ReLU → Linear → FILM_OUTPUT
///     → split into NUM_RESIDUAL_BLOCKS × (gamma[ch], beta[ch]) for each residual block
///   Backbone: stem (NUM_CHANNELS ch → 64ch) → (GroupNorm + FiLM) ResidualBlock ×6 (64ch)
///   Shared FC trunk:
///     flatten(backbone) + context → Linear(1554→256) → ReLU → Dropout
///   Policy Head: Linear(256→NUM_ACTIONS)
///   Value Head: Linear(256→1)
///
/// Board input: [batch, NUM_CHANNELS, ROWS, COLS]
/// Context input: [batch, CONTEXT_TENSOR_SIZE] (pieces one-hot encoding)
/// Output: (policy_logits [batch, NUM_ACTIONS], value [batch, 1])
#[derive(Module, Debug)]
pub struct PuyoNet<B: Backend> {
    // FiLM generator
    film_fc1: Linear<B>,
    film_fc2: Linear<B>,
    // CNN backbone
    stem: Conv2d<B>,
    res_blocks: Vec<ResidualBlock<B>>,
    // Shared FC trunk + heads
    shared_fc: Linear<B>,
    policy_fc: Linear<B>,
    value_fc: Linear<B>,
    activation: Relu,
    head_dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct PuyoNetConfig {}

impl PuyoNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PuyoNet<B> {
        let res_block_config = ResidualBlockConfig::new(RESIDUAL_CHANNELS);
        let res_blocks = (0..NUM_RESIDUAL_BLOCKS)
            .map(|_| res_block_config.init(device))
            .collect();

        PuyoNet {
            film_fc1: LinearConfig::new(CONTEXT_TENSOR_SIZE, FILM_HIDDEN).init(device),
            film_fc2: LinearConfig::new(FILM_HIDDEN, FILM_OUTPUT).init(device),
            stem: Conv2dConfig::new([NUM_CHANNELS, RESIDUAL_CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            res_blocks,
            // Shared FC trunk + heads
            shared_fc: LinearConfig::new(BACKBONE_FLAT + CONTEXT_TENSOR_SIZE, SHARED_HIDDEN).init(device),
            policy_fc: LinearConfig::new(SHARED_HIDDEN, NUM_ACTIONS).init(device),
            value_fc: LinearConfig::new(SHARED_HIDDEN, 1).init(device),
            activation: Relu::new(),
            head_dropout: DropoutConfig::new(HEAD_DROPOUT).init(),
        }
    }
}

impl<B: Backend> PuyoNet<B> {
    /// Forward pass.
    /// board: [batch, NUM_CHANNELS, ROWS, COLS], context: [batch, CONTEXT_TENSOR_SIZE]
    /// Returns: (policy_logits [batch, NUM_ACTIONS], value [batch, 1])
    pub fn forward(
        &self,
        board: Tensor<B, 4>,
        context: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = board.dims()[0];

        // FiLM generator: context → per-block (gamma, beta)
        let film = self.film_fc1.forward(context.clone());
        let film = self.activation.forward(film);
        let film = self.film_fc2.forward(film); // [batch, 768]

        // Stem
        let mut x = self.stem.forward(board);
        x = self.activation.forward(x);

        // Residual blocks with per-block FiLM
        let ch = RESIDUAL_CHANNELS;
        for (i, block) in self.res_blocks.iter().enumerate() {
            let g_start = i * ch;
            let b_start = NUM_RESIDUAL_BLOCKS * ch + i * ch;
            let gamma = film.clone().slice([0..batch_size, g_start..g_start + ch]);
            let beta = film.clone().slice([0..batch_size, b_start..b_start + ch]);
            x = block.forward(x, gamma, beta);
        }

        // Shared FC trunk: flatten backbone + context
        let flat = x.reshape([batch_size, BACKBONE_FLAT]);
        let combined = Tensor::cat(vec![flat, context], 1);
        let shared = self.shared_fc.forward(combined);
        let shared = self.activation.forward(shared);

        // Policy head (no dropout to preserve MCTS policy quality)
        let policy_logits = self.policy_fc.forward(shared.clone());

        // Value head (dropout for regularization)
        let v = self.head_dropout.forward(shared);
        let value = self.value_fc.forward(v);

        (policy_logits, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use puyo_core::config::ROWS;

    type B = NdArray;

    #[test]
    fn test_forward_output_shapes() {
        let device = <B as Backend>::Device::default();
        let model = PuyoNetConfig::new().init::<B>(&device);

        let board = Tensor::<B, 4>::zeros([2, NUM_CHANNELS, ROWS, COLS], &device);
        let context = Tensor::<B, 2>::zeros([2, CONTEXT_TENSOR_SIZE], &device);
        let (policy, value) = model.forward(board, context);

        assert_eq!(policy.dims(), [2, NUM_ACTIONS]);
        assert_eq!(value.dims(), [2, 1]);
    }
}
