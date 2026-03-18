use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

use crate::encoding::{CONTEXT_TENSOR_SIZE, NUM_CHANNELS};

const RESIDUAL_CHANNELS: usize = 64;
const NUM_RESIDUAL_BLOCKS: usize = 6;
const HEAD_CHANNELS: usize = 128;
const POOL_H: usize = 4;
const POOL_W: usize = 3;
const HIDDEN_SIZE: usize = 256;
const NUM_ACTIONS: usize = 24; // 6 cols × 4 orientations
const BACKBONE_OUTPUT: usize = HEAD_CHANNELS * POOL_H * POOL_W; // 1536

/// FiLM parameters generated from context (pieces).
const FILM_HIDDEN: usize = 64;
/// FiLM output size: gamma (RESIDUAL_CHANNELS) + beta (RESIDUAL_CHANNELS).
const FILM_OUTPUT: usize = RESIDUAL_CHANNELS * 2;

/// Residual block with FiLM conditioning.
///
/// Conv → ReLU → Conv → FiLM(gamma, beta) → add skip → ReLU.
/// FiLM applies channel-wise affine transform: y = gamma * x + beta.
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
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
            conv2: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ResidualBlock<B> {
    /// Forward pass with FiLM conditioning.
    /// gamma, beta: [batch, channels] — broadcast over spatial dims.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        gamma: Tensor<B, 2>,
        beta: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let residual = x.clone();
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);

        // FiLM: reshape [batch, channels] → [batch, channels, 1, 1] for broadcast
        let gamma_dims = gamma.dims();
        let gamma = gamma.reshape([gamma_dims[0], gamma_dims[1], 1, 1]);
        let beta_dims = beta.dims();
        let beta = beta.reshape([beta_dims[0], beta_dims[1], 1, 1]);
        let x = x * gamma + beta;

        self.activation.forward(x + residual)
    }
}

/// Dual-head CNN for Puyo Puyo with FiLM conditioning (AlphaZero-style).
///
/// Architecture:
///   FiLM generator: context(25) → Linear(25→64) → ReLU → Linear(64→128) → split(gamma, beta)
///   Backbone: stem (6ch → 64ch) → FiLMResidualBlock ×6 (64ch) → head_conv (64ch → 128ch)
///     → AdaptiveAvgPool → flatten [1536]
///   Policy Head: Linear(1536→256) → ReLU → Linear(256→24)
///   Value Head:  Linear(1536→256) → ReLU → Linear(256→1)
///
/// Board input: [batch, 6, 14, 6]
/// Context input: [batch, 25] (pieces one-hot encoding + remaining turns ratio)
/// Output: (policy_logits [batch, 24], value [batch, 1])
#[derive(Module, Debug)]
pub struct PuyoNet<B: Backend> {
    // FiLM generator
    film_fc1: Linear<B>,
    film_fc2: Linear<B>,
    // CNN backbone
    stem: Conv2d<B>,
    res_blocks: Vec<ResidualBlock<B>>,
    head_conv: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    // Policy head
    policy_fc1: Linear<B>,
    policy_fc2: Linear<B>,
    // Value head
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,
    activation: Relu,
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
            head_conv: Conv2dConfig::new([RESIDUAL_CHANNELS, HEAD_CHANNELS], [1, 1]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([POOL_H, POOL_W]).init(),
            policy_fc1: LinearConfig::new(BACKBONE_OUTPUT, HIDDEN_SIZE).init(device),
            policy_fc2: LinearConfig::new(HIDDEN_SIZE, NUM_ACTIONS).init(device),
            value_fc1: LinearConfig::new(BACKBONE_OUTPUT, HIDDEN_SIZE).init(device),
            value_fc2: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PuyoNet<B> {
    /// Forward pass.
    /// board: [batch, 6, 14, 6], context: [batch, 25]
    /// Returns: (policy_logits [batch, 24], value [batch, 1])
    pub fn forward(
        &self,
        board: Tensor<B, 4>,
        context: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = board.dims()[0];

        // FiLM generator: context → gamma, beta
        let film = self.film_fc1.forward(context);
        let film = self.activation.forward(film);
        let film = self.film_fc2.forward(film); // [batch, 128]

        // Split into gamma [batch, 64] and beta [batch, 64]
        let gamma = film.clone().slice([0..batch_size, 0..RESIDUAL_CHANNELS]);
        let beta = film.slice([0..batch_size, RESIDUAL_CHANNELS..FILM_OUTPUT]);

        // Stem
        let mut x = self.stem.forward(board);
        x = self.activation.forward(x);

        // Residual blocks with FiLM
        for block in &self.res_blocks {
            x = block.forward(x, gamma.clone(), beta.clone());
        }

        // Backbone head
        let x = self.head_conv.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        let backbone = x.reshape([batch_size, BACKBONE_OUTPUT]);

        // Policy head
        let p = self.policy_fc1.forward(backbone.clone());
        let p = self.activation.forward(p);
        let policy_logits = self.policy_fc2.forward(p);

        // Value head (outputs discounted cumulative reward, no activation)
        let v = self.value_fc1.forward(backbone);
        let v = self.activation.forward(v);
        let value = self.value_fc2.forward(v);

        (policy_logits, value)
    }
}
