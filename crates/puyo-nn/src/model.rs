use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

use crate::encoding::NUM_CHANNELS;

const RESIDUAL_CHANNELS: usize = 32;
const NUM_RESIDUAL_BLOCKS: usize = 2;
const HEAD_CHANNELS: usize = 64;
const POOL_H: usize = 4;
const POOL_W: usize = 3;
const HIDDEN_SIZE: usize = 128;

/// Residual block: Conv → ReLU → Conv, then add input (skip connection) → ReLU.
///
/// Both convolutions preserve spatial dimensions (padding=Same) and channel count.
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
    /// Forward pass: y = ReLU(Conv(ReLU(Conv(x))) + x)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        self.activation.forward(x + residual)
    }
}

/// CNN value network for Puyo Puyo board evaluation with residual connections.
///
/// Architecture:
///   stem (5ch → 32ch) → ResidualBlock ×2 (32ch) → head_conv (32ch → 64ch)
///   → AdaptiveAvgPool → Linear(768→128) → Linear(128→1)
///
/// Input: [batch, 5, 13, 6] (one-hot encoded board)
/// Output: [batch, 1] (scalar evaluation value)
#[derive(Module, Debug)]
pub struct PuyoValueNet<B: Backend> {
    stem: Conv2d<B>,
    res_blocks: Vec<ResidualBlock<B>>,
    head_conv: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct PuyoValueNetConfig {}

impl PuyoValueNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PuyoValueNet<B> {
        let res_block_config = ResidualBlockConfig::new(RESIDUAL_CHANNELS);
        let res_blocks = (0..NUM_RESIDUAL_BLOCKS)
            .map(|_| res_block_config.init(device))
            .collect();

        PuyoValueNet {
            stem: Conv2dConfig::new([NUM_CHANNELS, RESIDUAL_CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            res_blocks,
            head_conv: Conv2dConfig::new([RESIDUAL_CHANNELS, HEAD_CHANNELS], [1, 1]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([POOL_H, POOL_W]).init(),
            linear1: LinearConfig::new(HEAD_CHANNELS * POOL_H * POOL_W, HIDDEN_SIZE).init(device),
            linear2: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> PuyoValueNet<B> {
    /// Forward pass.
    /// Input shape: [batch, NUM_CHANNELS, ROWS, COLS] = [batch, 5, 13, 6]
    /// Output shape: [batch, 1]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = x.dims()[0];

        // Stem: project input channels to residual channels
        let mut x = self.stem.forward(x);
        x = self.activation.forward(x);

        // Residual blocks
        for block in &self.res_blocks {
            x = block.forward(x);
        }

        // Head: expand channels and pool
        let x = self.head_conv.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, HEAD_CHANNELS * POOL_H * POOL_W]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
