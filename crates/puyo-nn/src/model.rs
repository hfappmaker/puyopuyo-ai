use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::*;

use crate::encoding::NUM_CHANNELS;

const CONV1_CHANNELS: usize = 32;
const CONV2_CHANNELS: usize = 32;
const CONV3_CHANNELS: usize = 64;
const POOL_H: usize = 4;
const POOL_W: usize = 3;
const HIDDEN_SIZE: usize = 128;

/// CNN value network for Puyo Puyo board evaluation.
///
/// Input: [batch, 5, 13, 6] (one-hot encoded board)
/// Output: [batch, 1] (scalar evaluation value)
#[derive(Module, Debug)]
pub struct PuyoValueNet<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct PuyoValueNetConfig {}

impl PuyoValueNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PuyoValueNet<B> {
        PuyoValueNet {
            conv1: Conv2dConfig::new([NUM_CHANNELS, CONV1_CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([CONV1_CHANNELS, CONV2_CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([CONV2_CHANNELS, CONV3_CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([POOL_H, POOL_W]).init(),
            linear1: LinearConfig::new(CONV3_CHANNELS * POOL_H * POOL_W, HIDDEN_SIZE)
                .init(device),
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

        let x = self.conv1.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, CONV3_CHANNELS * POOL_H * POOL_W]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
