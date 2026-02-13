use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_nn::encoding::{board_to_tensor_data, NUM_CHANNELS};
use puyo_nn::model::PuyoValueNet;

use crate::eval::Evaluator;

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
}

impl Evaluator for NnEvaluator {
    fn evaluate(&self, board: &Board) -> f64 {
        if board.is_game_over() {
            return -100000.0;
        }

        let data = board_to_tensor_data(board);
        let tensor = Tensor::<InferBackend, 1>::from_floats(data.as_slice(), &self.device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let output = self.model.forward(tensor);
        let normalized = output.into_data().to_vec::<f32>().unwrap()[0];
        // Denormalize
        (normalized * self.std_dev + self.mean) as f64
    }
}
