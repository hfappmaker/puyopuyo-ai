//! Self-play reinforcement learning for the policy network.
//!
//! NOTE: This is a placeholder for future RL training.
//! Currently, supervised learning (train binary) is the primary training method.
//! This binary compiles but the RL loop is not yet adapted to the policy network.

#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_nn::model::PuyoPolicyNetConfig;

#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<CudaJit<f32>>;
#[cfg(not(feature = "gpu"))]
type TrainBackend = Autodiff<NdArray>;

const MODEL_PATH: &str = "artifacts/puyo_model";

fn main() {
    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let device: <TrainBackend as Backend>::Device = Default::default();

    let config = PuyoPolicyNetConfig::new();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let _model = config
        .init::<TrainBackend>(&device)
        .load_file(MODEL_PATH, &recorder, &device)
        .expect("Failed to load model. Run 'train' first.");

    println!("Self-play RL for policy network is not yet implemented.");
    println!("Use 'cargo run --bin train -p puyo-trainer' for supervised learning.");
}
