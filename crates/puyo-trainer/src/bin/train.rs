#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_nn::encoding::{NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::PuyoValueNetConfig;
use puyo_trainer::data::Dataset;
use puyo_core::board::{COLS, ROWS};

#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<CudaJit<f32>>;
#[cfg(not(feature = "gpu"))]
type TrainBackend = Autodiff<NdArray>;

const DATA_PATH: &str = "data/training_data.bin";
const MODEL_PATH: &str = "artifacts/puyo_model";
#[cfg(feature = "gpu")]
const BATCH_SIZE: usize = 8;
#[cfg(not(feature = "gpu"))]
const BATCH_SIZE: usize = 512;
#[cfg(feature = "gpu")]
const VAL_BATCH_SIZE: usize = 512;
#[cfg(not(feature = "gpu"))]
const VAL_BATCH_SIZE: usize = 8;
const NUM_EPOCHS: usize = 20;
const LEARNING_RATE: f64 = 5e-4;

fn main() {
    std::fs::create_dir_all("artifacts").expect("Failed to create artifacts directory");

    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let device = Default::default();

    // Load dataset
    println!("Loading dataset from {}...", DATA_PATH);
    let dataset = Dataset::load(DATA_PATH).expect("Failed to load dataset");
    let num_samples = dataset.samples.len();
    println!("Loaded {} samples", num_samples);

    // Split into train/val (90/10) — use full dataset on GPU
    let split = (num_samples as f64 * 0.9) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..];
    println!("Train: {}, Val: {}", train_samples.len(), val_samples.len());

    // Compute normalization stats on training set
    let targets: Vec<f32> = train_samples.iter().map(|s| s.target).collect();
    let mean = targets.iter().sum::<f32>() / targets.len() as f32;
    let variance = targets.iter().map(|t| (t - mean).powi(2)).sum::<f32>() / targets.len() as f32;
    let std_dev = variance.sqrt().max(1e-6);
    println!("Target stats: mean={:.4}, std={:.4}", mean, std_dev);

    // Save normalization params
    let norm_params = format!("{}\n{}", mean, std_dev);
    std::fs::write("artifacts/norm_params.txt", norm_params)
        .expect("Failed to save normalization params");

    // Initialize model
    let config = PuyoValueNetConfig::new();
    let mut model = config.init::<TrainBackend>(&device);
    let mut optim = AdamConfig::new().init();

    // Training loop
    for epoch in 0..NUM_EPOCHS {
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        let mut rng_state = epoch as u64 + 42;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        for batch_start in (0..train_samples.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_samples.len());
            let batch_size = batch_end - batch_start;
            if batch_size == 0 {
                break;
            }

            // Build input tensor [batch, 5, 13, 6]
            let mut input_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut target_data = Vec::with_capacity(batch_size);

            for &idx in &indices[batch_start..batch_end] {
                let sample = &train_samples[idx];
                input_data.extend_from_slice(&sample.board_data);
                target_data.push((sample.target - mean) / std_dev);
            }

            let inputs = Tensor::<TrainBackend, 1>::from_floats(
                input_data.as_slice(),
                &device,
            )
            .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);

            let targets = Tensor::<TrainBackend, 1>::from_floats(
                target_data.as_slice(),
                &device,
            )
            .reshape([batch_size, 1]);

            // Forward pass
            let predictions = model.forward(inputs);

            // MSE loss
            let diff = predictions - targets;
            let loss = diff.clone().mul(diff).mean();

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;
            num_batches += 1;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = (train_samples.len() + BATCH_SIZE - 1) / BATCH_SIZE;
                eprint!(
                    "\r  batch {}/{} loss={:.6}",
                    num_batches, total_batches,
                    epoch_loss / num_batches as f32
                );
            }
        }
        eprintln!();

        // Validation (model.valid()で勾配グラフなしの推論モード)
        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let val_loss = compute_val_loss(&val_model, val_samples, mean, std_dev, &val_device);

        println!(
            "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
            epoch + 1,
            NUM_EPOCHS,
            epoch_loss / num_batches as f32,
            val_loss
        );
    }

    // Save model
    let model_valid = model.valid();
    model_valid
        .save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model");
    println!("Model saved to {}", MODEL_PATH);
}

type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

fn compute_val_loss(
    model: &puyo_nn::model::PuyoValueNet<InnerBackend>,
    val_samples: &[puyo_trainer::data::Sample],
    mean: f32,
    std_dev: f32,
    device: &<InnerBackend as Backend>::Device,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(VAL_BATCH_SIZE) {
        let batch_end = (batch_start + VAL_BATCH_SIZE).min(val_samples.len());
        let batch_size = batch_end - batch_start;
        if batch_size == 0 {
            break;
        }

        let mut input_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut target_data = Vec::with_capacity(batch_size);

        for sample in &val_samples[batch_start..batch_end] {
            input_data.extend_from_slice(&sample.board_data);
            target_data.push((sample.target - mean) / std_dev);
        }

        let inputs = Tensor::<InnerBackend, 1>::from_floats(
            input_data.as_slice(),
            device,
        )
        .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);

        let targets = Tensor::<InnerBackend, 1>::from_floats(
            target_data.as_slice(),
            device,
        )
        .reshape([batch_size, 1]);

        let predictions = model.forward(inputs);
        let diff = predictions - targets;
        let loss = diff.clone().mul(diff).mean();

        total_loss += loss.into_data().to_vec::<f32>().unwrap()[0];
        num_batches += 1;
    }

    total_loss / num_batches.max(1) as f32
}
