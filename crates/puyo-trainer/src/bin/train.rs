use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_nn::encoding::{NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::PuyoValueNetConfig;
use puyo_trainer::data::Dataset;
use puyo_core::board::{COLS, ROWS};

type TrainBackend = Autodiff<NdArray>;

const DATA_PATH: &str = "data/training_data.bin";
const MODEL_PATH: &str = "artifacts/puyo_model";
const BATCH_SIZE: usize = 256;
const NUM_EPOCHS: usize = 20;
const LEARNING_RATE: f64 = 1e-3;
const MAX_TRAIN_SAMPLES: usize = 100_000;

fn main() {
    std::fs::create_dir_all("artifacts").expect("Failed to create artifacts directory");

    let device = Default::default();

    // Load dataset
    println!("Loading dataset from {}...", DATA_PATH);
    let dataset = Dataset::load(DATA_PATH).expect("Failed to load dataset");
    let num_samples = dataset.samples.len();
    println!("Loaded {} samples", num_samples);

    // Subsample for CPU training speed
    let used_samples = num_samples.min(MAX_TRAIN_SAMPLES);
    println!("Using {} samples (of {})", used_samples, num_samples);

    // Split into train/val (90/10)
    let split = (used_samples as f64 * 0.9) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..used_samples];
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

        // Shuffle indices (simple deterministic shuffle)
        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        // Fisher-Yates shuffle using epoch as seed
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
                // Normalize target
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

        // Validation
        let val_loss = compute_val_loss(&model, val_samples, mean, std_dev, &device);

        println!(
            "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
            epoch + 1,
            NUM_EPOCHS,
            epoch_loss / num_batches as f32,
            val_loss
        );
    }

    // Save model
    let model_valid = model.valid(); // Remove autodiff
    model_valid
        .save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model");
    println!("Model saved to {}", MODEL_PATH);
}

fn compute_val_loss(
    model: &puyo_nn::model::PuyoValueNet<TrainBackend>,
    val_samples: &[puyo_trainer::data::Sample],
    mean: f32,
    std_dev: f32,
    device: &<TrainBackend as Backend>::Device,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(val_samples.len());
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

        let inputs = Tensor::<TrainBackend, 1>::from_floats(
            input_data.as_slice(),
            device,
        )
        .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);

        let targets = Tensor::<TrainBackend, 1>::from_floats(
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
