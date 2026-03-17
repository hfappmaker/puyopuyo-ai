#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

use puyo_core::board::{COLS, ROWS};
use puyo_nn::encoding::{NUM_CHANNELS, PIECE_TENSOR_SIZE, TENSOR_SIZE};
use puyo_nn::model::PuyoPolicyNetConfig;
use puyo_trainer::data::Dataset;

#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<CudaJit<f32>>;
#[cfg(not(feature = "gpu"))]
type TrainBackend = Autodiff<NdArray>;

const DATA_PATH: &str = "data/training_data.bin";
const MODEL_PATH: &str = "artifacts/puyo_model";
#[cfg(feature = "gpu")]
const BATCH_SIZE: usize = 512;
#[cfg(not(feature = "gpu"))]
const BATCH_SIZE: usize = 64;
#[cfg(feature = "gpu")]
const VAL_BATCH_SIZE: usize = 512;
#[cfg(not(feature = "gpu"))]
const VAL_BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 50;
const LR_MAX: f64 = 5e-4;
const LR_MIN: f64 = 1e-5;
const EARLY_STOPPING_PATIENCE: usize = 5;
const TRAIN_SPLIT_RATIO: f64 = 0.9;
const NUM_ACTIONS: usize = 24;

fn cosine_lr(epoch: usize, total_epochs: usize) -> f64 {
    LR_MIN
        + 0.5
            * (LR_MAX - LR_MIN)
            * (1.0 + (std::f64::consts::PI * epoch as f64 / total_epochs as f64).cos())
}

/// Compute cross-entropy loss: -sum(log_softmax(logits)[target]) / batch_size
fn cross_entropy_loss<B: Backend>(logits: Tensor<B, 2>, targets: &[u8], device: &B::Device) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];

    // log_softmax along action dimension
    let max_logits = logits.clone().max_dim(1);
    let shifted = logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp; // [batch, 24]

    // Gather target log probabilities using one-hot encoding
    let mut target_one_hot = vec![0.0f32; batch_size * NUM_ACTIONS];
    for (i, &t) in targets.iter().enumerate() {
        target_one_hot[i * NUM_ACTIONS + t as usize] = 1.0;
    }
    let target_tensor = Tensor::<B, 1>::from_floats(target_one_hot.as_slice(), device)
        .reshape([batch_size, NUM_ACTIONS]);

    // -sum(one_hot * log_softmax) / batch_size
    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

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

    // Split into train/val
    let split = (num_samples as f64 * TRAIN_SPLIT_RATIO) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..];
    println!("Train: {}, Val: {}", train_samples.len(), val_samples.len());

    // Initialize model
    let config = PuyoPolicyNetConfig::new();
    let mut model = config.init::<TrainBackend>(&device);
    let mut optim = AdamConfig::new().init();

    // Training loop
    let mut best_val_loss = f32::MAX;
    let mut patience_counter = 0usize;

    for epoch in 0..NUM_EPOCHS {
        let lr = cosine_lr(epoch, NUM_EPOCHS);
        println!("Learning rate: {:.6}", lr);

        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        let mut rng_state = epoch as u64 + 42;
        const LCG_MULTIPLIER: u64 = 6364136223846793005;
        const LCG_INCREMENT: u64 = 1;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        for batch_start in (0..train_samples.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_samples.len());
            let batch_size = batch_end - batch_start;
            if batch_size == 0 {
                break;
            }

            let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut piece_data = Vec::with_capacity(batch_size * PIECE_TENSOR_SIZE);
            let mut target_actions = Vec::with_capacity(batch_size);

            for &idx in &indices[batch_start..batch_end] {
                let sample = &train_samples[idx];
                board_data.extend_from_slice(&sample.board_data);
                piece_data.extend_from_slice(&sample.piece_data);
                target_actions.push(sample.action_index);
            }

            let board_inputs =
                Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device)
                    .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
            let piece_inputs =
                Tensor::<TrainBackend, 1>::from_floats(piece_data.as_slice(), &device)
                    .reshape([batch_size, PIECE_TENSOR_SIZE]);

            // Forward pass
            let logits = model.forward(board_inputs, piece_inputs);

            // Cross-entropy loss
            let loss = cross_entropy_loss(logits, &target_actions, &device);

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;
            num_batches += 1;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = (train_samples.len() + BATCH_SIZE - 1) / BATCH_SIZE;
                eprint!(
                    "\r  batch {}/{} loss={:.6}",
                    num_batches,
                    total_batches,
                    epoch_loss / num_batches as f32
                );
            }
        }
        eprintln!();

        // Validation
        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let val_loss = compute_val_loss(&val_model, val_samples, &val_device);

        let avg_train_loss = epoch_loss / num_batches as f32;
        println!(
            "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.6}",
            epoch + 1,
            NUM_EPOCHS,
            avg_train_loss,
            val_loss,
            lr
        );

        // Early Stopping + Best Model Save
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
            let best_model = model.valid();
            best_model
                .save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
                .expect("Failed to save best model");
            println!("  -> Best model saved (val_loss={:.6})", val_loss);
        } else {
            patience_counter += 1;
            println!(
                "  -> No improvement ({}/{})",
                patience_counter, EARLY_STOPPING_PATIENCE
            );
            if patience_counter >= EARLY_STOPPING_PATIENCE {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }
    }

    if patience_counter < EARLY_STOPPING_PATIENCE {
        let model_valid = model.valid();
        model_valid
            .save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
            .expect("Failed to save model");
    }
    println!(
        "Training complete. Best model saved to {} (val_loss={:.6})",
        MODEL_PATH, best_val_loss
    );
}

type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

fn compute_val_loss(
    model: &puyo_nn::model::PuyoPolicyNet<InnerBackend>,
    val_samples: &[puyo_trainer::data::Sample],
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

        let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut piece_data = Vec::with_capacity(batch_size * PIECE_TENSOR_SIZE);
        let mut target_actions = Vec::with_capacity(batch_size);

        for sample in &val_samples[batch_start..batch_end] {
            board_data.extend_from_slice(&sample.board_data);
            piece_data.extend_from_slice(&sample.piece_data);
            target_actions.push(sample.action_index);
        }

        let board_inputs =
            Tensor::<InnerBackend, 1>::from_floats(board_data.as_slice(), device)
                .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
        let piece_inputs =
            Tensor::<InnerBackend, 1>::from_floats(piece_data.as_slice(), device)
                .reshape([batch_size, PIECE_TENSOR_SIZE]);

        let logits = model.forward(board_inputs, piece_inputs);

        // Cross-entropy loss (no grad)
        let loss = cross_entropy_loss(logits, &target_actions, device);
        total_loss += loss.into_data().to_vec::<f32>().unwrap()[0];
        num_batches += 1;
    }

    total_loss / num_batches.max(1) as f32
}
