#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

use puyo_core::board::{COLS, ROWS};
use puyo_nn::encoding::{CONTEXT_TENSOR_SIZE, NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::PuyoNetConfig;
use puyo_nn::value_transform::value_transform;
use puyo_trainer::data::{AlphaZeroDataset, Dataset};

#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<CudaJit<f32>>;
#[cfg(not(feature = "gpu"))]
type TrainBackend = Autodiff<NdArray>;
type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

const MODEL_PATH: &str = "artifacts/puyo_model";
#[cfg(feature = "gpu")]
const BATCH_SIZE: usize = 512;
#[cfg(not(feature = "gpu"))]
const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 50;
const LR_MAX: f64 = 5e-4;
const LR_MIN: f64 = 1e-5;
const EARLY_STOPPING_PATIENCE: usize = 5;

// AlphaZero-specific training parameters
const AZ_NUM_EPOCHS: usize = 40;
const AZ_LR_MAX: f64 = 2e-4;
const AZ_LR_MIN: f64 = 1e-5;
const AZ_EARLY_STOPPING_PATIENCE: usize = 10;
const TRAIN_SPLIT_RATIO: f64 = 0.9;
const NUM_ACTIONS: usize = 24;
const VALUE_LOSS_WEIGHT: f32 = 0.5;

/// MSE loss between predicted value and transformed targets.
fn value_mse_loss<B: Backend>(
    value: Tensor<B, 2>,
    targets: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let batch_size = value.dims()[0];
    let transformed: Vec<f32> = targets.iter().map(|&v| value_transform(v)).collect();
    let target_tensor = Tensor::<B, 1>::from_floats(transformed.as_slice(), device)
        .reshape([batch_size, 1]);
    let diff = value - target_tensor;
    diff.clone().mul(diff).mean()
}

fn cosine_lr(epoch: usize, total_epochs: usize) -> f64 {
    LR_MIN
        + 0.5
            * (LR_MAX - LR_MIN)
            * (1.0 + (std::f64::consts::PI * epoch as f64 / total_epochs as f64).cos())
}

/// Cross-entropy loss for policy: -sum(target * log_softmax(logits)) / batch_size.
/// `targets` can be one-hot (u8 index) or soft (f32 distribution).
fn cross_entropy_loss_hard<B: Backend>(logits: Tensor<B, 2>, targets: &[u8], device: &B::Device) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];
    let max_logits = logits.clone().max_dim(1);
    let shifted = logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp;

    let mut target_one_hot = vec![0.0f32; batch_size * NUM_ACTIONS];
    for (i, &t) in targets.iter().enumerate() {
        target_one_hot[i * NUM_ACTIONS + t as usize] = 1.0;
    }
    let target_tensor = Tensor::<B, 1>::from_floats(target_one_hot.as_slice(), device)
        .reshape([batch_size, NUM_ACTIONS]);

    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

/// Cross-entropy loss for soft policy targets (MCTS visit distribution).
fn cross_entropy_loss_soft<B: Backend>(logits: Tensor<B, 2>, targets_flat: &[f32], device: &B::Device) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];
    let max_logits = logits.clone().max_dim(1);
    let shifted = logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp;

    let target_tensor = Tensor::<B, 1>::from_floats(targets_flat, device)
        .reshape([batch_size, NUM_ACTIONS]);

    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

fn main() {
    std::fs::create_dir_all("artifacts").expect("Failed to create artifacts directory");

    let args: Vec<String> = std::env::args().collect();
    let alphazero_mode = args.iter().any(|a| a == "--alphazero");
    let data_dir = args.iter().position(|a| a == "--data-dir")
        .map(|i| args[i + 1].clone());

    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    if alphazero_mode {
        println!("Mode: AlphaZero (Policy CE + Value MSE)");
        train_alphazero(data_dir.as_deref());
    } else {
        println!("Mode: Supervised (Policy CE only)");
        train_supervised();
    }
}

// ---------------------------------------------------------------------------
// Supervised training (from generate-data)
// ---------------------------------------------------------------------------

fn train_supervised() {
    let device: <TrainBackend as Backend>::Device = Default::default();
    let data_path = "data/training_data.bin";

    println!("Loading dataset from {}...", data_path);
    let dataset = Dataset::load(data_path).expect("Failed to load dataset");
    let num_samples = dataset.samples.len();
    println!("Loaded {} samples", num_samples);

    let split = (num_samples as f64 * TRAIN_SPLIT_RATIO) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..];
    println!("Train: {}, Val: {}", train_samples.len(), val_samples.len());

    let config = PuyoNetConfig::new();
    let mut model = config.init::<TrainBackend>(&device);
    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init();
    let mut best_val_loss = f32::MAX;
    let mut patience_counter = 0usize;

    for epoch in 0..NUM_EPOCHS {
        let lr = cosine_lr(epoch, NUM_EPOCHS);
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        shuffle_indices(&mut indices, epoch as u64);

        for batch_start in (0..train_samples.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_samples.len());
            let batch_size = batch_end - batch_start;
            if batch_size == 0 { break; }

            let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut context_data = Vec::with_capacity(batch_size * CONTEXT_TENSOR_SIZE);
            let mut target_actions = Vec::with_capacity(batch_size);
            let mut value_targets = Vec::with_capacity(batch_size);

            for &idx in &indices[batch_start..batch_end] {
                let sample = &train_samples[idx];
                board_data.extend_from_slice(&sample.board_data);
                context_data.extend_from_slice(&sample.context_data);
                target_actions.push(sample.action_index);
                value_targets.push(sample.value_target);
            }

            let board_inputs = Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device)
                .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
            let context_inputs = Tensor::<TrainBackend, 1>::from_floats(context_data.as_slice(), &device)
                .reshape([batch_size, CONTEXT_TENSOR_SIZE]);

            let (logits, value) = model.forward(board_inputs, context_inputs);
            let policy_loss = cross_entropy_loss_hard(logits, &target_actions, &device);

            let value_loss = value_mse_loss(value, &value_targets, &device);

            let loss = policy_loss + value_loss * VALUE_LOSS_WEIGHT;

            let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            epoch_loss += loss_val;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = train_samples.len().div_ceil(BATCH_SIZE);
                eprint!("\r  batch {}/{} loss={:.6}", num_batches, total_batches, epoch_loss / num_batches as f32);
            }
        }
        eprintln!();

        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let val_loss = compute_val_loss_supervised(&val_model, val_samples, &val_device);

        let avg_train_loss = epoch_loss / num_batches as f32;
        println!("Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.6}", epoch + 1, NUM_EPOCHS, avg_train_loss, val_loss, lr);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
            model.valid().save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new()).expect("Failed to save model");
            println!("  -> Best model saved (val_loss={:.6})", val_loss);
        } else {
            patience_counter += 1;
            println!("  -> No improvement ({}/{})", patience_counter, EARLY_STOPPING_PATIENCE);
            if patience_counter >= EARLY_STOPPING_PATIENCE {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }
    }

    println!("Training complete. Best val_loss={:.6}", best_val_loss);
}

fn compute_val_loss_supervised(
    model: &puyo_nn::model::PuyoNet<InnerBackend>,
    val_samples: &[puyo_trainer::data::Sample],
    device: &<InnerBackend as Backend>::Device,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(val_samples.len());
        let batch_size = batch_end - batch_start;
        if batch_size == 0 { break; }

        let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut context_data = Vec::with_capacity(batch_size * CONTEXT_TENSOR_SIZE);
        let mut target_actions = Vec::with_capacity(batch_size);
        let mut value_targets = Vec::with_capacity(batch_size);

        for sample in &val_samples[batch_start..batch_end] {
            board_data.extend_from_slice(&sample.board_data);
            context_data.extend_from_slice(&sample.context_data);
            target_actions.push(sample.action_index);
            value_targets.push(sample.value_target);
        }

        let board_inputs = Tensor::<InnerBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
        let context_inputs = Tensor::<InnerBackend, 1>::from_floats(context_data.as_slice(), device)
            .reshape([batch_size, CONTEXT_TENSOR_SIZE]);

        let (logits, value) = model.forward(board_inputs, context_inputs);
        let policy_loss = cross_entropy_loss_hard(logits, &target_actions, device);

        let value_loss = value_mse_loss(value, &value_targets, device);

        let loss_val = policy_loss.into_data().to_vec::<f32>().unwrap()[0]
            + value_loss.into_data().to_vec::<f32>().unwrap()[0] * VALUE_LOSS_WEIGHT;
        total_loss += loss_val;
        num_batches += 1;
    }

    total_loss / num_batches.max(1) as f32
}

// ---------------------------------------------------------------------------
// AlphaZero training (from self-play data)
// ---------------------------------------------------------------------------

fn train_alphazero(data_dir: Option<&str>) {
    let device: <TrainBackend as Backend>::Device = Default::default();

    let dataset = if let Some(dir) = data_dir {
        // Replay buffer mode: load all alphazero_iter_*.bin files from the directory
        println!("Loading AlphaZero datasets from {}...", dir);
        let mut paths: Vec<String> = std::fs::read_dir(dir)
            .expect("Failed to read data directory")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?.to_string();
                if name.starts_with("alphazero_iter_") && name.ends_with(".bin") {
                    Some(path.to_str()?.to_string())
                } else {
                    None
                }
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            panic!("No alphazero_iter_*.bin files found in {}", dir);
        }
        println!("Found {} data files", paths.len());
        AlphaZeroDataset::load_multiple(&paths).expect("Failed to load datasets")
    } else {
        // Legacy single-file mode
        let data_path = "data/alphazero_data.bin";
        println!("Loading AlphaZero dataset from {}...", data_path);
        AlphaZeroDataset::load(data_path).expect("Failed to load dataset")
    };
    let num_samples = dataset.samples.len();
    println!("Loaded {} total samples", num_samples);

    // Shuffle before split to avoid systematic bias (games are sequential)
    let mut sample_indices: Vec<usize> = (0..num_samples).collect();
    shuffle_indices(&mut sample_indices, 12345);
    let mut shuffled_samples = Vec::with_capacity(num_samples);
    for &idx in &sample_indices {
        shuffled_samples.push(dataset.samples[idx].clone());
    }
    let dataset = AlphaZeroDataset { samples: shuffled_samples };

    let split = (num_samples as f64 * TRAIN_SPLIT_RATIO) as usize;
    let train_samples = &dataset.samples[..split];
    let val_samples = &dataset.samples[split..];
    println!("Train: {}, Val: {}", train_samples.len(), val_samples.len());

    let config = PuyoNetConfig::new();

    // Try to load existing model, otherwise init fresh
    // Use catch_unwind because burn may panic on incompatible model files
    let mut model = {
        let device_clone = device.clone();
        let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            config
                .init::<TrainBackend>(&device_clone)
                .load_file(MODEL_PATH, &recorder, &device_clone)
        }));
        match load_result {
            Ok(Ok(m)) => {
                println!("Loaded existing model from {}", MODEL_PATH);
                m
            }
            Ok(Err(e)) => {
                println!(
                    "Failed to load model from {}: {}. Initializing fresh",
                    MODEL_PATH, e
                );
                config.init::<TrainBackend>(&device)
            }
            Err(_) => {
                println!(
                    "Model file {} is incompatible with current architecture. Initializing fresh",
                    MODEL_PATH
                );
                config.init::<TrainBackend>(&device)
            }
        }
    };

    let num_epochs = AZ_NUM_EPOCHS;
    let patience_limit = AZ_EARLY_STOPPING_PATIENCE;

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .init();
    let mut best_val_loss = f32::MAX;
    let mut patience_counter = 0usize;

    println!("AlphaZero training: {} epochs, LR {:.0e}->{:.0e}, patience={}",
        num_epochs, AZ_LR_MAX, AZ_LR_MIN, patience_limit);

    for epoch in 0..num_epochs {
        let lr = AZ_LR_MIN
            + 0.5 * (AZ_LR_MAX - AZ_LR_MIN)
            * (1.0 + (std::f64::consts::PI * epoch as f64 / num_epochs as f64).cos());
        let mut epoch_policy_loss = 0.0f32;
        let mut epoch_value_loss = 0.0f32;
        let mut num_batches = 0;

        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        shuffle_indices(&mut indices, epoch as u64);

        for batch_start in (0..train_samples.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_samples.len());
            let batch_size = batch_end - batch_start;
            if batch_size == 0 { break; }

            let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut context_data = Vec::with_capacity(batch_size * CONTEXT_TENSOR_SIZE);
            let mut policy_targets = Vec::with_capacity(batch_size * NUM_ACTIONS);
            let mut value_targets = Vec::with_capacity(batch_size);

            for &idx in &indices[batch_start..batch_end] {
                let sample = &train_samples[idx];
                board_data.extend_from_slice(&sample.board_data);
                context_data.extend_from_slice(&sample.context_data);
                policy_targets.extend_from_slice(&sample.mcts_policy);
                value_targets.push(sample.value_target);
            }

            let board_inputs = Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device)
                .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
            let context_inputs = Tensor::<TrainBackend, 1>::from_floats(context_data.as_slice(), &device)
                .reshape([batch_size, CONTEXT_TENSOR_SIZE]);

            // Forward pass
            let (logits, value) = model.forward(board_inputs, context_inputs);

            // Policy loss: cross-entropy with soft MCTS targets
            let policy_loss = cross_entropy_loss_soft(logits, &policy_targets, &device);

            let value_loss = value_mse_loss(value, &value_targets, &device);

            // Total loss (value loss weighted to balance with policy loss)
            let p_loss_val = policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let v_loss_val = value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let total_loss = policy_loss + value_loss * VALUE_LOSS_WEIGHT;

            epoch_policy_loss += p_loss_val;
            epoch_value_loss += v_loss_val;
            num_batches += 1;

            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = train_samples.len().div_ceil(BATCH_SIZE);
                eprint!(
                    "\r  batch {}/{} p_loss={:.4} v_loss={:.4}",
                    num_batches, total_batches,
                    epoch_policy_loss / num_batches as f32,
                    epoch_value_loss / num_batches as f32,
                );
            }
        }
        eprintln!();

        let avg_p = epoch_policy_loss / num_batches as f32;
        let avg_v = epoch_value_loss / num_batches as f32;
        let avg_total = avg_p + avg_v * VALUE_LOSS_WEIGHT;

        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let (val_p, val_v) = compute_val_loss_alphazero(&val_model, val_samples, &val_device);
        let val_total = val_p + val_v * VALUE_LOSS_WEIGHT;

        println!(
            "Epoch {}/{}: train(p={:.6}, v={:.6}, t={:.6}), val(p={:.6}, v={:.6}, t={:.6}), lr={:.6}",
            epoch + 1, num_epochs, avg_p, avg_v, avg_total, val_p, val_v, val_total, lr,
        );

        if val_total < best_val_loss {
            best_val_loss = val_total;
            patience_counter = 0;
            model.valid().save_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new()).expect("Failed to save model");
            println!("  -> Best model saved (val_loss={:.6})", val_total);
        } else {
            patience_counter += 1;
            println!("  -> No improvement ({}/{})", patience_counter, patience_limit);
            if patience_counter >= patience_limit {
                println!("Early stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }
    }

    println!("AlphaZero training complete. Best val_loss={:.6}", best_val_loss);
}

fn compute_val_loss_alphazero(
    model: &puyo_nn::model::PuyoNet<InnerBackend>,
    val_samples: &[puyo_trainer::data::AlphaZeroSample],
    device: &<InnerBackend as Backend>::Device,
) -> (f32, f32) {
    let mut total_policy_loss = 0.0f32;
    let mut total_value_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(val_samples.len());
        let batch_size = batch_end - batch_start;
        if batch_size == 0 { break; }

        let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut context_data = Vec::with_capacity(batch_size * CONTEXT_TENSOR_SIZE);
        let mut policy_targets = Vec::with_capacity(batch_size * NUM_ACTIONS);
        let mut value_targets = Vec::with_capacity(batch_size);

        for sample in &val_samples[batch_start..batch_end] {
            board_data.extend_from_slice(&sample.board_data);
            context_data.extend_from_slice(&sample.context_data);
            policy_targets.extend_from_slice(&sample.mcts_policy);
            value_targets.push(sample.value_target);
        }

        let board_inputs = Tensor::<InnerBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
        let context_inputs = Tensor::<InnerBackend, 1>::from_floats(context_data.as_slice(), device)
            .reshape([batch_size, CONTEXT_TENSOR_SIZE]);

        let (logits, value) = model.forward(board_inputs, context_inputs);

        let policy_loss = cross_entropy_loss_soft(logits, &policy_targets, device);
        total_policy_loss += policy_loss.into_data().to_vec::<f32>().unwrap()[0];

        let value_loss = value_mse_loss(value, &value_targets, device);
        total_value_loss += value_loss.into_data().to_vec::<f32>().unwrap()[0];

        num_batches += 1;
    }

    let n = num_batches.max(1) as f32;
    (total_policy_loss / n, total_value_loss / n)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shuffle_indices(indices: &mut [usize], seed: u64) {
    let mut rng_state = seed + 42;
    const LCG_MULTIPLIER: u64 = 6364136223846793005;
    const LCG_INCREMENT: u64 = 1;
    for i in (1..indices.len()).rev() {
        rng_state = rng_state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}
