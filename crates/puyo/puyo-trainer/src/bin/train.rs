use burn::backend::Autodiff;
use burn::backend::CudaJit;
use burn::module::AutodiffModule;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{SgdConfig, GradientsParams, GradientsAccumulator, Optimizer};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::AutodiffBackend;

use puyo_core::board::{COLS, ROWS};
use puyo_core::config::NUM_ACTIONS;
use puyo_core::state::{CONTEXT_TENSOR_SIZE, NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::PuyoNetConfig;
use az_framework::value_transform::value_transform;
use puyo_core::rand::time_seed;
use puyo_trainer::data::{AlphaZeroDataset, Dataset};

type TrainBackend = Autodiff<CudaJit<f32>>;
type InnerBackend = <TrainBackend as AutodiffBackend>::InnerBackend;

const MODEL_PATH: &str = "artifacts/puyo_model";
const BATCH_SIZE: usize = 512;
const NUM_EPOCHS: usize = 50;
const LR_MAX: f64 = 0.1;
const LR_MIN: f64 = 1e-3;
const EARLY_STOPPING_PATIENCE: usize = 5;

// AlphaZero-specific training parameters (step-based)
const AZ_NUM_STEPS: usize = 1000;
const ACCUM_STEPS: usize = 4; // gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS = 2048
// Global step LR schedule (AlphaZero-style 4-stage drop)
const AZ_LR_STAGES: [(usize, f64); 4] = [
    (0,      0.2),    // step 0〜100k: LR = 0.2
    (10000, 0.02),   // step 100k〜300k: LR = 0.02
    (30000, 0.002),  // step 300k〜500k: LR = 0.002
    (50000, 0.0002),  // step 500k〜: LR = 0.0002
];
const TRAIN_SPLIT_RATIO: f64 = 0.9;
const VALUE_LOSS_WEIGHT: f32 = 0.5;
const VALUE_SCALE: f32 = 15.0;

/// MSE loss between predicted value and transformed targets.
fn value_mse_loss<B: Backend>(
    value: Tensor<B, 2>,
    targets: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let batch_size = value.dims()[0];
    let transformed: Vec<f32> = targets.iter().map(|&v| value_transform(v, VALUE_SCALE)).collect();
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
/// Invalid actions (target == 0.0) are masked out of the softmax computation
/// so that no gradient flows through their logits.
fn cross_entropy_loss_soft<B: Backend>(logits: Tensor<B, 2>, targets_flat: &[f32], device: &B::Device) -> Tensor<B, 1> {
    let batch_size = logits.dims()[0];

    // Build mask: -1e9 for invalid actions (target == 0), 0 for valid
    let mut mask_data = vec![0.0f32; batch_size * NUM_ACTIONS];
    for i in 0..batch_size {
        for a in 0..NUM_ACTIONS {
            if targets_flat[i * NUM_ACTIONS + a] == 0.0 {
                mask_data[i * NUM_ACTIONS + a] = -1e9;
            }
        }
    }
    let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        .reshape([batch_size, NUM_ACTIONS]);

    let masked_logits = logits + mask_tensor;
    let max_logits = masked_logits.clone().max_dim(1);
    let shifted = masked_logits - max_logits;
    let exp = shifted.clone().exp();
    let sum_exp = exp.sum_dim(1);
    let log_sum_exp = sum_exp.log();
    let log_softmax = shifted - log_sum_exp;

    let target_tensor = Tensor::<B, 1>::from_floats(targets_flat, device)
        .reshape([batch_size, NUM_ACTIONS]);

    let selected = (log_softmax * target_tensor).sum();
    selected.neg() / (batch_size as f32)
}

/// Get learning rate based on global step (AlphaZero-style stage drop).
fn az_lr_for_global_step(global_step: usize) -> f64 {
    let mut lr = AZ_LR_STAGES[0].1;
    for &(threshold, stage_lr) in &AZ_LR_STAGES {
        if global_step >= threshold {
            lr = stage_lr;
        }
    }
    lr
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let alphazero_mode = args.iter().any(|a| a == "--alphazero");
    let data_dir = args.iter().position(|a| a == "--data-dir")
        .map(|i| args[i + 1].clone());
    let global_step = args.iter().position(|a| a == "--global-step")
        .map(|i| args[i + 1].parse::<usize>().expect("--global-step requires integer"))
        .unwrap_or(0);
    let model_path = args.iter().position(|a| a == "--model-path")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| MODEL_PATH.to_string());
    let artifacts_dir = args.iter().position(|a| a == "--artifacts-dir")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "artifacts".to_string());
    let batch_size = args.iter().position(|a| a == "--batch-size")
        .map(|i| args[i + 1].parse::<usize>().expect("--batch-size requires integer"))
        .unwrap_or(BATCH_SIZE);

    std::fs::create_dir_all(&artifacts_dir).expect("Failed to create artifacts directory");

    println!("Backend: CUDA (GPU)");

    if alphazero_mode {
        println!("Mode: AlphaZero (Policy CE + Value MSE)");
        train_alphazero(data_dir.as_deref(), global_step, &model_path, &artifacts_dir, batch_size);
    } else {
        println!("Mode: Supervised (Policy CE only)");
        train_supervised(&model_path, batch_size);
    }
}

// ---------------------------------------------------------------------------
// Supervised training (from generate-data)
// ---------------------------------------------------------------------------

fn train_supervised(model_path: &str, batch_size: usize) {
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
    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut best_val_loss = f32::MAX;
    let mut patience_counter = 0usize;

    for epoch in 0..NUM_EPOCHS {
        let lr = cosine_lr(epoch, NUM_EPOCHS);
        let mut epoch_loss = 0.0f32;
        let mut num_batches = 0;

        let mut indices: Vec<usize> = (0..train_samples.len()).collect();
        shuffle_indices(&mut indices);

        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
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

            let loss_val = loss.clone().into_data().to_vec::<f32>().expect("Failed to extract loss")[0];
            epoch_loss += loss_val;
            num_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            if num_batches % 50 == 0 {
                let total_batches = train_samples.len().div_ceil(batch_size);
                eprint!("\r  batch {}/{} loss={:.6}", num_batches, total_batches, epoch_loss / num_batches as f32);
            }
        }
        eprintln!();

        let val_model = model.valid();
        let val_device: <InnerBackend as Backend>::Device = Default::default();
        let val_loss = compute_val_loss_supervised(&val_model, val_samples, &val_device, batch_size);

        let avg_train_loss = epoch_loss / num_batches as f32;
        println!("Epoch {}/{}: train_loss={:.6}, val_loss={:.6}, lr={:.6}", epoch + 1, NUM_EPOCHS, avg_train_loss, val_loss, lr);

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
            model.valid().save_file(model_path, &BinFileRecorder::<FullPrecisionSettings>::new()).expect("Failed to save model");
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
    batch_size: usize,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut num_batches = 0;

    for batch_start in (0..val_samples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(val_samples.len());
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

        let loss_val = policy_loss.into_data().to_vec::<f32>().expect("Failed to extract policy loss")[0]
            + value_loss.into_data().to_vec::<f32>().expect("Failed to extract value loss")[0] * VALUE_LOSS_WEIGHT;
        total_loss += loss_val;
        num_batches += 1;
    }

    total_loss / num_batches.max(1) as f32
}

// ---------------------------------------------------------------------------
// AlphaZero training (from self-play data)
// ---------------------------------------------------------------------------

fn train_alphazero(data_dir: Option<&str>, global_step_start: usize, model_path: &str, artifacts_dir: &str, batch_size: usize) {
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

    // All data used for training (no val split — performance judged by self-play reward)
    let all_perms = puyo_trainer::data::all_color_permutations();
    let train_samples = dataset.samples;

    println!("Train: {} samples (color augmented on-the-fly, no val split)", train_samples.len());

    let config = PuyoNetConfig::new();

    // Try to load existing model, otherwise init fresh
    let mut model = {
        let device_clone = device.clone();
        let path = model_path.to_string();
        let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            config
                .init::<TrainBackend>(&device_clone)
                .load_file(&path, &recorder, &device_clone)
        }));
        match load_result {
            Ok(Ok(m)) => {
                println!("Loaded existing model from {}", model_path);
                m
            }
            Ok(Err(e)) => {
                println!(
                    "Failed to load model from {}: {}. Initializing fresh",
                    model_path, e
                );
                config.init::<TrainBackend>(&device)
            }
            Err(_) => {
                println!(
                    "Model file {} is incompatible with current architecture. Initializing fresh",
                    model_path
                );
                config.init::<TrainBackend>(&device)
            }
        }
    };

    let mut optim = SgdConfig::new()
        .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_gradient_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let start_lr = az_lr_for_global_step(global_step_start);
    let end_lr = az_lr_for_global_step(global_step_start + AZ_NUM_STEPS);
    println!("AlphaZero training: {} steps, global_step={}, LR {:.0e} (end ~{:.0e})",
        AZ_NUM_STEPS, global_step_start, start_lr, end_lr);

    let mut running_p_loss = 0.0f32;
    let mut running_v_loss = 0.0f32;
    let mut running_count = 0usize;
    let mut accum: GradientsAccumulator<puyo_nn::model::PuyoNet<TrainBackend>> = GradientsAccumulator::new();

    for step in 0..AZ_NUM_STEPS {
        let lr = az_lr_for_global_step(global_step_start + step);

        // Gradient accumulation: run ACCUM_STEPS micro-batches per optimizer step
        for _micro in 0..ACCUM_STEPS {
            let batch_size = batch_size.min(train_samples.len());
            let mut board_data = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut context_data = Vec::with_capacity(batch_size * CONTEXT_TENSOR_SIZE);
            let mut policy_targets = Vec::with_capacity(batch_size * NUM_ACTIONS);
            let mut value_targets = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let idx = (time_seed() >> 33) as usize % train_samples.len();
                let sample = &train_samples[idx];

                let perm_idx = (time_seed() >> 33) as usize % all_perms.len();
                let perm = &all_perms[perm_idx];

                let mut bd = sample.board_data.clone();
                let mut cd = sample.context_data.clone();
                puyo_trainer::data::apply_color_perm_board(&mut bd, perm);
                puyo_trainer::data::apply_color_perm_context(&mut cd, perm);

                board_data.extend_from_slice(&bd);
                context_data.extend_from_slice(&cd);
                policy_targets.extend_from_slice(&sample.mcts_policy);
                value_targets.push(sample.value_target);
            }

            let board_inputs = Tensor::<TrainBackend, 1>::from_floats(board_data.as_slice(), &device)
                .reshape([batch_size, NUM_CHANNELS, ROWS, COLS]);
            let context_inputs = Tensor::<TrainBackend, 1>::from_floats(context_data.as_slice(), &device)
                .reshape([batch_size, CONTEXT_TENSOR_SIZE]);

            let (logits, value) = model.forward(board_inputs, context_inputs);

            let policy_loss = cross_entropy_loss_soft(logits, &policy_targets, &device);
            let value_loss = value_mse_loss(value, &value_targets, &device);

            let p_loss_val = policy_loss.clone().into_data().to_vec::<f32>().expect("Failed to extract policy loss")[0];
            let v_loss_val = value_loss.clone().into_data().to_vec::<f32>().expect("Failed to extract value loss")[0];
            // Scale loss by 1/ACCUM_STEPS so accumulated gradients average correctly
            let total_loss = (policy_loss + value_loss * VALUE_LOSS_WEIGHT) / (ACCUM_STEPS as f32);

            running_p_loss += p_loss_val;
            running_v_loss += v_loss_val;
            running_count += 1;

            let grads = total_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            accum.accumulate(&model, grads);
        }

        // Apply accumulated gradients
        let grads = accum.grads();
        model = optim.step(lr, model, grads);

        if (step + 1) % 50 == 0 {
            eprint!(
                "\r  step {}/{} p_loss={:.4} v_loss={:.4} lr={:.6}",
                step + 1, AZ_NUM_STEPS,
                running_p_loss / running_count as f32,
                running_v_loss / running_count as f32,
                lr,
            );
        }
    }
    eprintln!();

    let final_p = running_p_loss / running_count as f32;
    let final_v = running_v_loss / running_count as f32;

    // Save model (always — no val-based selection, performance judged by self-play)
    model.valid().save_file(model_path, &BinFileRecorder::<FullPrecisionSettings>::new()).expect("Failed to save model");

    let global_step_end = global_step_start + AZ_NUM_STEPS;
    println!("AlphaZero training complete. final train_loss(p={:.6}, v={:.6}) global_step={}", final_p, final_v, global_step_end);

    // Write final global step to file for loop script
    let global_step_path = format!("{}/global_step.txt", artifacts_dir);
    std::fs::write(&global_step_path, global_step_end.to_string())
        .expect("Failed to write global_step.txt");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shuffle_indices(indices: &mut [usize]) {
    for i in (1..indices.len()).rev() {
        let j = (time_seed() >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}
