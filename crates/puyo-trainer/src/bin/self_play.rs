#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_ai::eval::{count_connectivity, count_potential_chains, Evaluator};
use puyo_ai::search;
use puyo_core::board::Board;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::{board_to_tensor_data, NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::{PuyoValueNet, PuyoValueNetConfig};
use puyo_core::board::{COLS, ROWS};

#[cfg(feature = "gpu")]
type TrainBackend = Autodiff<CudaJit<f32>>;
#[cfg(not(feature = "gpu"))]
type TrainBackend = Autodiff<NdArray>;

#[cfg(feature = "gpu")]
type InferBackend = CudaJit<f32>;
#[cfg(not(feature = "gpu"))]
type InferBackend = NdArray;

const MODEL_PATH: &str = "artifacts/puyo_model";
const OUTPUT_PATH: &str = "artifacts/puyo_model_selfplay";
const NUM_GAMES: u64 = 500;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f64 = 1e-4;
const EPSILON_START: f32 = 0.1;
const EPSILON_END: f32 = 0.01;
const TARGET_UPDATE_INTERVAL: u64 = 50;
const MAX_MOVES_PER_GAME: u32 = 50;
const GAME_OVER_PENALTY: f32 = -100.0;
const CONNECTIVITY_WEIGHT: f32 = 0.1;
const POTENTIAL_CHAIN_WEIGHT: f32 = 0.5;

/// NN-based evaluator for self-play search.
struct SelfPlayEvaluator<'a> {
    model: &'a PuyoValueNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
    mean: f32,
    std_dev: f32,
}

impl<'a> Evaluator for SelfPlayEvaluator<'a> {
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

/// Compute shaping reward for a board state (encourages connectivity and potential chains).
fn compute_shaping_reward(board: &Board) -> f32 {
    let connectivity = count_connectivity(board) as f32;
    let potential = count_potential_chains(board) as f32;
    CONNECTIVITY_WEIGHT * connectivity + POTENTIAL_CHAIN_WEIGHT * potential
}

fn main() {
    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let device: <TrainBackend as Backend>::Device = Default::default();
    let infer_device: <InferBackend as Backend>::Device = Default::default();

    // Load normalization params
    let norm_text = std::fs::read_to_string("artifacts/norm_params.txt")
        .expect("Failed to load norm_params.txt. Run 'train' first.");
    let mut lines = norm_text.lines();
    let mean: f32 = lines.next().unwrap().parse().unwrap();
    let std_dev: f32 = lines.next().unwrap().parse().unwrap();
    println!("Normalization: mean={:.4}, std={:.4}", mean, std_dev);

    // Load pre-trained model for training
    let config = PuyoValueNetConfig::new();
    let mut model: PuyoValueNet<TrainBackend> = config
        .init(&device)
        .load_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new(), &device)
        .expect("Failed to load model. Run 'train' first.");

    // Target network (frozen copy for stable targets)
    let mut target_model: PuyoValueNet<InferBackend> = config
        .init(&infer_device)
        .load_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new(), &infer_device)
        .expect("Failed to load target model");

    let mut optim = AdamConfig::new().init();

    for game_idx in 0..NUM_GAMES {
        let epsilon = EPSILON_START
            + (EPSILON_END - EPSILON_START) * (game_idx as f32 / NUM_GAMES as f32);

        let seed = 100_000 + game_idx;
        let mut game = GameState::new(seed);

        // Collect trajectory: (board_data, reward)
        let mut trajectory: Vec<([f32; TENSOR_SIZE], f32)> = Vec::new();
        let mut move_count = 0u32;

        while game.phase == GamePhase::Falling && move_count < MAX_MOVES_PER_GAME {
            let current_piece = match &game.current_piece {
                Some(fp) => fp.piece,
                None => break,
            };

            let board_data = board_to_tensor_data(&game.board);

            // Epsilon-greedy: with probability epsilon, use random placement
            let rng_val = simple_rng(seed + game.total_pieces as u64);
            let use_random = (rng_val as f32 / u64::MAX as f32) < epsilon;

            let chain_result = if use_random {
                // Random placement
                let placements =
                    puyo_ai::placement::enumerate_placements(&game.board, &current_piece);
                if placements.is_empty() {
                    break;
                }
                let idx = (rng_val as usize) % placements.len();
                game.apply_placement(&placements[idx])
            } else {
                // NN-guided placement
                let evaluator = SelfPlayEvaluator {
                    model: &target_model,
                    device: infer_device.clone(),
                    mean,
                    std_dev,
                };
                let result = search::find_best_move(
                    &game.board,
                    &current_piece,
                    &game.next_piece,
                    &evaluator,
                );
                match result {
                    Some(r) => game.apply_placement(&r.best_placement),
                    None => break,
                }
            };

            move_count += 1;
            let chain_reward = (chain_result.chain_count as f32).powi(2);
            let shaping_reward = compute_shaping_reward(&game.board);
            trajectory.push((board_data, chain_reward + shaping_reward));
        }

        let is_game_over = game.board.is_game_over();

        if trajectory.is_empty() {
            continue;
        }

        // Monte Carlo: compute discounted returns from the end of the episode
        let num_steps = trajectory.len();
        let mut returns = vec![0.0f32; num_steps];
        let terminal_bonus = if is_game_over { GAME_OVER_PENALTY } else { 0.0 };
        let mut running_return = terminal_bonus;
        for t in (0..num_steps).rev() {
            let (_, reward) = &trajectory[t];
            running_return = reward + GAMMA * running_return;
            returns[t] = running_return;
        }

        // Update model for each step using Monte Carlo return as target
        for t in 0..num_steps {
            let (board_data, _) = &trajectory[t];
            let mc_target = (returns[t] - mean) / std_dev;

            let input = Tensor::<TrainBackend, 1>::from_floats(
                board_data.as_slice(),
                &device,
            )
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
            let prediction = model.forward(input);

            let target = Tensor::<TrainBackend, 1>::from_floats(
                [mc_target].as_slice(),
                &device,
            )
            .reshape([1, 1]);

            // MSE loss
            let diff = prediction - target;
            let loss = diff.clone().mul(diff).mean();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);
        }

        println!(
            "Game {:4}/{}: max_chain={:2}, moves={:2}, eps={:.3}, {}",
            game_idx + 1,
            NUM_GAMES,
            game.max_chain,
            move_count,
            epsilon,
            if is_game_over { "GAMEOVER" } else { "ok" },
        );

        // Update target network periodically
        if (game_idx + 1) % TARGET_UPDATE_INTERVAL == 0 {
            let valid_model = model.valid();
            valid_model
                .save_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new())
                .expect("Failed to save temp model");
            target_model = config
                .init(&infer_device)
                .load_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new(), &infer_device)
                .expect("Failed to load target model");
            println!("[TARGET UPDATE] game={}", game_idx + 1);
        }
    }

    // Save final model
    let final_model = model.valid();
    final_model
        .save_file(OUTPUT_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model");
    println!("Self-play model saved to {}", OUTPUT_PATH);
}

/// Simple deterministic RNG for epsilon-greedy.
fn simple_rng(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}
