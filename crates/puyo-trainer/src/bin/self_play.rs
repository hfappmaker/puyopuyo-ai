#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_ai::eval::Evaluator;
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
const NUM_GAMES: u64 = 5000;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f64 = 1e-4;
const EPSILON_HIGH: f32 = 0.3;   // カリキュラム序盤の探索率
const EPSILON_START: f32 = 0.1;  // 閾値達成後の開始値
const EPSILON_END: f32 = 0.01;   // 最終的な探索率
const CHAIN_WINDOW: usize = 20;  // 移動平均のウィンドウサイズ
const TARGET_UPDATE_INTERVAL: u64 = 20;
const MAX_MOVES_PER_GAME: u32 = 50;
const GAME_OVER_PENALTY: f32 = -500.0;

/// カリキュラム学習フェーズごとの最小連鎖数
/// 昇格条件: avg_chain >= min_chain + 1.0/min_chain
const CURRICULUM_MIN_CHAINS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8, 9];

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

    let mut recent_chains: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut curriculum_phase: usize = 0;
    let mut phase_start_game: Option<u64> = None;
    let mut total_update_steps: u64 = 0;
    let mut trained_game_count: u64 = 0;

    for game_idx in 0..NUM_GAMES {
        let avg_chain = if recent_chains.is_empty() {
            0.0
        } else {
            recent_chains.iter().sum::<u32>() as f32 / recent_chains.len() as f32
        };

        // カリキュラム昇格チェック
        let min_chain_cur = CURRICULUM_MIN_CHAINS[curriculum_phase];
        let promote_at = min_chain_cur as f32 + 1.0 / min_chain_cur as f32;
        if curriculum_phase + 1 < CURRICULUM_MIN_CHAINS.len() && avg_chain >= promote_at {
            curriculum_phase += 1;
            println!(
                "[CURRICULUM ADVANCE] phase={}, avg_chain={:.1}, min_chain>={}",
                curriculum_phase,
                avg_chain,
                CURRICULUM_MIN_CHAINS[curriculum_phase]
            );
        }

        // フェーズ0は探索率高め、昇格後はdecay
        let epsilon = if curriculum_phase == 0 {
            EPSILON_HIGH
        } else {
            if phase_start_game.is_none() {
                phase_start_game = Some(game_idx);
            }
            let start = phase_start_game.unwrap();
            let progress = (game_idx - start) as f32 / (NUM_GAMES - start).max(1) as f32;
            EPSILON_START + (EPSILON_END - EPSILON_START) * progress
        };

        let seed = 100_000 + game_idx;
        let mut game = GameState::new(seed);
        let mut trajectory: Vec<([f32; TENSOR_SIZE], f32)> = Vec::new();
        let mut move_count = 0u32;
        let mut max_chain_step: Option<usize> = None;

        while game.phase == GamePhase::Falling && move_count < MAX_MOVES_PER_GAME {
            let current_piece = match &game.current_piece {
                Some(fp) => fp.piece,
                None => break,
            };

            let board_data = board_to_tensor_data(&game.board);

            let rng_val = simple_rng(seed + game.total_pieces as u64);
            let use_random = (rng_val as f32 / u64::MAX as f32) < epsilon;

            let chain_result = if use_random {
                let placements =
                    puyo_ai::placement::enumerate_placements(&game.board, &current_piece);
                if placements.is_empty() {
                    break;
                }
                let idx = (rng_val as usize) % placements.len();
                game.apply_placement(&placements[idx])
            } else {
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
            let chain_reward = (chain_result.chain_count as f32).powi(3);
            trajectory.push((board_data, chain_reward));
            if chain_result.chain_count == game.max_chain && chain_result.chain_count > 0 {
                max_chain_step = Some(trajectory.len() - 1);
            }
        }

        let is_game_over = game.board.is_game_over();

        recent_chains.push_back(game.max_chain);
        if recent_chains.len() > CHAIN_WINDOW {
            recent_chains.pop_front();
        }

        if trajectory.is_empty() {
            continue;
        }

        // カリキュラム閾値未満の連鎖ゲームはスキップ
        let min_chain = CURRICULUM_MIN_CHAINS[curriculum_phase];
        if game.max_chain < min_chain {
            println!(
                "Game {:4}/{}: max_chain={:2}, moves={:2}, eps={:.3}, phase={}, steps={}, {} [SKIP min={}]",
                game_idx + 1,
                NUM_GAMES,
                game.max_chain,
                move_count,
                epsilon,
                curriculum_phase,
                total_update_steps,
                if is_game_over { "GAMEOVER" } else { "ok" },
                min_chain,
            );
            continue;
        }

        // Monte Carlo: compute discounted returns up to the step where max_chain was achieved
        let num_steps = match max_chain_step {
            Some(idx) => idx + 1,
            None => trajectory.len(),
        };
        let mut returns = vec![0.0f32; num_steps];
        let terminal_bonus = if is_game_over { GAME_OVER_PENALTY } else { 0.0 };
        let mut running_return = terminal_bonus;
        for t in (0..num_steps).rev() {
            let (_, reward) = &trajectory[t];
            running_return = reward + GAMMA * running_return;
            returns[t] = running_return;
        }

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

            let diff = prediction - target;
            let loss = diff.clone().mul(diff).mean();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);
            total_update_steps += 1;
        }
        trained_game_count += 1;

        println!(
            "Game {:4}/{}: max_chain={:2}, moves={:2}, eps={:.3}, phase={}, steps={}, {}",
            game_idx + 1,
            NUM_GAMES,
            game.max_chain,
            move_count,
            epsilon,
            curriculum_phase,
            total_update_steps,
            if is_game_over { "GAMEOVER" } else { "ok" },
        );

        // Update target network periodically
        if trained_game_count % TARGET_UPDATE_INTERVAL == 0 {
            let valid_model = model.valid();
            valid_model
                .save_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new())
                .expect("Failed to save temp model");
            target_model = config
                .init(&infer_device)
                .load_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new(), &infer_device)
                .expect("Failed to load target model");
            println!("[TARGET UPDATE] game={}, steps={}", game_idx + 1, total_update_steps);
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
