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
use puyo_core::chain;
use puyo_core::game::GameState;
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
const TOTAL_STEPS: u64 = 200_000;
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f64 = 1e-4;
const EPSILON_START: f32 = 0.3;
const EPSILON_END: f32 = 0.01;
const TARGET_UPDATE_INTERVAL: u64 = 1000;
const LOG_INTERVAL: u64 = 100;

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

/// ターゲットモデルで盤面の価値を非正規化して返す
fn eval_target(
    model: &PuyoValueNet<InferBackend>,
    board_data: &[f32; TENSOR_SIZE],
    device: &<InferBackend as Backend>::Device,
    mean: f32,
    std_dev: f32,
) -> f32 {
    let tensor = Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), device)
        .reshape([1, NUM_CHANNELS, ROWS, COLS]);
    let output = model.forward(tensor);
    let normalized = output.into_data().to_vec::<f32>().unwrap()[0];
    normalized * std_dev + mean
}

/// TD(0) 更新を実行するマクロ
/// model の所有権を消費して新しい model を返すため、マクロで展開する
macro_rules! td_update {
    ($model:expr, $optim:expr, $target_model:expr,
     $state_data:expr, $reward:expr, $next_data:expr,
     $device:expr, $infer_device:expr, $mean:expr, $std_dev:expr) => {{
        let v_next = eval_target($target_model, $next_data, $infer_device, $mean, $std_dev);
        let td_target_raw = $reward + GAMMA * v_next;
        let td_target = (td_target_raw - $mean) / $std_dev;

        let input = Tensor::<TrainBackend, 1>::from_floats($state_data.as_slice(), $device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let prediction = $model.forward(input);
        let target = Tensor::<TrainBackend, 1>::from_floats([td_target].as_slice(), $device)
            .reshape([1, 1]);
        let diff = prediction - target;
        let loss = diff.clone().mul(diff).mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &$model);
        $model = $optim.step(LEARNING_RATE, $model, grads);
    }};
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

    let mut game_seed: u64 = 100_000;
    let mut game = GameState::new(game_seed);
    let mut game_count: u64 = 0;
    let mut move_count: u32 = 0;
    let mut step: u64 = 0;
    let mut reward_neg: u64 = 0; // -1 の回数
    let mut reward_zero: u64 = 0; // 0 の回数
    let mut reward_pos: u64 = 0; // +1 の回数

    while step < TOTAL_STEPS {
        // εの線形減衰
        let progress = step as f32 / TOTAL_STEPS.max(1) as f32;
        let epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * progress;

        // 配置前の盤面を記録
        let board_data = board_to_tensor_data(&game.board);

        // 現在のピースを取得
        let current_piece = match &game.current_piece {
            Some(fp) => fp.piece,
            None => {
                game_seed += 1;
                game = GameState::new(game_seed);
                game_count += 1;
                move_count = 0;
                continue;
            }
        };

        // ε-greedy で配置を選択
        let rng_val = simple_rng(game_seed + game.total_pieces as u64);
        let use_random = (rng_val as f32 / u64::MAX as f32) < epsilon;

        let placement = if use_random {
            let placements =
                puyo_ai::placement::enumerate_placements(&game.board, &current_piece);
            if placements.is_empty() {
                game_seed += 1;
                game = GameState::new(game_seed);
                game_count += 1;
                move_count = 0;
                continue;
            }
            let idx = (rng_val as usize) % placements.len();
            placements[idx]
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
                Some(r) => r.best_placement,
                None => {
                    game_seed += 1;
                    game = GameState::new(game_seed);
                    game_count += 1;
                    move_count = 0;
                    continue;
                }
            }
        };

        // ピースを配置（連鎖解決・ピース送りはしない）
        game.place_piece_only(&placement);
        move_count += 1;

        // 配置後・消去前の盤面
        let placed_data = board_to_tensor_data(&game.board);

        // 連鎖を1ステップずつ解決
        let has_chain = !chain::find_groups(&game.board).is_empty();
        let mut chain_count: u32 = 0;
        let mut total_score: u32 = 0;

        if has_chain {
            // 配置ステップ: reward=0, next=配置後盤面（消去前）
            td_update!(model, optim, &target_model,
                &board_data, 0.0_f32, &placed_data,
                &device, &infer_device, mean, std_dev);
            step += 1;
            reward_zero += 1;

            let mut prev_data = placed_data;

            // 各連鎖ステップ: reward=1
            loop {
                if step >= TOTAL_STEPS {
                    break;
                }
                chain_count += 1;
                match chain::resolve_one_step(&mut game.board, chain_count) {
                    Some(chain_step) => {
                        total_score += chain_step.score;
                        let next_data = board_to_tensor_data(&game.board);

                        td_update!(model, optim, &target_model,
                            &prev_data, 1.0_f32, &next_data,
                            &device, &infer_device, mean, std_dev);
                        step += 1;
                        reward_pos += 1;
                        prev_data = next_data;
                    }
                    None => {
                        chain_count -= 1;
                        break;
                    }
                }
            }

            // 連鎖ありパス: finalize + ゲームオーバー判定
            game.finalize_after_chains(total_score, chain_count);

            if game.board.is_game_over() {
                println!(
                    "Game {:4}: max_chain={:2}, moves={:2}, eps={:.3}, step={}",
                    game_count + 1, game.max_chain, move_count, epsilon, step,
                );

                game_seed += 1;
                game = GameState::new(game_seed);
                game_count += 1;
                move_count = 0;

                if step < TOTAL_STEPS {
                    let next_data = board_to_tensor_data(&game.board);
                    td_update!(model, optim, &target_model,
                        &prev_data, -1.0_f32, &next_data,
                        &device, &infer_device, mean, std_dev);
                    step += 1;
                    reward_neg += 1;
                }
            }
        } else {
            // 連鎖なし: finalize してゲームオーバー判定→報酬を決定
            game.finalize_after_chains(0, 0);

            if game.board.is_game_over() {
                // ゲームオーバー: reward=-1
                println!(
                    "Game {:4}: max_chain={:2}, moves={:2}, eps={:.3}, step={}",
                    game_count + 1, game.max_chain, move_count, epsilon, step,
                );

                game_seed += 1;
                game = GameState::new(game_seed);
                game_count += 1;
                move_count = 0;

                let next_data = board_to_tensor_data(&game.board);
                td_update!(model, optim, &target_model,
                    &board_data, -1.0_f32, &next_data,
                    &device, &infer_device, mean, std_dev);
                step += 1;
                reward_neg += 1;
            } else {
                // 生存報酬: reward=+1
                td_update!(model, optim, &target_model,
                    &board_data, 1.0_f32, &placed_data,
                    &device, &infer_device, mean, std_dev);
                step += 1;
                reward_pos += 1;
            }
        }

        // 進捗ログ
        if step > 0 && step % LOG_INTERVAL == 0 {
            println!(
                "[PROGRESS] step={}/{}, games={}, eps={:.3}, rewards(-1/0/+1)={}/{}/{}",
                step, TOTAL_STEPS, game_count, epsilon,
                reward_neg, reward_zero, reward_pos,
            );
            reward_neg = 0;
            reward_zero = 0;
            reward_pos = 0;
        }

        // ターゲットネットワーク更新
        if step > 0 && step % TARGET_UPDATE_INTERVAL == 0 {
            let valid_model = model.valid();
            valid_model
                .save_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new())
                .expect("Failed to save temp model");
            target_model = config
                .init(&infer_device)
                .load_file("/tmp/puyo_temp_model", &BinFileRecorder::<FullPrecisionSettings>::new(), &infer_device)
                .expect("Failed to load target model");
            println!("[TARGET UPDATE] step={}", step);
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
