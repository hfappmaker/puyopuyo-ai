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
use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::chain;
use puyo_core::game::GameState;
use puyo_core::piece::Placement;
use puyo_nn::encoding::{board_to_tensor_data, NUM_CHANNELS, TENSOR_SIZE};
use puyo_nn::model::{PuyoValueNet, PuyoValueNetConfig};

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

// ---------------------------------------------------------------------------
// 構造体定義
// ---------------------------------------------------------------------------

/// z-score 正規化パラメータ
struct NormParams {
    mean: f32,
    std_dev: f32,
}

impl NormParams {
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path)
            .expect("Failed to load norm_params.txt. Run 'train' first.");
        let mut lines = text.lines();
        let mean: f32 = lines.next().unwrap().parse().unwrap();
        let std_dev: f32 = lines.next().unwrap().parse().unwrap();
        NormParams { mean, std_dev }
    }

    fn normalize(&self, raw: f32) -> f32 {
        (raw - self.mean) / self.std_dev
    }

    /// ターゲットモデルで盤面の価値を非正規化して返す
    fn eval_target(
        &self,
        model: &PuyoValueNet<InferBackend>,
        board_data: &[f32; TENSOR_SIZE],
        device: &<InferBackend as Backend>::Device,
    ) -> f32 {
        let tensor = Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let output = model.forward(tensor);
        let normalized = output.into_data().to_vec::<f32>().unwrap()[0];
        normalized * self.std_dev + self.mean
    }
}

/// 報酬の統計カウンタ
struct RewardStats {
    negative: u64,
    zero: u64,
    positive: u64,
}

impl RewardStats {
    fn new() -> Self {
        RewardStats { negative: 0, zero: 0, positive: 0 }
    }

    fn record(&mut self, reward: f32) {
        if reward < 0.0 {
            self.negative += 1;
        } else if reward > 0.0 {
            self.positive += 1;
        } else {
            self.zero += 1;
        }
    }

    fn log_and_reset(&mut self, step: u64, game_count: u64, epsilon: f32) {
        println!(
            "[PROGRESS] step={}/{}, games={}, eps={:.3}, rewards(-1/0/+1)={}/{}/{}",
            step, TOTAL_STEPS, game_count, epsilon,
            self.negative, self.zero, self.positive,
        );
        *self = Self::new();
    }
}

/// 1ゲームセッションの状態
struct GameSession {
    game: GameState,
    seed: u64,
    game_count: u64,
    move_count: u32,
}

impl GameSession {
    fn new(initial_seed: u64) -> Self {
        GameSession {
            game: GameState::new(initial_seed),
            seed: initial_seed,
            game_count: 0,
            move_count: 0,
        }
    }

    fn reset(&mut self) {
        self.seed += 1;
        self.game = GameState::new(self.seed);
        self.game_count += 1;
        self.move_count = 0;
    }

    fn log_if_new_max_chain(&self, prev_max_chain: u32, step: u64) {
        if self.game.max_chain > prev_max_chain {
            println!(
                "[NEW MAX CHAIN] game={}, chain={}, step={}",
                self.game_count + 1, self.game.max_chain, step,
            );
        }
    }

    fn log_game_over_and_reset(&mut self, epsilon: f32, step: u64) {
        println!(
            "Game {:4}: max_chain={:2}, moves={:2}, eps={:.3}, step={}",
            self.game_count + 1, self.game.max_chain, self.move_count, epsilon, step,
        );
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// NN ベースの Evaluator（探索用）
// ---------------------------------------------------------------------------

struct SelfPlayEvaluator<'a> {
    model: &'a PuyoValueNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
    norm: &'a NormParams,
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
        (normalized * self.norm.std_dev + self.norm.mean) as f64
    }
}

// ---------------------------------------------------------------------------
// ヘルパー関数
// ---------------------------------------------------------------------------

/// εの線形減衰を計算
fn compute_epsilon(step: u64) -> f32 {
    let progress = step as f32 / TOTAL_STEPS.max(1) as f32;
    EPSILON_START + (EPSILON_END - EPSILON_START) * progress
}

/// ε-greedy で配置を選択。配置不能な場合は None を返す。
fn select_placement(
    session: &GameSession,
    target_model: &PuyoValueNet<InferBackend>,
    infer_device: &<InferBackend as Backend>::Device,
    norm: &NormParams,
    epsilon: f32,
) -> Option<Placement> {
    let current_piece = session.game.current_piece.as_ref()?.piece;

    let rng_val = simple_rng(session.seed + session.game.total_pieces as u64);
    let use_random = (rng_val as f32 / u64::MAX as f32) < epsilon;

    if use_random {
        let placements =
            puyo_ai::placement::enumerate_placements(&session.game.board, &current_piece);
        if placements.is_empty() {
            return None;
        }
        let idx = (rng_val as usize) % placements.len();
        Some(placements[idx])
    } else {
        let evaluator = SelfPlayEvaluator {
            model: target_model,
            device: infer_device.clone(),
            norm,
        };
        search::find_best_move(
            &session.game.board,
            &current_piece,
            &session.game.next_piece,
            &evaluator,
        )
        .map(|r| r.best_placement)
    }
}

/// TD(0) 更新を1ステップ実行する。
/// model を消費して更新済み model を返す（Burn の所有権セマンティクス対応）。
fn td_update(
    model: PuyoValueNet<TrainBackend>,
    optim: &mut impl Optimizer<PuyoValueNet<TrainBackend>, TrainBackend>,
    target_model: &PuyoValueNet<InferBackend>,
    state_data: &[f32; TENSOR_SIZE],
    reward: f32,
    next_data: &[f32; TENSOR_SIZE],
    device: &<TrainBackend as Backend>::Device,
    norm: &NormParams,
    infer_device: &<InferBackend as Backend>::Device,
) -> PuyoValueNet<TrainBackend> {
    let v_next = norm.eval_target(target_model, next_data, infer_device);
    let td_target = norm.normalize(reward + GAMMA * v_next);

    let input = Tensor::<TrainBackend, 1>::from_floats(state_data.as_slice(), device)
        .reshape([1, NUM_CHANNELS, ROWS, COLS]);
    let prediction = model.forward(input);
    let target = Tensor::<TrainBackend, 1>::from_floats([td_target].as_slice(), device)
        .reshape([1, 1]);
    let diff = prediction - target;
    let loss = diff.clone().mul(diff).mean();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    optim.step(LEARNING_RATE, model, grads)
}

/// ターゲットネットワークをオンラインモデルから同期する
fn sync_target_network(
    model: &PuyoValueNet<TrainBackend>,
    config: &PuyoValueNetConfig,
    infer_device: &<InferBackend as Backend>::Device,
) -> PuyoValueNet<InferBackend> {
    let valid_model = model.valid();
    valid_model
        .save_file(
            "/tmp/puyo_temp_model",
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save temp model");
    let target = config
        .init(infer_device)
        .load_file(
            "/tmp/puyo_temp_model",
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            infer_device,
        )
        .expect("Failed to load target model");
    println!("[TARGET UPDATE] step completed");
    target
}

/// TD更新後のステップ管理（カウンタ更新、ログ、ターゲット同期）
fn step_bookkeeping(
    step: &mut u64,
    reward: f32,
    stats: &mut RewardStats,
    session: &GameSession,
    epsilon: f32,
    model: &PuyoValueNet<TrainBackend>,
    target_model: &mut PuyoValueNet<InferBackend>,
    config: &PuyoValueNetConfig,
    infer_device: &<InferBackend as Backend>::Device,
) {
    *step += 1;
    stats.record(reward);

    if *step % LOG_INTERVAL == 0 {
        stats.log_and_reset(*step, session.game_count, epsilon);
    }

    if *step % TARGET_UPDATE_INTERVAL == 0 {
        *target_model = sync_target_network(model, config, infer_device);
    }
}

/// 決定論的 RNG（ε-greedy 用）
fn simple_rng(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

// ---------------------------------------------------------------------------
// 学習ループ
// ---------------------------------------------------------------------------

fn run_training_loop(
    mut model: PuyoValueNet<TrainBackend>,
    mut target_model: PuyoValueNet<InferBackend>,
    mut optim: impl Optimizer<PuyoValueNet<TrainBackend>, TrainBackend>,
    config: &PuyoValueNetConfig,
    device: &<TrainBackend as Backend>::Device,
    infer_device: &<InferBackend as Backend>::Device,
    norm: &NormParams,
) -> PuyoValueNet<InferBackend> {
    let mut session = GameSession::new(100_000);
    let mut step: u64 = 0;
    let mut stats = RewardStats::new();

    while step < TOTAL_STEPS {
        let epsilon = compute_epsilon(step);
        let board_data = board_to_tensor_data(&session.game.board);

        // 配置を選択
        let placement = match select_placement(&session, &target_model, infer_device, norm, epsilon)
        {
            Some(p) => p,
            None => {
                session.reset();
                continue;
            }
        };

        // ピースを配置（連鎖解決・ピース送りはしない）
        session.game.place_piece_only(&placement);
        session.move_count += 1;
        let placed_data = board_to_tensor_data(&session.game.board);

        // 連鎖判定
        let has_chain = !chain::find_groups(&session.game.board).is_empty();

        if has_chain {
            // 配置ステップ: reward=0, next=配置後盤面（消去前）
            model = td_update(
                model, &mut optim, &target_model,
                &board_data, 0.0, &placed_data, device, norm, infer_device,
            );
            step_bookkeeping(
                &mut step, 0.0, &mut stats, &session, epsilon,
                &model, &mut target_model, config, infer_device,
            );

            // 各連鎖ステップを解決: reward=1
            let mut prev_data = placed_data;
            let mut chain_count: u32 = 0;
            let mut total_score: u32 = 0;

            loop {
                if step >= TOTAL_STEPS {
                    break;
                }
                chain_count += 1;
                match chain::resolve_one_step(&mut session.game.board, chain_count) {
                    Some(chain_step) => {
                        total_score += chain_step.score;
                        let next_data = board_to_tensor_data(&session.game.board);
                        model = td_update(
                            model, &mut optim, &target_model,
                            &prev_data, 1.0, &next_data, device, norm, infer_device,
                        );
                        step_bookkeeping(
                            &mut step, 1.0, &mut stats, &session, epsilon,
                            &model, &mut target_model, config, infer_device,
                        );
                        prev_data = next_data;
                    }
                    None => {
                        chain_count -= 1;
                        break;
                    }
                }
            }

            // finalize + ゲームオーバー判定
            let prev_max = session.game.max_chain;
            session.game.finalize_after_chains(total_score, chain_count);
            session.log_if_new_max_chain(prev_max, step);

            if session.game.board.is_game_over() {
                session.log_game_over_and_reset(epsilon, step);

                if step < TOTAL_STEPS {
                    let next_data = board_to_tensor_data(&session.game.board);
                    model = td_update(
                        model, &mut optim, &target_model,
                        &prev_data, -1.0, &next_data, device, norm, infer_device,
                    );
                    step_bookkeeping(
                        &mut step, -1.0, &mut stats, &session, epsilon,
                        &model, &mut target_model, config, infer_device,
                    );
                }
            }
        } else {
            // 連鎖なし: finalize してゲームオーバー判定
            session.game.finalize_after_chains(0, 0);

            if session.game.board.is_game_over() {
                session.log_game_over_and_reset(epsilon, step);

                let next_data = board_to_tensor_data(&session.game.board);
                model = td_update(
                    model, &mut optim, &target_model,
                    &board_data, -1.0, &next_data, device, norm, infer_device,
                );
                step_bookkeeping(
                    &mut step, -1.0, &mut stats, &session, epsilon,
                    &model, &mut target_model, config, infer_device,
                );
            } else {
                // 生存: reward=1
                model = td_update(
                    model, &mut optim, &target_model,
                    &board_data, 1.0, &placed_data, device, norm, infer_device,
                );
                step_bookkeeping(
                    &mut step, 1.0, &mut stats, &session, epsilon,
                    &model, &mut target_model, config, infer_device,
                );
            }
        }
    }

    model.valid()
}

// ---------------------------------------------------------------------------
// エントリポイント
// ---------------------------------------------------------------------------

fn main() {
    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let device: <TrainBackend as Backend>::Device = Default::default();
    let infer_device: <InferBackend as Backend>::Device = Default::default();

    let norm = NormParams::load("artifacts/norm_params.txt");
    println!("Normalization: mean={:.4}, std={:.4}", norm.mean, norm.std_dev);

    let config = PuyoValueNetConfig::new();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model: PuyoValueNet<TrainBackend> = config
        .init(&device)
        .load_file(MODEL_PATH, &recorder, &device)
        .expect("Failed to load model. Run 'train' first.");

    let target_model: PuyoValueNet<InferBackend> = config
        .init(&infer_device)
        .load_file(MODEL_PATH, &BinFileRecorder::<FullPrecisionSettings>::new(), &infer_device)
        .expect("Failed to load target model");

    let optim = AdamConfig::new().init();

    let final_model = run_training_loop(
        model, target_model, optim, &config, &device, &infer_device, &norm,
    );

    final_model
        .save_file(OUTPUT_PATH, &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model");
    println!("Self-play model saved to {}", OUTPUT_PATH);
}
