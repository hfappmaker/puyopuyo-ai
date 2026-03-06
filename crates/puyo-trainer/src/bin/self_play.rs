#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
use burn::backend::Autodiff;
#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
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
const GAMMA: f32 = 0.99;
const LEARNING_RATE: f64 = 1e-4;
const EPSILON_START: f32 = 0.3;
const EPSILON_END: f32 = 0.01;
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

/// 報酬の統計カウンタ（損失の累積も含む）
struct RewardStats {
    negative: u64,
    zero: u64,
    positive: u64,
    loss_sum: f64,
    loss_count: u64,
}

impl RewardStats {
    fn new() -> Self {
        RewardStats {
            negative: 0,
            zero: 0,
            positive: 0,
            loss_sum: 0.0,
            loss_count: 0,
        }
    }

    fn record(&mut self, reward: f32, loss: f32) {
        if reward < 0.0 {
            self.negative += 1;
        } else if reward > 0.0 {
            self.positive += 1;
        } else {
            self.zero += 1;
        }
        self.loss_sum += loss as f64;
        self.loss_count += 1;
    }

    fn log_and_reset(
        &mut self,
        step: u64,
        total_steps: u64,
        game_count: u64,
        epsilon: f32,
        game_stats: &mut GameStats,
    ) {
        let avg_loss = if self.loss_count > 0 {
            self.loss_sum / self.loss_count as f64
        } else {
            0.0
        };
        let (avg_chain, avg_moves) = game_stats.averages();
        println!(
            "[PROGRESS] step={}/{}, games={}, eps={:.3}, rewards(-1/0/+1)={}/{}/{}, loss={:.4}, avg_chain={:.1}, avg_moves={:.1}",
            step, total_steps, game_count, epsilon,
            self.negative, self.zero, self.positive,
            avg_loss, avg_chain, avg_moves,
        );
        *self = Self::new();
        *game_stats = GameStats::new();
    }
}

/// ゲームパフォーマンスの統計（収束確認用）
struct GameStats {
    chain_sum: u64,
    moves_sum: u64,
    count: u64,
}

impl GameStats {
    fn new() -> Self {
        GameStats {
            chain_sum: 0,
            moves_sum: 0,
            count: 0,
        }
    }

    fn record(&mut self, max_chain: u32, moves: u32) {
        self.chain_sum += max_chain as u64;
        self.moves_sum += moves as u64;
        self.count += 1;
    }

    fn averages(&self) -> (f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0);
        }
        (
            self.chain_sum as f64 / self.count as f64,
            self.moves_sum as f64 / self.count as f64,
        )
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
                self.game_count + 1,
                self.game.max_chain,
                step,
            );
        }
    }

    fn log_game_over_and_reset(&mut self, epsilon: f32, step: u64) {
        println!(
            "Game {:4}: max_chain={:2}, moves={:2}, eps={:.3}, step={}",
            self.game_count + 1,
            self.game.max_chain,
            self.move_count,
            epsilon,
            step,
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
fn compute_epsilon(step: u64, total_steps: u64) -> f32 {
    let progress = step as f32 / total_steps.max(1) as f32;
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
/// model を消費して更新済み model と損失値を返す（Burn の所有権セマンティクス対応）。
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
) -> (PuyoValueNet<TrainBackend>, f32) {
    let v_next = norm.eval_target(target_model, next_data, infer_device);
    let td_target = norm.normalize(reward + GAMMA * v_next);

    let input = Tensor::<TrainBackend, 1>::from_floats(state_data.as_slice(), device).reshape([
        1,
        NUM_CHANNELS,
        ROWS,
        COLS,
    ]);
    let prediction = model.forward(input);
    let target =
        Tensor::<TrainBackend, 1>::from_floats([td_target].as_slice(), device).reshape([1, 1]);
    let diff = prediction - target;
    let loss = diff.clone().mul(diff).mean();
    let loss_val = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    (optim.step(LEARNING_RATE, model, grads), loss_val)
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
    total_steps: u64,
    target_update_interval: u64,
    reward: f32,
    loss: f32,
    stats: &mut RewardStats,
    game_stats: &mut GameStats,
    session: &GameSession,
    epsilon: f32,
    model: &PuyoValueNet<TrainBackend>,
    target_model: &mut PuyoValueNet<InferBackend>,
    config: &PuyoValueNetConfig,
    infer_device: &<InferBackend as Backend>::Device,
) {
    *step += 1;
    stats.record(reward, loss);

    if *step % LOG_INTERVAL == 0 {
        stats.log_and_reset(*step, total_steps, session.game_count, epsilon, game_stats);
    }

    if *step % target_update_interval == 0 {
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
    total_steps: u64,
    target_update_interval: u64,
) -> PuyoValueNet<InferBackend> {
    let mut session = GameSession::new(100_000);
    let mut step: u64 = 0;
    let mut stats = RewardStats::new();
    let mut game_stats = GameStats::new();

    while step < total_steps {
        let epsilon = compute_epsilon(step, total_steps);
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
            let (m, loss) = td_update(
                model,
                &mut optim,
                &target_model,
                &board_data,
                0.0,
                &placed_data,
                device,
                norm,
                infer_device,
            );
            model = m;
            step_bookkeeping(
                &mut step,
                total_steps,
                target_update_interval,
                0.0,
                loss,
                &mut stats,
                &mut game_stats,
                &session,
                epsilon,
                &model,
                &mut target_model,
                config,
                infer_device,
            );

            // 各連鎖ステップを解決: reward=1
            let mut prev_data = placed_data;
            let mut chain_count: u32 = 0;
            let mut total_score: u32 = 0;

            loop {
                if step >= total_steps {
                    break;
                }
                chain_count += 1;
                match chain::resolve_one_step(&mut session.game.board, chain_count) {
                    Some(chain_step) => {
                        total_score += chain_step.score;
                        let next_data = board_to_tensor_data(&session.game.board);
                        let (m, loss) = td_update(
                            model,
                            &mut optim,
                            &target_model,
                            &prev_data,
                            1.0,
                            &next_data,
                            device,
                            norm,
                            infer_device,
                        );
                        model = m;
                        step_bookkeeping(
                            &mut step,
                            total_steps,
                            target_update_interval,
                            1.0,
                            loss,
                            &mut stats,
                            &mut game_stats,
                            &session,
                            epsilon,
                            &model,
                            &mut target_model,
                            config,
                            infer_device,
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
                game_stats.record(session.game.max_chain, session.move_count);
                session.log_game_over_and_reset(epsilon, step);

                if step < total_steps {
                    let next_data = board_to_tensor_data(&session.game.board);
                    let (m, loss) = td_update(
                        model,
                        &mut optim,
                        &target_model,
                        &prev_data,
                        -1.0,
                        &next_data,
                        device,
                        norm,
                        infer_device,
                    );
                    model = m;
                    step_bookkeeping(
                        &mut step,
                        total_steps,
                        target_update_interval,
                        -1.0,
                        loss,
                        &mut stats,
                        &mut game_stats,
                        &session,
                        epsilon,
                        &model,
                        &mut target_model,
                        config,
                        infer_device,
                    );
                }
            }
        } else {
            // 連鎖なし: finalize してゲームオーバー判定
            session.game.finalize_after_chains(0, 0);

            if session.game.board.is_game_over() {
                game_stats.record(session.game.max_chain, session.move_count);
                session.log_game_over_and_reset(epsilon, step);

                let next_data = board_to_tensor_data(&session.game.board);
                let (m, loss) = td_update(
                    model,
                    &mut optim,
                    &target_model,
                    &board_data,
                    -1.0,
                    &next_data,
                    device,
                    norm,
                    infer_device,
                );
                model = m;
                step_bookkeeping(
                    &mut step,
                    total_steps,
                    target_update_interval,
                    -1.0,
                    loss,
                    &mut stats,
                    &mut game_stats,
                    &session,
                    epsilon,
                    &model,
                    &mut target_model,
                    config,
                    infer_device,
                );
            } else {
                // 生存: reward=1
                let (m, loss) = td_update(
                    model,
                    &mut optim,
                    &target_model,
                    &board_data,
                    1.0,
                    &placed_data,
                    device,
                    norm,
                    infer_device,
                );
                model = m;
                step_bookkeeping(
                    &mut step,
                    total_steps,
                    target_update_interval,
                    1.0,
                    loss,
                    &mut stats,
                    &mut game_stats,
                    &session,
                    epsilon,
                    &model,
                    &mut target_model,
                    config,
                    infer_device,
                );
            }
        }
    }

    model.valid()
}

// ---------------------------------------------------------------------------
// エントリポイント
// ---------------------------------------------------------------------------

fn parse_args() -> (u64, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut total_steps: u64 = 200_000;
    let mut target_update_interval: u64 = 1_000;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                total_steps = args[i].parse().expect("--steps には整数を指定してください");
            }
            "--target-update" => {
                i += 1;
                target_update_interval = args[i]
                    .parse()
                    .expect("--target-update には整数を指定してください");
            }
            other => eprintln!("不明なオプション: {}（無視します）", other),
        }
        i += 1;
    }
    (total_steps, target_update_interval)
}

fn main() {
    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let (total_steps, target_update_interval) = parse_args();
    println!(
        "total_steps={}, target_update_interval={}",
        total_steps, target_update_interval
    );

    let device: <TrainBackend as Backend>::Device = Default::default();
    let infer_device: <InferBackend as Backend>::Device = Default::default();

    let norm = NormParams::load("artifacts/norm_params.txt");
    println!(
        "Normalization: mean={:.4}, std={:.4}",
        norm.mean, norm.std_dev
    );

    let config = PuyoValueNetConfig::new();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model: PuyoValueNet<TrainBackend> = config
        .init(&device)
        .load_file(MODEL_PATH, &recorder, &device)
        .expect("Failed to load model. Run 'train' first.");

    let target_model: PuyoValueNet<InferBackend> = config
        .init(&infer_device)
        .load_file(
            MODEL_PATH,
            &BinFileRecorder::<FullPrecisionSettings>::new(),
            &infer_device,
        )
        .expect("Failed to load target model");

    let optim = AdamConfig::new().init();

    let final_model = run_training_loop(
        model,
        target_model,
        optim,
        &config,
        &device,
        &infer_device,
        &norm,
        total_steps,
        target_update_interval,
    );

    final_model
        .save_file(
            OUTPUT_PATH,
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save model");
    println!("Self-play model saved to {}", OUTPUT_PATH);
}
