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
const LAMBDA: f32 = 0.8;
const BUFFER_SIZE: usize = 64;
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
            "[PROGRESS] step={}/{}, games={}, eps={:.3}, rewards(-/0/+)={}/{}/{}, loss={:.4}, avg_chain={:.1}, avg_moves={:.1}",
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

/// TD(λ) バッファの1遷移
struct Transition {
    board_data: [f32; TENSOR_SIZE],
    reward: f32,
    terminal: bool,
}

/// 1エピソード（ゲーム）分の遷移バッファ
struct TrajectoryBuffer {
    transitions: Vec<Transition>,
    /// バッファ最後の遷移の次状態（次の盤面）
    last_next_board: Option<[f32; TENSOR_SIZE]>,
    capacity: usize,
}

impl TrajectoryBuffer {
    fn new(capacity: usize) -> Self {
        TrajectoryBuffer {
            transitions: Vec::with_capacity(capacity),
            last_next_board: None,
            capacity,
        }
    }

    fn push(&mut self, transition: Transition, next_board: [f32; TENSOR_SIZE]) {
        self.transitions.push(transition);
        self.last_next_board = Some(next_board);
    }

    fn is_full(&self) -> bool {
        self.transitions.len() >= self.capacity
    }

    fn clear(&mut self) {
        self.transitions.clear();
        self.last_next_board = None;
    }

    fn len(&self) -> usize {
        self.transitions.len()
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
            Some(&session.game.next_next_piece),
            &evaluator,
        )
        .map(|r| r.best_placement)
    }
}

/// バッファ内の遷移に対してλ-returnを計算する。
/// 返り値は各遷移に対する正規化済みターゲット値。
fn compute_lambda_returns(
    buffer: &TrajectoryBuffer,
    target_model: &PuyoValueNet<InferBackend>,
    norm: &NormParams,
    infer_device: &<InferBackend as Backend>::Device,
    gamma: f32,
    lambda: f32,
) -> Vec<f32> {
    let n = buffer.len();
    let mut targets = vec![0.0f32; n];

    // 末尾の次状態の価値（bootstrap）
    let mut g = if buffer.transitions[n - 1].terminal {
        0.0
    } else {
        match &buffer.last_next_board {
            Some(board) => norm.eval_target(target_model, board, infer_device),
            None => 0.0,
        }
    };

    for t in (0..n).rev() {
        let tr = &buffer.transitions[t];
        if tr.terminal {
            // ゲームオーバー: 伝搬を断ち切る
            g = tr.reward;
        } else {
            let v_next = if t + 1 < n {
                norm.eval_target(target_model, &buffer.transitions[t + 1].board_data, infer_device)
            } else {
                match &buffer.last_next_board {
                    Some(board) => norm.eval_target(target_model, board, infer_device),
                    None => 0.0,
                }
            };
            g = tr.reward + gamma * ((1.0 - lambda) * v_next + lambda * g);
        }
        targets[t] = norm.normalize(g);
    }

    targets
}

/// バッファの全遷移をまとめて1回のforward + backwardで学習する。
/// model を消費して更新済み model と平均損失を返す。
fn batch_update(
    model: PuyoValueNet<TrainBackend>,
    optim: &mut impl Optimizer<PuyoValueNet<TrainBackend>, TrainBackend>,
    buffer: &TrajectoryBuffer,
    targets: &[f32],
    device: &<TrainBackend as Backend>::Device,
) -> (PuyoValueNet<TrainBackend>, f32) {
    let n = buffer.len();

    // 全盤面データを1つのテンソルにまとめる [n, NUM_CHANNELS, ROWS, COLS]
    let mut all_data = Vec::with_capacity(n * TENSOR_SIZE);
    for tr in &buffer.transitions {
        all_data.extend_from_slice(&tr.board_data);
    }
    let input = Tensor::<TrainBackend, 1>::from_floats(all_data.as_slice(), device)
        .reshape([n, NUM_CHANNELS, ROWS, COLS]);
    let prediction = model.forward(input); // [n, 1]

    let target_tensor =
        Tensor::<TrainBackend, 1>::from_floats(targets, device).reshape([n, 1]);
    let diff = prediction - target_tensor;
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

/// バッファのλ-return計算 → バッチ学習 → 統計記録 → バッファクリアを一括実行。
/// model を消費して更新済み model を返す。
fn flush_buffer(
    model: PuyoValueNet<TrainBackend>,
    optim: &mut impl Optimizer<PuyoValueNet<TrainBackend>, TrainBackend>,
    buffer: &mut TrajectoryBuffer,
    target_model: &mut PuyoValueNet<InferBackend>,
    device: &<TrainBackend as Backend>::Device,
    norm: &NormParams,
    infer_device: &<InferBackend as Backend>::Device,
    step: &mut u64,
    total_steps: u64,
    target_update_interval: u64,
    stats: &mut RewardStats,
    game_stats: &mut GameStats,
    session: &GameSession,
    epsilon: f32,
    config: &PuyoValueNetConfig,
    gamma: f32,
    lambda: f32,
) -> PuyoValueNet<TrainBackend> {
    if buffer.len() == 0 {
        return model;
    }

    let targets = compute_lambda_returns(buffer, target_model, norm, infer_device, gamma, lambda);

    // 統計記録（バッファ内の各遷移の報酬を記録）
    for tr in &buffer.transitions {
        stats.record(tr.reward, 0.0); // loss は batch 全体で後から記録
    }

    let (model, loss) = batch_update(model, optim, buffer, &targets, device);

    // loss をバッファサイズ分の遷移に按分して記録（上で 0.0 で記録済みなので上書き）
    // 簡易化: loss_sum に直接加算
    stats.loss_sum += loss as f64;
    // record() で loss_count を既に増やしているので調整不要（0.0 で n 回記録済み）

    *step += buffer.len() as u64;

    if *step % LOG_INTERVAL < buffer.len() as u64 {
        stats.log_and_reset(*step, total_steps, session.game_count, epsilon, game_stats);
    }

    if *step % target_update_interval < buffer.len() as u64 {
        *target_model = sync_target_network(&model, config, infer_device);
    }

    buffer.clear();
    model
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
    buffer_size: usize,
    lambda: f32,
) -> PuyoValueNet<InferBackend> {
    let mut session = GameSession::new(100_000);
    let mut step: u64 = 0;
    let mut stats = RewardStats::new();
    let mut game_stats = GameStats::new();
    let mut buffer = TrajectoryBuffer::new(buffer_size);

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

        // 配置 + 連鎖解決を一括実行
        let prev_max = session.game.max_chain;
        let chain_result = session.game.apply_placement(&placement);
        session.move_count += 1;
        session.log_if_new_max_chain(prev_max, step);

        if session.game.board.is_game_over() {
            // ゲームオーバー: reward=-1, terminal=true
            game_stats.record(session.game.max_chain, session.move_count);
            session.log_game_over_and_reset(epsilon, step);

            let next_board = board_to_tensor_data(&session.game.board);
            buffer.push(
                Transition {
                    board_data,
                    reward: -1.0,
                    terminal: true,
                },
                next_board,
            );

            // ゲームオーバー時は即座にバッファを消化
            model = flush_buffer(
                model,
                &mut optim,
                &mut buffer,
                &mut target_model,
                device,
                norm,
                infer_device,
                &mut step,
                total_steps,
                target_update_interval,
                &mut stats,
                &mut game_stats,
                &session,
                epsilon,
                config,
                GAMMA,
                lambda,
            );
        } else {
            // 生存: reward=スコア
            let reward = chain_result.score as f32;
            let next_board = board_to_tensor_data(&session.game.board);
            buffer.push(
                Transition {
                    board_data,
                    reward,
                    terminal: false,
                },
                next_board,
            );

            if buffer.is_full() {
                model = flush_buffer(
                    model,
                    &mut optim,
                    &mut buffer,
                    &mut target_model,
                    device,
                    norm,
                    infer_device,
                    &mut step,
                    total_steps,
                    target_update_interval,
                    &mut stats,
                    &mut game_stats,
                    &session,
                    epsilon,
                    config,
                    GAMMA,
                    lambda,
                );
            }
        }
    }

    // 残りのバッファを消化
    if buffer.len() > 0 {
        let epsilon = compute_epsilon(step, total_steps);
        model = flush_buffer(
            model,
            &mut optim,
            &mut buffer,
            &mut target_model,
            device,
            norm,
            infer_device,
            &mut step,
            total_steps,
            target_update_interval,
            &mut stats,
            &mut game_stats,
            &session,
            epsilon,
            config,
            GAMMA,
            lambda,
        );
    }

    model.valid()
}

// ---------------------------------------------------------------------------
// エントリポイント
// ---------------------------------------------------------------------------

struct Args {
    total_steps: u64,
    target_update_interval: u64,
    buffer_size: usize,
    lambda: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        total_steps: 200_000,
        target_update_interval: 1_000,
        buffer_size: BUFFER_SIZE,
        lambda: LAMBDA,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                result.total_steps = args[i].parse().expect("--steps には整数を指定してください");
            }
            "--target-update" => {
                i += 1;
                result.target_update_interval = args[i]
                    .parse()
                    .expect("--target-update には整数を指定してください");
            }
            "--buffer-size" => {
                i += 1;
                result.buffer_size = args[i]
                    .parse()
                    .expect("--buffer-size には整数を指定してください");
            }
            "--lambda" => {
                i += 1;
                result.lambda = args[i]
                    .parse()
                    .expect("--lambda には浮動小数点数を指定してください");
            }
            other => eprintln!("不明なオプション: {}（無視します）", other),
        }
        i += 1;
    }
    result
}

fn main() {
    #[cfg(feature = "gpu")]
    println!("Backend: CUDA (GPU)");
    #[cfg(not(feature = "gpu"))]
    println!("Backend: NdArray (CPU)");

    let args = parse_args();
    println!(
        "total_steps={}, target_update_interval={}, buffer_size={}, lambda={}",
        args.total_steps, args.target_update_interval, args.buffer_size, args.lambda
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
        args.total_steps,
        args.target_update_interval,
        args.buffer_size,
        args.lambda,
    );

    final_model
        .save_file(
            OUTPUT_PATH,
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Failed to save model");
    println!("Self-play model saved to {}", OUTPUT_PATH);
}
