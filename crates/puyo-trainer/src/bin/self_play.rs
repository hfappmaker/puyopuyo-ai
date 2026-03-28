//! AlphaZero-style self-play for Puyo Puyo (Gumbel MCTS).
//!
//! Plays games using Gumbel MCTS + neural network, collects training data,
//! and saves it for the training binary.
//!
//! CPU mode: each thread clones the model and runs NdArray inference.
//! GPU mode: a dedicated GPU thread batches inference requests from game threads.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_ai::hash_util::splitmix64;
use puyo_ai::mcts::{mcts_search, InferenceProvider};
use puyo_ai::nn_eval::MctsConfig;
use puyo_ai::placement::NUM_ACTIONS;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data};
use puyo_nn::model::{PuyoNet, PuyoNetConfig};
use puyo_trainer::data::{AlphaZeroDataset, AlphaZeroSample};

#[cfg(not(feature = "gpu"))]
use burn::backend::ndarray::NdArray;
#[cfg(not(feature = "gpu"))]
use puyo_ai::nn_eval::DirectInference;

#[cfg(feature = "gpu")]
use burn::backend::CudaJit;
#[cfg(feature = "gpu")]
use puyo_ai::inference_server;

const MODEL_PATH: &str = "artifacts/puyo_model";
const DEFAULT_OUTPUT_PATH: &str = "data/alphazero_data.bin";
const MAX_TURNS: u32 = 50;
#[cfg(feature = "gpu")]
const DEFAULT_GPU_THREADS: usize = 128;
#[cfg(feature = "gpu")]
const DEFAULT_MAX_BATCH_SIZE: usize = 128;

struct Args {
    num_games: u64,
    num_simulations: usize,
    c_puct: f32,
    seed_offset: u64,
    m: usize,
    c_visit: f32,
    gamma: f32,
    output_path: String,
    threads: Option<usize>,
    batch_size: Option<usize>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        num_games: 300,
        num_simulations: 64,
        c_puct: 1.5,
        seed_offset: 200_000,
        m: 16,
        c_visit: 5.0,
        gamma: 0.95,
        output_path: DEFAULT_OUTPUT_PATH.to_string(),
        threads: None,
        batch_size: None,
    };
    let next_val = |i: usize, flag: &str| -> &String {
        args.get(i).unwrap_or_else(|| {
            eprintln!("Error: {flag} requires a value");
            std::process::exit(1);
        })
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                result.num_games = next_val(i, "--games").parse().expect("--games requires integer");
            }
            "--simulations" => {
                i += 1;
                result.num_simulations = next_val(i, "--simulations").parse().expect("--simulations requires integer");
            }
            "--c-puct" => {
                i += 1;
                result.c_puct = next_val(i, "--c-puct").parse().expect("--c-puct requires float");
            }
            "--seed-offset" => {
                i += 1;
                result.seed_offset = next_val(i, "--seed-offset").parse().expect("--seed-offset requires integer");
            }
            "--m" => {
                i += 1;
                result.m = next_val(i, "--m").parse().expect("--m requires integer");
            }
            "--c-visit" => {
                i += 1;
                result.c_visit = next_val(i, "--c-visit").parse().expect("--c-visit requires float");
            }
            "--gamma" => {
                i += 1;
                result.gamma = next_val(i, "--gamma").parse().expect("--gamma requires float");
            }
            "--output" => {
                i += 1;
                result.output_path = next_val(i, "--output").clone();
            }
            "--threads" => {
                i += 1;
                result.threads = Some(next_val(i, "--threads").parse().expect("--threads requires integer"));
            }
            "--batch-size" => {
                i += 1;
                result.batch_size = Some(next_val(i, "--batch-size").parse().expect("--batch-size requires integer"));
            }
            other => eprintln!("Unknown option: {} (ignoring)", other),
        }
        i += 1;
    }
    result
}

/// Record from a single move during self-play.
struct MoveRecord {
    board_data: Vec<f32>,
    context_data: Vec<f32>,
    mcts_policy: Vec<f32>,
    reward: f32,
}

/// Result from a single self-play game.
struct GameResult {
    samples: Vec<AlphaZeroSample>,
    max_chain: u32,
    total_reward: f32,
}

/// Play one self-play game and return training samples.
fn play_one_game(
    game_idx: u64,
    provider: &dyn InferenceProvider,
    args: &Args,
) -> GameResult {
    let seed = args.seed_offset + game_idx;
    let mut game = GameState::new(seed);
    let mut move_records: Vec<MoveRecord> = Vec::with_capacity(MAX_TURNS as usize);
    let mut move_count = 0u32;

    let mcts_config = MctsConfig {
        num_simulations: args.num_simulations,
        c_puct: args.c_puct,
        m: args.m,
        c_visit: args.c_visit,
        gamma: args.gamma,
    };

    while game.phase != GamePhase::GameOver && move_count < MAX_TURNS {
        let current_piece = game.current_piece.as_ref().unwrap().piece;

        let board_data = board_to_tensor_data(&game.board).to_vec();
        let context_data = context_to_tensor_data(
            &current_piece,
            &game.next_piece,
            &game.next_next_piece,
        )
        .to_vec();

        let gumbel_seed = splitmix64(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(move_count as u64),
        );
        let (mcts_policy, _q_values) = mcts_search(
            &game.board,
            &current_piece,
            &game.next_piece,
            &game.next_next_piece,
            provider,
            &mcts_config,
            gumbel_seed,
        );

        let selection_seed = splitmix64(
            seed.wrapping_add(move_count as u64)
                .wrapping_add(0x9e3779b97f4a7c15),
        );
        let action = select_from_policy(&mcts_policy, selection_seed);

        let placement = puyo_ai::placement::index_to_placement(action);
        let chain_result = game.apply_placement(&placement);

        move_records.push(MoveRecord {
            board_data,
            context_data,
            mcts_policy: mcts_policy.to_vec(),
            reward: chain_result.score as f32,
        });

        move_count += 1;
    }

    let num_moves = move_records.len();
    let total_reward: f32 = move_records.iter().map(|r| r.reward).sum();
    let truncated = move_count >= MAX_TURNS && game.phase != GamePhase::GameOver;
    let mut samples = Vec::new();
    if num_moves > 0 {
        let bootstrap_value = if truncated {
            estimate_value(provider, &game)
        } else {
            0.0
        };
        let mut value_targets = vec![0.0f32; num_moves];
        value_targets[num_moves - 1] =
            move_records[num_moves - 1].reward + args.gamma * bootstrap_value;
        for i in (0..num_moves - 1).rev() {
            value_targets[i] =
                move_records[i].reward + args.gamma * value_targets[i + 1];
        }

        for (i, record) in move_records.into_iter().enumerate() {
            samples.push(AlphaZeroSample {
                board_data: record.board_data,
                context_data: record.context_data,
                mcts_policy: record.mcts_policy,
                value_target: value_targets[i],
            });
        }
    }

    GameResult {
        samples,
        max_chain: game.max_chain,
        total_reward,
    }
}

fn load_model<B: Backend>(device: &B::Device) -> PuyoNet<B> {
    let config = PuyoNetConfig::new();
    let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        config
            .init::<B>(device)
            .load_file(MODEL_PATH, &recorder, device)
    }));
    match load_result {
        Ok(Ok(m)) => {
            println!("Loaded existing model from {}", MODEL_PATH);
            m
        }
        Ok(Err(e)) => {
            println!(
                "Failed to load model from {}: {}. Initializing random weights",
                MODEL_PATH, e
            );
            config.init::<B>(device)
        }
        Err(_) => {
            println!(
                "Model file {} is incompatible with current architecture. Initializing random weights",
                MODEL_PATH
            );
            config.init::<B>(device)
        }
    }
}

fn main() {
    let args = parse_args();

    println!(
        "games={}, simulations={}, c_puct={}, m={}, c_visit={}, gamma={}, seed_offset={}",
        args.num_games, args.num_simulations, args.c_puct, args.m,
        args.c_visit, args.gamma, args.seed_offset,
    );

    #[cfg(feature = "gpu")]
    {
        main_gpu(args);
        return;
    }

    #[cfg(not(feature = "gpu"))]
    {
        main_cpu(args);
    }
}

#[cfg(not(feature = "gpu"))]
fn main_cpu(args: Args) {
    println!("Backend: NdArray (CPU) — Gumbel MCTS self-play (parallel)");
    let device: <NdArray as Backend>::Device = Default::default();
    let model = load_model::<NdArray>(&device);

    let num_threads = args.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    });
    println!("Using {} threads for parallel self-play", num_threads);

    // CPU: clone model per thread (NdArray model is not Sync)
    run_games_parallel(&args, num_threads, |thread_idx| {
        let thread_provider = DirectInference::new(model.clone(), device);
        (thread_provider, thread_idx)
    });
}

#[cfg(feature = "gpu")]
fn main_gpu(args: Args) {
    type GpuBackend = CudaJit<f32>;

    println!("Backend: CudaJit (GPU) — Gumbel MCTS self-play (batched)");
    let device: <GpuBackend as Backend>::Device = Default::default();
    let model = load_model::<GpuBackend>(&device);

    let num_threads = args.threads.unwrap_or(DEFAULT_GPU_THREADS);
    let batch_size = args.batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE);
    println!(
        "Using {} game threads, max_batch_size={}",
        num_threads, batch_size,
    );

    let client = inference_server::start_inference_server(model, device, batch_size);

    // GPU: clone client per thread (InferenceClient is Send+Clone)
    run_games_parallel(&args, num_threads, |_| {
        let thread_client = client.clone();
        (thread_client, 0usize)
    });
}

/// Run self-play games in parallel.
/// `make_provider_data` is called once per thread from the main thread,
/// returning a tuple of (provider, extra_data). The provider must be Send.
fn run_games_parallel<F, P>(args: &Args, num_threads: usize, make_provider_data: F)
where
    F: Fn(usize) -> (P, usize),
    P: InferenceProvider + Send,
{
    let start_time = std::time::Instant::now();

    let games_done = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let total_max_chain = AtomicU32::new(0);
    let total_chain_sum = AtomicU64::new(0);
    let total_reward_sum = AtomicU64::new(0);

    let games_per_thread = args.num_games.div_ceil(num_threads as u64);

    // Pre-create providers on main thread
    let providers: Vec<P> = (0..num_threads).map(|i| make_provider_data(i).0).collect();

    let game_results: Vec<GameResult> = std::thread::scope(|s| {
        let handles: Vec<_> = providers
            .into_iter()
            .enumerate()
            .map(|(thread_idx, thread_provider)| {
                let start = thread_idx as u64 * games_per_thread;
                let end = (start + games_per_thread).min(args.num_games);

                let args = &args;
                let games_done = &games_done;
                let total_samples = &total_samples;
                let total_max_chain = &total_max_chain;
                let total_chain_sum = &total_chain_sum;
                let total_reward_sum = &total_reward_sum;
                let start_time = &start_time;

                s.spawn(move || {
                    let mut thread_results = Vec::new();
                    for game_idx in start..end {
                        let result = play_one_game(game_idx, &thread_provider, args);

                        let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                        total_samples.fetch_add(result.samples.len() as u64, Ordering::Relaxed);
                        total_max_chain.fetch_max(result.max_chain, Ordering::Relaxed);
                        total_chain_sum.fetch_add(result.max_chain as u64, Ordering::Relaxed);
                        total_reward_sum.fetch_add(result.total_reward as u64, Ordering::Relaxed);

                        let elapsed = start_time.elapsed().as_secs_f64();
                        let games_per_sec = done as f64 / elapsed;
                        let samples_so_far = total_samples.load(Ordering::Relaxed);
                        let max_chain = total_max_chain.load(Ordering::Relaxed);
                        let chain_sum = total_chain_sum.load(Ordering::Relaxed);
                        let avg_chain = chain_sum as f64 / done as f64;
                        let reward_sum = total_reward_sum.load(Ordering::Relaxed);
                        let avg_reward = reward_sum as f64 / done as f64;
                        let eta = (args.num_games - done) as f64 / games_per_sec;
                        println!(
                            "[{:>4}/{}] samples: {:>6} | chain(game/max/avg): {}/{}/{:.1} | reward(game/avg): {}/{:.0} | {:.2} games/s | ETA: {:.0}s",
                            done, args.num_games, samples_so_far,
                            result.max_chain, max_chain, avg_chain,
                            result.total_reward as u64, avg_reward,
                            games_per_sec, eta,
                        );

                        thread_results.push(result);
                    }
                    thread_results
                })
            })
            .collect();

        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    let mut dataset = AlphaZeroDataset::new();
    let mut final_max_chain = 0u32;
    for result in game_results {
        final_max_chain = final_max_chain.max(result.max_chain);
        dataset.samples.extend(result.samples);
    }

    println!(
        "Self-play complete: {} games, {} samples, max chain: {}",
        args.num_games,
        dataset.samples.len(),
        final_max_chain
    );

    std::fs::create_dir_all("data").expect("Failed to create data directory");
    dataset.save(&args.output_path).expect("Failed to save dataset");
    println!("Saved to {}", args.output_path);
}

/// Estimate the value of the current game state using the neural network.
fn estimate_value(
    provider: &dyn InferenceProvider,
    game: &GameState,
) -> f32 {
    let current_piece = match &game.current_piece {
        Some(fp) => fp.piece,
        None => return 0.0,
    };
    let board_data = board_to_tensor_data(&game.board);
    let context_data = context_to_tensor_data(
        &current_piece,
        &game.next_piece,
        &game.next_next_piece,
    );

    let (_logits, value) = provider.infer(&board_data, &context_data);
    value
}

/// Select an action by sampling from the MCTS policy.
fn select_from_policy(policy: &[f32; NUM_ACTIONS], seed: u64) -> usize {
    let x = splitmix64(seed.wrapping_mul(6364136223846793005).wrapping_add(1));

    let r = (x as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cumulative += p as f64;
        if r < cumulative {
            return i;
        }
    }
    policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
