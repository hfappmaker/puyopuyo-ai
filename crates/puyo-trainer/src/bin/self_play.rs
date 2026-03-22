//! AlphaZero-style self-play for Puyo Puyo (Gumbel MCTS).
//!
//! Plays games using Gumbel MCTS + neural network, collects training data,
//! and saves it for the training binary.
//! Games are parallelized across threads for speed.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_ai::hash_util::splitmix64;
use puyo_ai::mcts::mcts_search;
use puyo_ai::nn_eval::MctsConfig;
use puyo_ai::placement::NUM_ACTIONS;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data, CONTEXT_TENSOR_SIZE, NUM_CHANNELS};
use puyo_nn::value_transform::value_inverse_transform;
use puyo_nn::model::{PuyoNet, PuyoNetConfig};
use puyo_trainer::data::{AlphaZeroDataset, AlphaZeroSample};

// MCTS uses NdArray backend (CPU) for inference during self-play.
type InferBackend = NdArray;

const MODEL_PATH: &str = "artifacts/puyo_model";
const DEFAULT_OUTPUT_PATH: &str = "data/alphazero_data.bin";
const MAX_TURNS: u32 = 50;

struct Args {
    num_games: u64,
    num_simulations: usize,
    c_puct: f32,
    seed_offset: u64,
    m: usize,
    c_visit: f32,
    c_scale: f32,
    gamma: f32,
    output_path: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        num_games: 300,
        num_simulations: 64,
        c_puct: 1.5,
        seed_offset: 200_000,
        m: 16,
        c_visit: 50.0,
        c_scale: 1.0,
        gamma: 0.95,
        output_path: DEFAULT_OUTPUT_PATH.to_string(),
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
            "--c-scale" => {
                i += 1;
                result.c_scale = next_val(i, "--c-scale").parse().expect("--c-scale requires float");
            }
            "--gamma" => {
                i += 1;
                result.gamma = next_val(i, "--gamma").parse().expect("--gamma requires float");
            }
            "--output" => {
                i += 1;
                result.output_path = next_val(i, "--output").clone();
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
    reward: f32, // game score for this move
}

/// Result from a single self-play game.
struct GameResult {
    samples: Vec<AlphaZeroSample>,
    max_chain: u32,
}

/// Play one self-play game and return training samples.
fn play_one_game(
    game_idx: u64,
    model: &PuyoNet<InferBackend>,
    device: &<InferBackend as Backend>::Device,
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
        c_scale: args.c_scale,
        gamma: args.gamma,
    };

    while game.phase != GamePhase::GameOver {
        if game.phase != GamePhase::Falling {
            break;
        }
        if move_count >= MAX_TURNS {
            break;
        }

        let current_piece = match &game.current_piece {
            Some(fp) => fp.piece,
            None => break,
        };

        // Encode state
        let board_data = board_to_tensor_data(&game.board).to_vec();
        let context_data = context_to_tensor_data(
            &current_piece,
            &game.next_piece,
            &game.next_next_piece,
        )
        .to_vec();

        // Run Gumbel MCTS (Gumbel noise provides exploration, no Dirichlet needed)
        // Use splitmix64-style hash mixing to decorrelate seeds across consecutive moves
        let gumbel_seed = splitmix64(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(move_count as u64),
        );
        let (mcts_policy, _q_values) = mcts_search(
            &game.board,
            &current_piece,
            &game.next_piece,
            &game.next_next_piece,
            model,
            device,
            &mcts_config,
            gumbel_seed,
        );

        // Select action: sample from improved policy
        // Use splitmix64-style hash mixing to decorrelate from gumbel_seed
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

    // Compute discounted cumulative rewards (backwards) with bootstrap for truncated games
    let num_moves = move_records.len();
    let truncated = move_count >= MAX_TURNS && game.phase != GamePhase::GameOver;
    let mut samples = Vec::new();
    if num_moves > 0 {
        // Bootstrap: if game was truncated (not game over), estimate remaining value with NN
        let bootstrap_value = if truncated {
            estimate_value(model, &game, device)
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
    }
}

fn main() {
    let args = parse_args();

    println!("Backend: NdArray (CPU) — Gumbel MCTS self-play (parallel)");

    println!(
        "games={}, simulations={}, c_puct={}, m={}, c_visit={}, c_scale={}, gamma={}, seed_offset={}",
        args.num_games, args.num_simulations, args.c_puct, args.m,
        args.c_visit, args.c_scale, args.gamma, args.seed_offset,
    );

    let device: <InferBackend as Backend>::Device = Default::default();

    // Load model (fall back to random initialization if no model exists or load fails)
    let config = PuyoNetConfig::new();
    let model = {
        let device_clone = device;
        let load_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            config
                .init::<InferBackend>(&device_clone)
                .load_file(MODEL_PATH, &recorder, &device_clone)
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
                config.init::<InferBackend>(&device)
            }
            Err(_) => {
                println!(
                    "Model file {} is incompatible with current architecture. Initializing random weights",
                    MODEL_PATH
                );
                config.init::<InferBackend>(&device)
            }
        }
    };

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("Using {} threads for parallel self-play", num_threads);

    let start_time = std::time::Instant::now();

    // Shared counters for progress reporting (atomic-only, no Mutex)
    let games_done = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let total_max_chain = AtomicU32::new(0);
    let total_chain_sum = AtomicU64::new(0);

    // Distribute games across threads, each thread returns its results via JoinHandle
    let games_per_thread = args.num_games.div_ceil(num_threads as u64);

    let game_results: Vec<GameResult> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_idx| {
                let start = thread_idx as u64 * games_per_thread;
                let end = (start + games_per_thread).min(args.num_games);

                let thread_model = model.clone();
                let device = &device;
                let args = &args;
                let games_done = &games_done;
                let total_samples = &total_samples;
                let total_max_chain = &total_max_chain;
                let total_chain_sum = &total_chain_sum;
                let start_time = &start_time;

                s.spawn(move || {
                    let mut thread_results = Vec::new();
                    for game_idx in start..end {
                        let result =
                            play_one_game(game_idx, &thread_model, device, args);

                        // Update progress atomically
                        let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                        total_samples.fetch_add(result.samples.len() as u64, Ordering::Relaxed);
                        total_max_chain.fetch_max(result.max_chain, Ordering::Relaxed);
                        total_chain_sum.fetch_add(result.max_chain as u64, Ordering::Relaxed);

                        // Print progress (minor interleaving between threads is acceptable)
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let games_per_sec = done as f64 / elapsed;
                        let samples_so_far = total_samples.load(Ordering::Relaxed);
                        let max_chain = total_max_chain.load(Ordering::Relaxed);
                        let chain_sum = total_chain_sum.load(Ordering::Relaxed);
                        let avg_chain = chain_sum as f64 / done as f64;
                        let eta = (args.num_games - done) as f64 / games_per_sec;
                        println!(
                            "[{:>4}/{}] samples: {:>6} | chain(game/max/avg): {}/{}/{:.1} | {:.2} games/s | ETA: {:.0}s",
                            done, args.num_games, samples_so_far,
                            result.max_chain, max_chain, avg_chain,
                            games_per_sec, eta,
                        );

                        thread_results.push(result);
                    }
                    thread_results
                })
            })
            .collect();

        // Collect in thread_idx order → deterministic sample ordering
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    // Aggregate results
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
/// Used for bootstrapping when the game is truncated at MAX_TURNS.
fn estimate_value(
    model: &PuyoNet<InferBackend>,
    game: &GameState,
    device: &<InferBackend as burn::prelude::Backend>::Device,
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

    let board_tensor =
        burn::tensor::Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([1, NUM_CHANNELS, puyo_core::board::ROWS, puyo_core::board::COLS]);
    let context_tensor =
        burn::tensor::Tensor::<InferBackend, 1>::from_floats(context_data.as_slice(), device)
            .reshape([1, CONTEXT_TENSOR_SIZE]);

    let (_logits, value) = model.forward(board_tensor, context_tensor);
    let value_scalar = value.into_data().to_vec::<f32>().expect("Failed to extract value tensor");
    let v_raw = value_scalar[0];
    value_inverse_transform(v_raw)
}

/// Select an action by sampling from the MCTS policy.
fn select_from_policy(policy: &[f32; NUM_ACTIONS], seed: u64) -> usize {
    // Deterministic sampling using hash
    let x = splitmix64(seed.wrapping_mul(6364136223846793005).wrapping_add(1));

    let r = (x as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cumulative += p as f64;
        if r < cumulative {
            return i;
        }
    }
    // Fallback: return the action with highest probability
    policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
