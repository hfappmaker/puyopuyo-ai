//! AlphaZero-style self-play for Puyo Puyo.
//!
//! Plays games using MCTS + neural network, collects training data,
//! and saves it for the training binary.

use burn::backend::ndarray::NdArray;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};

use puyo_ai::mcts::{mcts_search, DirichletConfig};
use puyo_ai::placement::NUM_ACTIONS;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data};
use puyo_nn::model::PuyoNetConfig;
use puyo_trainer::data::{AlphaZeroDataset, AlphaZeroSample};

// MCTS uses NdArray backend (CPU) for inference during self-play.
type InferBackend = NdArray;

const MODEL_PATH: &str = "artifacts/puyo_model";
const DEFAULT_OUTPUT_PATH: &str = "data/alphazero_data.bin";
const MAX_TURNS: u32 = 50;
const GAMMA: f32 = 0.99;

struct Args {
    num_games: u64,
    num_simulations: usize,
    c_puct: f32,
    temperature: f32,
    seed_offset: u64,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    temp_threshold: u32,
    output_path: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut result = Args {
        num_games: 100,
        num_simulations: 200,
        c_puct: 1.5,
        temperature: 1.0,
        seed_offset: 200_000,
        dirichlet_alpha: 0.4,
        dirichlet_epsilon: 0.25,
        temp_threshold: 15,
        output_path: DEFAULT_OUTPUT_PATH.to_string(),
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => {
                i += 1;
                result.num_games = args[i].parse().expect("--games requires integer");
            }
            "--simulations" => {
                i += 1;
                result.num_simulations = args[i].parse().expect("--simulations requires integer");
            }
            "--c-puct" => {
                i += 1;
                result.c_puct = args[i].parse().expect("--c-puct requires float");
            }
            "--temperature" => {
                i += 1;
                result.temperature = args[i].parse().expect("--temperature requires float");
            }
            "--seed-offset" => {
                i += 1;
                result.seed_offset = args[i].parse().expect("--seed-offset requires integer");
            }
            "--dirichlet-alpha" => {
                i += 1;
                result.dirichlet_alpha = args[i].parse().expect("--dirichlet-alpha requires float");
            }
            "--dirichlet-epsilon" => {
                i += 1;
                result.dirichlet_epsilon = args[i].parse().expect("--dirichlet-epsilon requires float");
            }
            "--temp-threshold" => {
                i += 1;
                result.temp_threshold = args[i].parse().expect("--temp-threshold requires integer");
            }
            "--output" => {
                i += 1;
                result.output_path = args[i].clone();
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

fn main() {
    let args = parse_args();

    println!("Backend: NdArray (CPU) — MCTS self-play");

    println!(
        "games={}, simulations={}, c_puct={}, temperature={}, seed_offset={}, dirichlet_alpha={}, dirichlet_epsilon={}",
        args.num_games, args.num_simulations, args.c_puct, args.temperature, args.seed_offset,
        args.dirichlet_alpha, args.dirichlet_epsilon
    );

    let dirichlet = DirichletConfig {
        alpha: args.dirichlet_alpha,
        epsilon: args.dirichlet_epsilon,
    };

    let device: <InferBackend as Backend>::Device = Default::default();

    // Load model (fall back to random initialization if no model exists or load fails)
    let config = PuyoNetConfig::new();
    let model = {
        let device_clone = device.clone();
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

    let mut dataset = AlphaZeroDataset::new();
    let mut total_max_chain = 0u32;
    let mut total_chain_sum = 0u64;
    let start_time = std::time::Instant::now();

    for game_idx in 0..args.num_games {
        let seed = args.seed_offset + game_idx;
        let mut game = GameState::new(seed);
        let mut move_records: Vec<MoveRecord> = Vec::new();
        let mut move_count = 0u32;

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
            let remaining_ratio = (MAX_TURNS - move_count) as f32 / MAX_TURNS as f32;
            let context_data = context_to_tensor_data(
                &current_piece,
                &game.next_piece,
                &game.next_next_piece,
                remaining_ratio,
            )
            .to_vec();

            // Run MCTS with temperature schedule: high exploration early, greedy later
            let temperature = if move_count < args.temp_threshold {
                args.temperature
            } else {
                0.1
            };
            let move_start = std::time::Instant::now();
            let mcts_policy = mcts_search(
                &game.board,
                &current_piece,
                &game.next_piece,
                &game.next_next_piece,
                &model,
                &device,
                args.num_simulations,
                args.c_puct,
                temperature,
                MAX_TURNS,
                move_count,
                Some(&dirichlet),
            );

            println!(
                "  game {} move {}: MCTS {:.2}s",
                game_idx + 1,
                move_count + 1,
                move_start.elapsed().as_secs_f64(),
            );

            // Select action: sample from MCTS policy
            let action = select_from_policy(&mcts_policy, seed + move_count as u64);

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

        // Compute discounted cumulative rewards (backwards)
        let num_moves = move_records.len();
        if num_moves > 0 {
            let mut value_targets = vec![0.0f32; num_moves];
            value_targets[num_moves - 1] = move_records[num_moves - 1].reward;
            for i in (0..num_moves - 1).rev() {
                value_targets[i] =
                    move_records[i].reward + GAMMA * value_targets[i + 1];
            }

            for (i, record) in move_records.into_iter().enumerate() {
                dataset.samples.push(AlphaZeroSample {
                    board_data: record.board_data,
                    context_data: record.context_data,
                    mcts_policy: record.mcts_policy,
                    value_target: value_targets[i],
                });
            }
        }

        total_max_chain = total_max_chain.max(game.max_chain);
        total_chain_sum += game.max_chain as u64;

        {
            let elapsed = start_time.elapsed().as_secs_f64();
            let done = game_idx + 1;
            let games_per_sec = done as f64 / elapsed;
            let eta = (args.num_games - done) as f64 / games_per_sec;
            let avg_chain = total_chain_sum as f64 / done as f64;
            println!(
                "[{:>4}/{}] samples: {:>6} | chain(game/max/avg): {}/{}/{:.1} | {:.2} games/s | ETA: {:.0}s",
                done, args.num_games, dataset.samples.len(),
                game.max_chain, total_max_chain, avg_chain,
                games_per_sec, eta,
            );
        }
    }

    println!(
        "Self-play complete: {} games, {} samples, max chain: {}",
        args.num_games,
        dataset.samples.len(),
        total_max_chain
    );

    std::fs::create_dir_all("data").expect("Failed to create data directory");
    dataset.save(&args.output_path).expect("Failed to save dataset");
    println!("Saved to {}", args.output_path);
}

/// Select an action by sampling from the MCTS policy.
fn select_from_policy(policy: &[f32; NUM_ACTIONS], seed: u64) -> usize {
    // Deterministic sampling using hash
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);

    let r = (x as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0;
    for i in 0..NUM_ACTIONS {
        cumulative += policy[i] as f64;
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
