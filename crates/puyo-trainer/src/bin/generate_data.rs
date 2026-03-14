use puyo_ai::eval::{Evaluator, SimulationEvaluator};
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::board_to_tensor_data;
use puyo_trainer::data::{Dataset, Sample};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

const NUM_GAMES: u64 = 10_000;
const OUTPUT_PATH: &str = "data/training_data.bin";
const MAX_MOVES_PER_GAME: usize = 50;

struct GameResult {
    samples: Vec<Sample>,
    max_chain: u32,
    game_max_chain: u32,
}

fn run_single_game(seed: u64, evaluator: &SimulationEvaluator) -> GameResult {
    let mut game = GameState::new(seed);
    let mut game_moves: Vec<(Vec<f32>, u32)> = Vec::new();
    let mut game_max_chain = 0u32;

    while game.phase != GamePhase::GameOver {
        if game.phase != GamePhase::Falling {
            break;
        }

        if game_moves.len() >= MAX_MOVES_PER_GAME {
            break;
        }

        let current_piece = match &game.current_piece {
            Some(fp) => fp.piece,
            None => break,
        };

        let board_data = board_to_tensor_data(&game.board).to_vec();

        let result = evaluator.find_best_move(
            &game.board,
            &current_piece,
            &game.next_piece,
            &game.next_next_piece,
        );

        match result {
            Some((placement, _score)) => {
                let chain_result = game.apply_placement(&placement);
                game_max_chain = game_max_chain.max(chain_result.chain_count);
                game_moves.push((board_data, chain_result.score));
            }
            None => break,
        }
    }

    // Compute discounted future scores (backwards)
    let gamma = 0.95f32;
    let num_moves = game_moves.len();
    let mut samples = Vec::with_capacity(num_moves);

    if num_moves > 0 {
        let mut future_values = vec![0.0f32; num_moves];
        future_values[num_moves - 1] = game_moves[num_moves - 1].1 as f32;
        for i in (0..num_moves - 1).rev() {
            let immediate = game_moves[i].1 as f32;
            future_values[i] = immediate + gamma * future_values[i + 1];
        }

        for (i, (board_data, _)) in game_moves.iter().enumerate() {
            samples.push(Sample {
                board_data: board_data.clone(),
                target: future_values[i],
            });
        }
    }

    GameResult {
        samples,
        max_chain: game.max_chain,
        game_max_chain,
    }
}

fn main() {
    std::fs::create_dir_all("data").expect("Failed to create data directory");

    let evaluator = SimulationEvaluator;
    let start_time = std::time::Instant::now();
    let games_done = AtomicU64::new(0);
    let total_chain_sum = AtomicU64::new(0);

    let results: Vec<GameResult> = (0..NUM_GAMES)
        .into_par_iter()
        .map(|seed| {
            let result = run_single_game(seed, &evaluator);

            let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
            total_chain_sum.fetch_add(result.game_max_chain as u64, Ordering::Relaxed);

            if done % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let games_per_sec = done as f64 / elapsed;
                let eta_secs = (NUM_GAMES - done) as f64 / games_per_sec;
                let avg_chain =
                    total_chain_sum.load(Ordering::Relaxed) as f64 / done as f64;
                println!(
                    "[{:>5}/{}] | {:.1} games/s | avg_chain: {:.1} | ETA: {:.0}s",
                    done, NUM_GAMES, games_per_sec, avg_chain, eta_secs,
                );
            }

            result
        })
        .collect();

    // Merge results sequentially (collect preserves order)
    let mut dataset = Dataset::new();
    let mut total_max_chain = 0u32;
    for result in results {
        total_max_chain = total_max_chain.max(result.max_chain);
        dataset.samples.extend(result.samples);
    }

    println!(
        "Data generation complete: {} games, {} samples, max chain: {}",
        NUM_GAMES,
        dataset.samples.len(),
        total_max_chain
    );

    dataset.save(OUTPUT_PATH).expect("Failed to save dataset");
    println!("Saved to {}", OUTPUT_PATH);
}
