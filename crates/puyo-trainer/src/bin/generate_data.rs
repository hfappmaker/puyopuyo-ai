use puyo_ai::eval::{Evaluator, SimulationEvaluator};
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::board_to_tensor_data;
use puyo_trainer::data::{Dataset, Sample};

const NUM_GAMES: u64 = 10_000;
const OUTPUT_PATH: &str = "data/training_data.bin";
const MAX_MOVES_PER_GAME: usize = 50;

fn main() {
    std::fs::create_dir_all("data").expect("Failed to create data directory");

    let evaluator = SimulationEvaluator;
    let mut dataset = Dataset::new();
    let mut total_max_chain = 0u32;
    let mut total_chain_sum = 0u64;
    let start_time = std::time::Instant::now();

    for seed in 0..NUM_GAMES {
        let mut game = GameState::new(seed);
        let mut game_moves: Vec<(Vec<f32>, u32)> = Vec::new();
        let mut game_max_chain = 0u32;

        while game.phase != GamePhase::GameOver {
            if game.phase != GamePhase::Falling {
                break;
            }

            if game_moves.len() >= MAX_MOVES_PER_GAME {
                println!("Game {} reached max moves limit ({})", seed, MAX_MOVES_PER_GAME);
                break;
            }

            let current_piece = match &game.current_piece {
                Some(fp) => fp.piece,
                None => break,
            };

            // Record board state BEFORE placement
            let board_data = board_to_tensor_data(&game.board).to_vec();

            // Find and apply best move
            let result = evaluator.find_best_move(
                &game.board,
                &current_piece,
                &game.next_piece,
                Some(&game.next_next_piece),
            );

            match result {
                Some(placement) => {
                    let chain_result = game.apply_placement(&placement);
                    game_max_chain = game_max_chain.max(chain_result.chain_count);
                    game_moves.push((board_data, chain_result.score));
                }
                None => break,
            }
        }

        // Now create training samples.
        // For each move, the target is the score achieved on that move.
        // We also add a discounted future score to encourage setup.
        let gamma = 0.95f32;
        let num_moves = game_moves.len();
        if num_moves == 0 {
            continue;
        }

        // Compute discounted future scores (backwards)
        let mut future_values = vec![0.0f32; num_moves];
        future_values[num_moves - 1] = game_moves[num_moves - 1].1 as f32;
        for i in (0..num_moves - 1).rev() {
            let immediate = game_moves[i].1 as f32;
            future_values[i] = immediate + gamma * future_values[i + 1];
        }

        for (i, (board_data, _)) in game_moves.iter().enumerate() {
            dataset.samples.push(Sample {
                board_data: board_data.clone(),
                target: future_values[i],
            });
        }

        total_max_chain = total_max_chain.max(game.max_chain);
        total_chain_sum += game_max_chain as u64;

        if (seed + 1) % 100 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let games_done = seed + 1;
            let games_per_sec = games_done as f64 / elapsed;
            let eta_secs = (NUM_GAMES - games_done) as f64 / games_per_sec;
            let avg_chain = total_chain_sum as f64 / games_done as f64;
            println!(
                "[{:>5}/{}] samples: {:>7} | moves: {:>3} | chain(game/max/avg): {}/{}/{:.1} | score: {} | {:.1} games/s | ETA: {:.0}s",
                games_done,
                NUM_GAMES,
                dataset.samples.len(),
                num_moves,
                game_max_chain,
                total_max_chain,
                avg_chain,
                game.score,
                games_per_sec,
                eta_secs,
            );
        }
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
