use puyo_ai::eval::{Evaluator, SimulationEvaluator};
use puyo_ai::placement::placement_to_index;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::{board_to_tensor_data, context_to_tensor_data};
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
        let mut move_count = 0usize;
        let mut game_max_chain = 0u32;

        while game.phase != GamePhase::GameOver {
            if game.phase != GamePhase::Falling {
                break;
            }

            if move_count >= MAX_MOVES_PER_GAME {
                println!("Game {} reached max moves limit ({})", seed, MAX_MOVES_PER_GAME);
                break;
            }

            let current_piece = match &game.current_piece {
                Some(fp) => fp.piece,
                None => break,
            };

            // Encode board and context BEFORE placement
            let board_data = board_to_tensor_data(&game.board).to_vec();
            let remaining_ratio = (MAX_MOVES_PER_GAME - move_count) as f32 / MAX_MOVES_PER_GAME as f32;
            let context_data = context_to_tensor_data(
                &current_piece,
                &game.next_piece,
                &game.next_next_piece,
                remaining_ratio,
            )
            .to_vec();

            // Find and apply best move
            let result = evaluator.find_best_move(
                &game.board,
                &current_piece,
                &game.next_piece,
                &game.next_next_piece,
                move_count as u32,
            );

            match result {
                Some((placement, _score)) => {
                    let action_index = placement_to_index(&placement) as u8;
                    let chain_result = game.apply_placement(&placement);
                    game_max_chain = game_max_chain.max(chain_result.chain_count);

                    dataset.samples.push(Sample {
                        board_data,
                        context_data,
                        action_index,
                    });
                    move_count += 1;
                }
                None => break,
            }
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
                move_count,
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
