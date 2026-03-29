use game_ai::game::Game;
use puyo_player::eval::SimulationEvaluator;
use puyo_player::placement::placement_to_index;
use puyo_player::puyo_game::PuyoGame;
use puyo_player::Evaluator;
use puyo_core::game::{GamePhase, GameState};
use puyo_core::state::PuyoState;
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
        let mut game = GameState::new();
        let mut move_count = 0usize;

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

            let puyo_state = PuyoState {
                board: game.board.clone(),
                current: current_piece,
                next: game.next_piece,
                next_next: game.next_next_piece,
            };

            // Encode board and context BEFORE placement
            let board_data = PuyoGame::encode_board(&puyo_state);
            let context_data = PuyoGame::encode_context(&puyo_state);

            // Find and apply best move
            let result = evaluator.find_best_move(&puyo_state);

            match result {
                Some((placement, score)) => {
                    let action_index = placement_to_index(&placement) as u8;
                    game.apply_placement(&placement);

                    dataset.samples.push(Sample {
                        board_data,
                        context_data,
                        action_index,
                        value_target: score as f32,
                    });
                    move_count += 1;
                }
                None => break,
            }
        }

        total_max_chain = total_max_chain.max(game.max_chain);
        total_chain_sum += game.max_chain as u64;

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
                game.max_chain,
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
