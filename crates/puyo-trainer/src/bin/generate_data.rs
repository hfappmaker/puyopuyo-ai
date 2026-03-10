use puyo_ai::eval::SimulationEvaluator;
use puyo_ai::search;
use puyo_core::game::{GamePhase, GameState};
use puyo_nn::encoding::board_to_tensor_data;
use puyo_trainer::data::{Dataset, Sample};

const NUM_GAMES: u64 = 10_000;
const OUTPUT_PATH: &str = "data/training_data.bin";

fn main() {
    std::fs::create_dir_all("data").expect("Failed to create data directory");

    let evaluator = SimulationEvaluator;
    let mut dataset = Dataset::new();
    let mut total_max_chain = 0u32;

    for seed in 0..NUM_GAMES {
        let mut game = GameState::new(seed);
        let mut game_moves: Vec<(Vec<f32>, u32)> = Vec::new();

        while game.phase != GamePhase::GameOver {
            if game.phase != GamePhase::Falling {
                break;
            }

            let current_piece = match &game.current_piece {
                Some(fp) => fp.piece,
                None => break,
            };

            // Record board state BEFORE placement
            let board_data = board_to_tensor_data(&game.board).to_vec();

            // Find and apply best move
            let result = search::find_best_move(
                &game.board,
                &current_piece,
                &game.next_piece,
                Some(&game.next_next_piece),
                &evaluator,
            );

            match result {
                Some(r) => {
                    let chain_result = game.apply_placement(&r.best_placement);
                    // Record (board_state, chain_count from this placement)
                    game_moves.push((board_data, chain_result.chain_count));
                }
                None => break,
            }
        }

        // Now create training samples.
        // For each move, the target is the chain count achieved on that move.
        // We also add a discounted future chain value to encourage setup.
        let gamma = 0.95f32;
        let num_moves = game_moves.len();
        if num_moves == 0 {
            continue;
        }

        // Compute discounted future chain counts (backwards)
        let mut future_values = vec![0.0f32; num_moves];
        future_values[num_moves - 1] = 2.0f32.powi(game_moves[num_moves - 1].1 as i32) - 1.0;
        for i in (0..num_moves - 1).rev() {
            let immediate = 2.0f32.powi(game_moves[i].1 as i32) - 1.0;
            future_values[i] = immediate + gamma * future_values[i + 1];
        }

        for (i, (board_data, _)) in game_moves.iter().enumerate() {
            dataset.samples.push(Sample {
                board_data: board_data.clone(),
                target: future_values[i],
            });
        }

        total_max_chain = total_max_chain.max(game.max_chain);

        if (seed + 1) % 1000 == 0 {
            println!(
                "Games: {}/{}, Samples: {}, Max chain so far: {}",
                seed + 1,
                NUM_GAMES,
                dataset.samples.len(),
                total_max_chain
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
