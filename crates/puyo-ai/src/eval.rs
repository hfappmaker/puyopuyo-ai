use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::chain;

/// Trait for board evaluation strategies.
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> f64;
}

/// The existing hand-tuned heuristic evaluator.
pub struct HeuristicEvaluator;

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&self, board: &Board) -> f64 {
        evaluate(board)
    }
}

/// Evaluation weights.
const W_CHAIN_SCORE: f64 = 1.0;
const W_CHAIN_LENGTH: f64 = 50.0;
const W_HEIGHT_PENALTY: f64 = -5.0;
const W_HEIGHT_VARIANCE: f64 = -3.0;
const W_CONNECTIVITY: f64 = 2.0;
const W_POTENTIAL_CHAIN: f64 = 15.0;
const W_CENTER_WEIGHT: f64 = 1.0;
const W_GAME_OVER: f64 = -100000.0;

/// Evaluate a board state. Higher is better.
pub fn evaluate(board: &Board) -> f64 {
    if board.is_game_over() {
        return W_GAME_OVER;
    }

    let mut score = 0.0;

    // 1. Simulate chains to get chain score and length
    let mut sim_board = board.clone();
    let chain_result = chain::resolve_chains(&mut sim_board);
    score += chain_result.score as f64 * W_CHAIN_SCORE;
    score += chain_result.chain_count as f64 * W_CHAIN_LENGTH;

    // Use the board after chain resolution for remaining evaluation
    let eval_board = &sim_board;

    // 2. Height penalty - penalize tall columns
    let heights: Vec<usize> = (0..COLS).map(|c| eval_board.column_height(c)).collect();
    let max_height = *heights.iter().max().unwrap_or(&0);
    if max_height > 8 {
        score += (max_height as f64 - 8.0) * W_HEIGHT_PENALTY * 2.0;
    }
    // Extra penalty for heights near death zone
    if max_height > 10 {
        score += (max_height as f64 - 10.0) * W_HEIGHT_PENALTY * 10.0;
    }

    // 3. Height variance - prefer even columns
    let avg_height: f64 = heights.iter().sum::<usize>() as f64 / COLS as f64;
    let variance: f64 = heights
        .iter()
        .map(|&h| {
            let diff = h as f64 - avg_height;
            diff * diff
        })
        .sum::<f64>()
        / COLS as f64;
    score += variance * W_HEIGHT_VARIANCE;

    // 4. Connectivity - count same-color adjacent pairs
    let connectivity = count_connectivity(eval_board);
    score += connectivity as f64 * W_CONNECTIVITY;

    // 5. Potential chains - groups of 3 (one more to clear)
    let potential = count_potential_chains(eval_board);
    score += potential as f64 * W_POTENTIAL_CHAIN;

    // 6. Center weight - prefer puyos in center columns
    let center = count_center_weight(eval_board);
    score += center * W_CENTER_WEIGHT;

    score
}

/// Evaluate board after placing a piece (does not modify the input board).
/// Resolves chains and evaluates the resulting board.
pub fn evaluate_placement(board: &Board) -> f64 {
    evaluate(board)
}

/// Count same-color adjacent pairs (horizontal and vertical).
fn count_connectivity(board: &Board) -> u32 {
    let mut count = 0;
    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if !color.is_color() {
                continue;
            }
            // Check right neighbor
            if col + 1 < COLS && board.get(col + 1, row) == color {
                count += 1;
            }
            // Check upper neighbor
            if row + 1 < ROWS && board.get(col, row + 1) == color {
                count += 1;
            }
        }
    }
    count
}

/// Count groups of exactly 3 same-color connected puyos (potential chains).
fn count_potential_chains(board: &Board) -> u32 {
    let mut visited = [[false; ROWS]; COLS];
    let mut count = 0;

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if !color.is_color() || visited[col][row] {
                continue;
            }

            // BFS to find group size
            let mut stack = vec![(col, row)];
            let mut group_size = 0;
            visited[col][row] = true;

            while let Some((c, r)) = stack.pop() {
                group_size += 1;
                let neighbors = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)];
                for (dc, dr) in neighbors {
                    let nc = c as i32 + dc;
                    let nr = r as i32 + dr;
                    if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= ROWS as i32 {
                        continue;
                    }
                    let nc = nc as usize;
                    let nr = nr as usize;
                    if !visited[nc][nr] && board.get(nc, nr) == color {
                        visited[nc][nr] = true;
                        stack.push((nc, nr));
                    }
                }
            }

            if group_size == 3 {
                count += 1;
            }
            // Groups of 2 are also somewhat valuable
            if group_size == 2 {
                count += 1; // half credit counted as same, but lower weight overall
            }
        }
    }

    count
}

/// Compute center-column weight. Center columns (2, 3) get higher weight.
fn count_center_weight(board: &Board) -> f64 {
    let weights = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5];
    let mut total = 0.0;
    for col in 0..COLS {
        let h = board.column_height(col);
        total += h as f64 * weights[col];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::{PuyoColor, VISIBLE_ROWS};

    #[test]
    fn test_empty_board_eval() {
        let board = Board::new();
        let score = evaluate(&board);
        // Empty board should have a baseline score (near 0)
        assert!(score.abs() < 100.0);
    }

    #[test]
    fn test_game_over_eval() {
        let mut board = Board::new();
        // Fill column 2 past visible rows
        for _ in 0..=VISIBLE_ROWS {
            board.drop_puyo(2, PuyoColor::Red);
        }
        let score = evaluate(&board);
        assert!(score < -10000.0);
    }

    #[test]
    fn test_chain_rewards_higher() {
        // Board with 4-in-a-row ready to clear should score higher
        let mut board_chain = Board::new();
        for _ in 0..4 {
            board_chain.drop_puyo(0, PuyoColor::Red);
        }

        let mut board_no_chain = Board::new();
        board_no_chain.drop_puyo(0, PuyoColor::Red);
        board_no_chain.drop_puyo(1, PuyoColor::Blue);
        board_no_chain.drop_puyo(2, PuyoColor::Green);
        board_no_chain.drop_puyo(3, PuyoColor::Yellow);

        let score_chain = evaluate(&board_chain);
        let score_no_chain = evaluate(&board_no_chain);
        assert!(score_chain > score_no_chain);
    }

    #[test]
    fn test_connectivity_bonus() {
        let mut board1 = Board::new();
        // Adjacent same-color puyos
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);

        let mut board2 = Board::new();
        // Scattered different colors
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(1, PuyoColor::Blue);
        board2.drop_puyo(2, PuyoColor::Green);

        // Board1 should have higher connectivity
        let conn1 = count_connectivity(&board1);
        let conn2 = count_connectivity(&board2);
        assert!(conn1 > conn2);
    }
}
