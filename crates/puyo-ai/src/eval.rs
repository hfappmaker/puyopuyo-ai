use puyo_core::board::{Board, COLS, ROWS, VISIBLE_ROWS};
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

    let mut sim_board = board.clone();
    let chain_result = chain::resolve_chains(&mut sim_board);

    chain_result.score as f64 * W_CHAIN_SCORE
        + chain_result.chain_count as f64 * W_CHAIN_LENGTH
        + height_penalty(&sim_board)
        + height_variance(&sim_board) * W_HEIGHT_VARIANCE
        + count_connectivity(&sim_board) as f64 * W_CONNECTIVITY
        + count_potential_chains(&sim_board) as f64 * W_POTENTIAL_CHAIN
        + count_center_weight(&sim_board) * W_CENTER_WEIGHT
}

/// Penalize tall columns, with extra penalty near the death zone.
fn height_penalty(board: &Board) -> f64 {
    let max_height = (0..COLS).map(|c| board.column_height(c)).max().unwrap_or(0);
    let base = if max_height > 8 { (max_height as f64 - 8.0) * W_HEIGHT_PENALTY * 2.0 } else { 0.0 };
    let extra = if max_height > 10 { (max_height as f64 - 10.0) * W_HEIGHT_PENALTY * 10.0 } else { 0.0 };
    base + extra
}

/// Variance of column heights. Lower variance = more even columns.
fn height_variance(board: &Board) -> f64 {
    let heights: Vec<f64> = (0..COLS).map(|c| board.column_height(c) as f64).collect();
    let avg = heights.iter().sum::<f64>() / COLS as f64;
    heights.iter().map(|&h| (h - avg) * (h - avg)).sum::<f64>() / COLS as f64
}

/// Evaluate board after placing a piece (does not modify the input board).
/// Resolves chains and evaluates the resulting board.
pub fn evaluate_placement(board: &Board) -> f64 {
    evaluate(board)
}

/// Count same-color adjacent pairs (horizontal and vertical).
pub fn count_connectivity(board: &Board) -> u32 {
    (0..COLS)
        .flat_map(|col| (0..VISIBLE_ROWS).map(move |row| (col, row)))
        .filter(|&(col, row)| board.get(col, row).is_color())
        .map(|(col, row)| {
            let color = board.get(col, row);
            let right = (col + 1 < COLS && board.get(col + 1, row) == color) as u32;
            let upper = (row + 1 < VISIBLE_ROWS && board.get(col, row + 1) == color) as u32;
            right + upper
        })
        .sum()
}

/// Count groups of exactly 3 same-color connected puyos (potential chains).
pub fn count_potential_chains(board: &Board) -> u32 {
    let mut visited = [[false; ROWS]; COLS];
    let mut count = 0;

    for col in 0..COLS {
        for row in 0..VISIBLE_ROWS {
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
                    if nc < 0 || nc >= COLS as i32 || nr < 0 || nr >= VISIBLE_ROWS as i32 {
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
