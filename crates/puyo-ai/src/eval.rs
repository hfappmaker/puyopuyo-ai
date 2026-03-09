use puyo_core::board::{Board, PuyoColor, COLS, ROWS, VISIBLE_ROWS};
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

/// Simulation-based evaluator: drops virtual puyos to estimate max chain potential.
pub struct SimulationEvaluator;

impl Evaluator for SimulationEvaluator {
    fn evaluate(&self, board: &Board) -> f64 {
        if board.is_game_over() {
            return W_GAME_OVER;
        }
        simulate_max_chain(board) as f64
    }
}

const VIRTUAL_PUYO_COUNT: usize = 3;
const COLORS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];

/// Simulate dropping virtual puyos (4 colors × 6 columns = 24 patterns) and return the max chain count.
/// For each column, drop up to 3 same-color puyos (or fewer if space is limited).
fn simulate_max_chain(board: &Board) -> u32 {
    // Check current board for existing chains
    let mut sim = board.clone();
    let result = chain::resolve_chains(&mut sim);
    let mut max_chain = result.chain_count;

    for &color in &COLORS {
        for col in 0..COLS {
            let available = ROWS - board.column_height(col);
            if available == 0 {
                continue;
            }
            let count = VIRTUAL_PUYO_COUNT.min(available);
            let mut sim = board.clone();
            for _ in 0..count {
                sim.drop_puyo(col, color);
            }
            let result = chain::resolve_chains(&mut sim);
            max_chain = max_chain.max(result.chain_count);
        }
    }
    max_chain
}

/// Evaluation weights.
const W_CHAIN_SCORE: f64 = 1.0;
const W_CHAIN_LENGTH: f64 = 50.0;
const W_HEIGHT_PENALTY: f64 = -5.0;
const W_HEIGHT_VARIANCE: f64 = -3.0;
const W_CONNECTIVITY: f64 = 2.0;
const W_POTENTIAL_CHAIN: f64 = 15.0;
const W_CENTER_WEIGHT: f64 = 1.0;
pub const W_GAME_OVER: f64 = -100000.0;

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

const HEIGHT_WARNING_THRESHOLD: usize = 8;
const HEIGHT_DANGER_THRESHOLD: usize = 10;
const HEIGHT_WARNING_MULTIPLIER: f64 = 2.0;
const HEIGHT_DANGER_MULTIPLIER: f64 = 10.0;

/// Penalize tall columns, with extra penalty near the death zone.
fn height_penalty(board: &Board) -> f64 {
    let max_height = (0..COLS).map(|c| board.column_height(c)).max().unwrap_or(0);
    let base = if max_height > HEIGHT_WARNING_THRESHOLD {
        (max_height as f64 - HEIGHT_WARNING_THRESHOLD as f64)
            * W_HEIGHT_PENALTY
            * HEIGHT_WARNING_MULTIPLIER
    } else {
        0.0
    };
    let extra = if max_height > HEIGHT_DANGER_THRESHOLD {
        (max_height as f64 - HEIGHT_DANGER_THRESHOLD as f64)
            * W_HEIGHT_PENALTY
            * HEIGHT_DANGER_MULTIPLIER
    } else {
        0.0
    };
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

/// Count groups of 2-3 same-color connected puyos (potential chains).
pub fn count_potential_chains(board: &Board) -> u32 {
    let mut visited = [[false; ROWS]; COLS];
    let mut count = 0;

    for col in 0..COLS {
        for row in 0..VISIBLE_ROWS {
            if !board.get(col, row).is_color() || visited[col][row] {
                continue;
            }
            let cells = chain::flood_fill(board, col, row, &mut visited);
            if matches!(cells.len(), 2 | 3) {
                count += 1;
            }
        }
    }

    count
}

const CENTER_WEIGHTS: [f64; COLS] = [0.5, 0.8, 1.0, 1.0, 0.8, 0.5];

/// Compute center-column weight. Center columns (2, 3) get higher weight.
fn count_center_weight(board: &Board) -> f64 {
    CENTER_WEIGHTS
        .iter()
        .enumerate()
        .map(|(col, &w)| board.column_height(col) as f64 * w)
        .sum()
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
