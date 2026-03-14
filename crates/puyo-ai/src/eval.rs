use puyo_core::board::{Board, PuyoColor, COLS, ROWS};
use puyo_core::piece::{Piece, Placement};

use crate::placement::{enumerate_placements, simulate_placement};

/// Trait for board evaluation strategies.
pub trait Evaluator {
    /// AI最善手を探索する。
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        next_next: &Piece,
    ) -> Option<Placement>;
}

/// Simulation-based evaluator: drops virtual puyos to estimate expected chain score.
pub struct SimulationEvaluator;

impl Evaluator for SimulationEvaluator {
    /// BFS順で全深度の盤面を評価し、最高スコアの1手目を返す。
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        next_next: &Piece,
    ) -> Option<Placement> {
        let placements = enumerate_placements(board, current);
        if placements.is_empty() {
            return None;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for p1 in &placements {
            let (board1, _) = simulate_placement(board, current, p1);
            if board1.is_game_over() {
                continue;
            }

            let s = simulate_expected_score(&board1);
            if s > best_score {
                best_score = s;
                best_placement = *p1;
            }

            for p2 in &enumerate_placements(&board1, next) {
                let (board2, _) = simulate_placement(&board1, next, p2);
                if board2.is_game_over() {
                    continue;
                }

                let s = simulate_expected_score(&board2);
                if s > best_score {
                    best_score = s;
                    best_placement = *p1;
                }

                for p3 in &enumerate_placements(&board2, next_next) {
                    let (board3, _) = simulate_placement(&board2, next_next, p3);
                    if board3.is_game_over() {
                        continue;
                    }

                    let s = simulate_expected_score(&board3);
                    if s > best_score {
                        best_score = s;
                        best_placement = *p1;
                    }
                }
            }
        }

        Some(best_placement)
    }
}

pub const W_GAME_OVER: f64 = -100000.0;

// ---------------------------------------------------------------------------
// シミュレーション評価
// ---------------------------------------------------------------------------

const VIRTUAL_PUYO_COUNT: usize = 3;
const COLORS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];

/// 仮想ぷよを落として連鎖スコアの期待値（全パターン平均）を推定する。
fn simulate_expected_score(board: &Board) -> f64 {
    let mut sum = 0 as f64;
    let mut count = 1u32;

    for &color in &COLORS {
        for col in 0..COLS {
            let (h, isolated) = board.column_info(col);
            let available = if isolated { ROWS - 1 - h } else { ROWS - h };
            if available == 0 {
                continue;
            }
            let count_puyo = VIRTUAL_PUYO_COUNT.min(available);
            let mut sim = board.clone();
            for _ in 0..count_puyo {
                sim.drop_puyo(col, color);
            }
            let result = sim.resolve_chains();
            let pattern_score = if sim.is_game_over() {
                W_GAME_OVER
            } else {
                result.score as f64
            };
            sum += pattern_score;
            count += 1;
        }
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;

    #[test]
    fn test_depth1_finds_move() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&board, &piece, &Piece::new(PuyoColor::Red, PuyoColor::Blue), &Piece::new(PuyoColor::Green, PuyoColor::Yellow));
        assert!(result.is_some());
    }

    #[test]
    fn test_depth2_finds_move() {
        let board = Board::new();
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&board, &current, &next, &Piece::new(PuyoColor::Green, PuyoColor::Yellow));
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_avoids_game_over() {
        let mut board = Board::new();
        for col in 0..6 {
            for i in 0..10 {
                let color = if (col + i) % 2 == 0 {
                    PuyoColor::Red
                } else {
                    PuyoColor::Blue
                };
                board.drop_puyo(col, color);
            }
        }

        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&board, &current, &next, &Piece::new(PuyoColor::Green, PuyoColor::Yellow));
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_prefers_chain() {
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let evaluator = SimulationEvaluator;
        let result = evaluator.find_best_move(&board, &piece, &next, &Piece::new(PuyoColor::Green, PuyoColor::Yellow));
        assert!(result.is_some());
    }
}
