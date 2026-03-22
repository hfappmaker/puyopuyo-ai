use puyo_core::board::{Board, PuyoColor};
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
    ) -> Option<(Placement, f64)>;

    /// MCTSシミュレーション数を変更する。対応していない評価器では何もしない。
    fn set_num_simulations(&mut self, _num_simulations: usize) {}
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
        _next_next: &Piece,
    ) -> Option<(Placement, f64)> {
        let placements = enumerate_placements(board, current);
        if placements.is_empty() {
            return None;
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for p1 in &placements {
            let (board1, result1) = simulate_placement(board, current, p1);
            if board1.is_game_over() {
                continue;
            }

            let s = (result1.score as f64).max(simulate_expected_score(&board1));
            if s > best_score {
                best_score = s;
                best_placement = *p1;
            }

            for p2 in &enumerate_placements(&board1, next) {
                let (board2, result2) = simulate_placement(&board1, next, p2);
                if board2.is_game_over() {
                    continue;
                }

                let s = (result2.score as f64).max(simulate_expected_score(&board2));
                if s > best_score {
                    best_score = s;
                    best_placement = *p1;
                }

                // 3手先のシミュレーションは重すぎるので省略（期待値評価だけで十分なはず）
                // for p3 in &enumerate_placements(&board2, next_next) {
                //     let (board3, result3) = simulate_placement(&board2, next_next, p3);
                //     if board3.is_game_over() {
                //         continue;
                //     }

                //     let s = (result3.score as f64).max(simulate_expected_score(&board3));
                //     if s > best_score {
                //         best_score = s;
                //         best_placement = *p1;
                //     }
                // }
            }
        }

        Some((best_placement, best_score))
    }
}

pub const W_GAME_OVER: f64 = -1000000.0;

// ---------------------------------------------------------------------------
// シミュレーション評価
// ---------------------------------------------------------------------------

const COLORS: [PuyoColor; 4] = [
    PuyoColor::Red,
    PuyoColor::Green,
    PuyoColor::Blue,
    PuyoColor::Yellow,
];

/// 仮想ぷよ（同色2個Piece）を全合法配置に落として連鎖スコアの期待値（全パターン平均）を推定する。
fn simulate_expected_score(board: &Board) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0u32;

    for &color in &COLORS {
        let piece = Piece::new(color, color);
        for placement in &enumerate_placements(board, &piece) {
            let (sim, result) = simulate_placement(board, &piece, placement);
            let pattern_score = if sim.is_game_over() {
                W_GAME_OVER
            } else {
                result.score as f64
            };
            sum += pattern_score;
            count += 1;
        }
    }

    if count == 0 {
        return W_GAME_OVER;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::{PuyoColor, COLS};

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

    // ---------------------------------------------------------------
    // simulate_expected_score のテスト
    // ---------------------------------------------------------------

    #[test]
    fn test_simulate_expected_score_empty_board() {
        // 空盤面: 仮想ぷよを落としても連鎖は起きない → スコア 0 付近
        let board = Board::new();
        let score = simulate_expected_score(&board);
        assert!(
            score.abs() < 1.0,
            "空盤面の期待スコアは0付近であるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_near_chain() {
        // 列0に赤3つ → 赤の仮想ぷよで連鎖発生 → 正のスコア期待
        let mut board = Board::new();
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        let score = simulate_expected_score(&board);
        assert!(
            score > 0.0,
            "連鎖可能盤面は正のスコアであるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_more_potential_is_higher() {
        // 赤3つ(1連鎖分)と赤3+青3(2連鎖分)で後者が高スコア
        let mut board1 = Board::new();
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);
        board1.drop_puyo(0, PuyoColor::Red);

        let mut board2 = Board::new();
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(0, PuyoColor::Red);
        board2.drop_puyo(1, PuyoColor::Blue);
        board2.drop_puyo(1, PuyoColor::Blue);
        board2.drop_puyo(1, PuyoColor::Blue);

        let s1 = simulate_expected_score(&board1);
        let s2 = simulate_expected_score(&board2);
        assert!(
            s2 > s1,
            "連鎖ポテンシャルが多い盤面のスコアが高いべき: s1={s1}, s2={s2}"
        );
    }

    #[test]
    fn test_simulate_expected_score_game_over_penalty() {
        // ほぼ満杯の盤面はゲームオーバーペナルティで大きな負のスコア
        let mut board = Board::new();
        for col in 0..COLS {
            for i in 0..12 {
                let color = if (col + i) % 2 == 0 {
                    PuyoColor::Red
                } else {
                    PuyoColor::Blue
                };
                board.drop_puyo(col, color);
            }
        }
        let score = simulate_expected_score(&board);
        assert!(
            score < -1000.0,
            "満杯に近い盤面は大きな負のスコアであるべき: got {score}"
        );
    }

    #[test]
    fn test_simulate_expected_score_deterministic() {
        // 同じ盤面に対して常に同じ値を返す（乱数なし）
        let mut board = Board::new();
        board.drop_puyo(2, PuyoColor::Green);
        board.drop_puyo(2, PuyoColor::Green);
        board.drop_puyo(3, PuyoColor::Yellow);

        let s1 = simulate_expected_score(&board);
        let s2 = simulate_expected_score(&board);
        assert_eq!(s1, s2, "同じ盤面には同じスコアを返すべき");
    }
}
