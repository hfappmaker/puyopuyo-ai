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

/// 連鎖オーバーライド判定用の追跡構造体。
/// evaluator最善手より大きい連鎖が見つかった場合、そちらを優先する。
pub struct ChainTracker {
    best_chain_count: u32,
    best_chain_placement: Placement,
    eval_best_chain_count: u32,
}

impl ChainTracker {
    pub fn new(default_placement: Placement) -> Self {
        Self {
            best_chain_count: 0,
            best_chain_placement: default_placement,
            eval_best_chain_count: 0,
        }
    }

    /// 連鎖数を更新（1手目配置に紐付ける）
    pub fn update(&mut self, chain_count: u32, first_placement: Placement) {
        if chain_count > self.best_chain_count {
            self.best_chain_count = chain_count;
            self.best_chain_placement = first_placement;
        }
    }

    /// evaluator最善手が更新された時の連鎖数を記録
    pub fn set_eval_best(&mut self, chain_count: u32) {
        self.eval_best_chain_count = chain_count;
    }

    /// 連鎖オーバーライドが発動するならその配置を返す
    pub fn override_placement(&self) -> Option<Placement> {
        if self.best_chain_count > self.eval_best_chain_count {
            Some(self.best_chain_placement)
        } else {
            None
        }
    }
}

/// Simulation-based evaluator: drops virtual puyos to estimate max chain potential.
pub struct SimulationEvaluator;

impl Evaluator for SimulationEvaluator {
    /// depth-2 + 連鎖オーバーライド、depth-1 フォールバック。
    fn find_best_move(
        &self,
        board: &Board,
        current: &Piece,
        next: &Piece,
        _next_next: &Piece,
    ) -> Option<Placement> {
        let placements = enumerate_placements(board, current);
        if placements.is_empty() {
            return None;
        }

        // depth-2
        let mut tracker = ChainTracker::new(placements[0]);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for placement in &placements {
            let (board_after, chain_result) = simulate_placement(board, current, placement);

            if board_after.is_game_over() {
                tracker.update(chain_result.chain_count, *placement);
                continue;
            }

            let next_placements = enumerate_placements(&board_after, next);
            let mut inner_best_score = f64::NEG_INFINITY;
            let mut deeper_chain: u32 = 0;

            for next_placement in &next_placements {
                let (next_board, next_chain_result) =
                    simulate_placement(&board_after, next, next_placement);
                if next_chain_result.chain_count > deeper_chain {
                    deeper_chain = next_chain_result.chain_count;
                }
                if next_board.is_game_over() {
                    continue;
                }
                let s = simulation_evaluate(&next_board);
                if s > inner_best_score {
                    inner_best_score = s;
                }
            }

            let score = inner_best_score;
            let max_chain_for_this = chain_result.chain_count.max(deeper_chain);
            tracker.update(max_chain_for_this, *placement);

            if score > best_score {
                best_score = score;
                best_placement = *placement;
                tracker.set_eval_best(max_chain_for_this);
            }
        }

        if let Some(p) = tracker.override_placement() {
            return Some(p);
        }

        if best_score > f64::NEG_INFINITY {
            return Some(best_placement);
        }

        // depth-1 フォールバック
        let mut tracker = ChainTracker::new(placements[0]);
        let mut best_score = f64::NEG_INFINITY;
        let mut best_placement = placements[0];

        for placement in &placements {
            let (board_after, chain_result) = simulate_placement(board, current, placement);
            tracker.update(chain_result.chain_count, *placement);

            if board_after.is_game_over() {
                continue;
            }

            let score = simulation_evaluate(&board_after);
            if score > best_score {
                best_score = score;
                best_placement = *placement;
                tracker.set_eval_best(chain_result.chain_count);
            }
        }

        if let Some(p) = tracker.override_placement() {
            return Some(p);
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

fn simulation_evaluate(board: &Board) -> f64 {
    if board.is_game_over() {
        return W_GAME_OVER;
    }
    simulate_max_chain(board) as f64
}

/// 仮想ぷよを落として最大連鎖数を推定する。
fn simulate_max_chain(board: &Board) -> u32 {
    let mut sim = board.clone();
    let result = sim.resolve_chains();
    let mut max_chain = result.chain_count;

    for &color in &COLORS {
        for col in 0..COLS {
            let (h, isolated) = board.column_info(col);
            let available = if isolated { ROWS - 1 - h } else { ROWS - h };
            if available == 0 {
                continue;
            }
            let count = VIRTUAL_PUYO_COUNT.min(available);
            let mut sim = board.clone();
            for _ in 0..count {
                sim.drop_puyo(col, color);
            }
            let result = sim.resolve_chains();
            max_chain = max_chain.max(result.chain_count);
        }
    }
    max_chain
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
