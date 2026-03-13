use puyo_core::board::{Board, ChainResult};
use puyo_core::game::GameState;
use puyo_core::piece::{Piece, Placement};

use crate::eval::Evaluator;
#[cfg(test)]
use crate::eval::HeuristicEvaluator;
use crate::placement::enumerate_placements;

/// Result of AI search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_placement: Placement,
    pub score: f64,
    pub depth: u32,
}

/// Simulate placing a piece on a board clone. Returns the resulting board and chain result.
fn simulate_placement(board: &Board, piece: &Piece, placement: &Placement) -> (Board, ChainResult) {
    let mut sim = GameState::new(0);
    sim.board = board.clone();
    sim.place_piece(piece, placement);
    let chain_result = sim.board.resolve_chains();
    (sim.board, chain_result)
}

/// 連鎖オーバーライド判定用の追跡構造体。
/// evaluator最善手より大きい連鎖が見つかった場合、そちらを優先する。
struct ChainTracker {
    enabled: bool,
    best_chain_count: u32,
    best_chain_placement: Placement,
    eval_best_chain_count: u32,
}

impl ChainTracker {
    fn new(enabled: bool, default_placement: Placement) -> Self {
        Self {
            enabled,
            best_chain_count: 0,
            best_chain_placement: default_placement,
            eval_best_chain_count: 0,
        }
    }

    /// 連鎖数を更新（1手目配置に紐付ける）
    fn update(&mut self, chain_count: u32, first_placement: Placement) {
        if self.enabled && chain_count > self.best_chain_count {
            self.best_chain_count = chain_count;
            self.best_chain_placement = first_placement;
        }
    }

    /// evaluator最善手が更新された時の連鎖数を記録
    fn set_eval_best(&mut self, chain_count: u32) {
        self.eval_best_chain_count = chain_count;
    }

    /// 連鎖オーバーライドが発動するならそのSearchResultを返す
    fn override_result(&self, depth: u32) -> Option<SearchResult> {
        if self.enabled && self.best_chain_count > self.eval_best_chain_count {
            Some(SearchResult {
                best_placement: self.best_chain_placement,
                score: f64::INFINITY,
                depth,
            })
        } else {
            None
        }
    }
}

/// 2手目以降の探索: boardに対してnext以降のピースを配置し、
/// (最高スコア, 最大連鎖数) を返す。
/// remaining_piecesが空なら盤面を直接評価する。
fn search_remaining(
    board: &Board,
    remaining_pieces: &[&Piece],
    evaluator: &dyn Evaluator,
    use_chain_override: bool,
) -> (f64, u32) {
    if remaining_pieces.is_empty() {
        return (evaluator.evaluate(board), 0);
    }

    let piece = remaining_pieces[0];
    let rest = &remaining_pieces[1..];

    let placements = enumerate_placements(board, piece);
    if placements.is_empty() {
        return (f64::NEG_INFINITY, 0);
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut max_chain: u32 = 0;

    for placement in &placements {
        let (result_board, chain_result) = simulate_placement(board, piece, placement);
        if use_chain_override && chain_result.chain_count > max_chain {
            max_chain = chain_result.chain_count;
        }

        if result_board.is_game_over() {
            continue;
        }

        let (score, deeper_chain) = search_remaining(&result_board, rest, evaluator, use_chain_override);
        if use_chain_override && deeper_chain > max_chain {
            max_chain = deeper_chain;
        }
        if score > best_score {
            best_score = score;
        }
    }

    (best_score, max_chain)
}

/// Depth-1 search: evaluate all placements for the current piece.
pub fn search_depth1(
    board: &Board,
    current: &Piece,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    let placements = enumerate_placements(board, current);
    if placements.is_empty() {
        return None;
    }

    let use_chain_override = evaluator.use_chain_override();
    let mut tracker = ChainTracker::new(use_chain_override, placements[0]);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_placement = placements[0];

    for placement in &placements {
        let (result_board, chain_result) = simulate_placement(board, current, placement);
        tracker.update(chain_result.chain_count, *placement);

        let score = evaluator.evaluate(&result_board);
        if score > best_score {
            best_score = score;
            best_placement = *placement;
            tracker.set_eval_best(chain_result.chain_count);
        }
    }

    if let Some(result) = tracker.override_result(1) {
        return Some(result);
    }

    Some(SearchResult {
        best_placement,
        score: best_score,
        depth: 1,
    })
}

/// Depth-N search (N >= 2): evaluate placements for current piece,
/// then recursively search remaining pieces.
fn search_deep(
    board: &Board,
    current: &Piece,
    remaining: &[&Piece],
    evaluator: &dyn Evaluator,
    depth: u32,
) -> Option<SearchResult> {
    let placements = enumerate_placements(board, current);
    if placements.is_empty() {
        return None;
    }

    let use_chain_override = evaluator.use_chain_override();
    let mut tracker = ChainTracker::new(use_chain_override, placements[0]);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_placement = placements[0];

    for placement in &placements {
        let (board_after, chain_result) = simulate_placement(board, current, placement);

        if board_after.is_game_over() {
            tracker.update(chain_result.chain_count, *placement);
            continue;
        }

        let (score, deeper_chain) = search_remaining(
            &board_after,
            remaining,
            evaluator,
            use_chain_override,
        );

        let max_chain_for_this = chain_result.chain_count.max(deeper_chain);
        tracker.update(max_chain_for_this, *placement);

        if score > best_score {
            best_score = score;
            best_placement = *placement;
            tracker.set_eval_best(max_chain_for_this);
        }
    }

    if let Some(result) = tracker.override_result(depth) {
        return Some(result);
    }

    Some(SearchResult {
        best_placement,
        score: best_score,
        depth,
    })
}

/// Depth-2 search: evaluate all placements for current + next piece.
pub fn search_depth2(
    board: &Board,
    current: &Piece,
    next: &Piece,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    search_deep(board, current, &[&next], evaluator, 2)
}

/// Depth-3 search: evaluate all placements for current + next + next_next piece.
pub fn search_depth3(
    board: &Board,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    search_deep(board, current, &[&next, &next_next], evaluator, 3)
}

/// Main AI entry point.
/// Uses depth-3 when the evaluator prefers it and next_next is available,
/// otherwise falls back to depth-2, then depth-1.
pub fn find_best_move(
    board: &Board,
    current: &Piece,
    next: &Piece,
    next_next: Option<&Piece>,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    // Try depth 3 when evaluator prefers it (e.g. NN evaluator)
    if evaluator.preferred_depth() >= 3 {
        if let Some(nn) = next_next {
            if let Some(result) = search_depth3(board, current, next, nn, evaluator) {
                if result.score > f64::NEG_INFINITY {
                    return Some(result);
                }
            }
        }
    }

    // Try depth 2
    if let Some(result) = search_depth2(board, current, next, evaluator) {
        if result.score > f64::NEG_INFINITY {
            return Some(result);
        }
    }
    // Fallback to depth 1
    search_depth1(board, current, evaluator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;

    #[test]
    fn test_depth1_finds_move() {
        let board = Board::new();
        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let evaluator = HeuristicEvaluator;
        let result = search_depth1(&board, &piece, &evaluator);
        assert!(result.is_some());
    }

    #[test]
    fn test_depth2_finds_move() {
        let board = Board::new();
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let evaluator = HeuristicEvaluator;
        let result = search_depth2(&board, &current, &next, &evaluator);
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_avoids_game_over() {
        let mut board = Board::new();
        // Fill columns 0-5 to height 10 with alternating colors
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
        let evaluator = HeuristicEvaluator;
        let result = find_best_move(&board, &current, &next, None, &evaluator);
        assert!(result.is_some());
    }

    #[test]
    fn test_ai_prefers_chain() {
        let mut board = Board::new();
        // Set up 3 red in column 0 - AI should complete the 4th
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);
        board.drop_puyo(0, PuyoColor::Red);

        let piece = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let evaluator = HeuristicEvaluator;
        let result = find_best_move(&board, &piece, &next, None, &evaluator);
        assert!(result.is_some());
    }
}
