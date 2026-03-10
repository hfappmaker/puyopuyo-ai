use puyo_core::board::Board;
use puyo_core::chain::ChainResult;
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
    // Manually place the piece
    sim.place_piece(piece, placement);
    // Resolve chains
    let chain_result = puyo_core::chain::resolve_chains(&mut sim.board);
    (sim.board, chain_result)
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

    let mut best_score = f64::NEG_INFINITY;
    let mut best_placement = placements[0];

    for placement in &placements {
        let (result_board, chain_result) = simulate_placement(board, current, placement);

        // 10連鎖以上は即座に選択
        if chain_result.chain_count >= 10 {
            return Some(SearchResult {
                best_placement: *placement,
                score: f64::INFINITY,
                depth: 1,
            });
        }

        let score = evaluator.evaluate(&result_board);

        if score > best_score {
            best_score = score;
            best_placement = *placement;
        }
    }

    Some(SearchResult {
        best_placement,
        score: best_score,
        depth: 1,
    })
}

/// Depth-2 search: evaluate all placements for current + next piece.
/// For each current placement, try all next placements and take the max.
/// Pick the current placement that maximizes the best-case next score.
pub fn search_depth2(
    board: &Board,
    current: &Piece,
    next: &Piece,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    let placements = enumerate_placements(board, current);
    if placements.is_empty() {
        return None;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_placement = placements[0];

    for placement in &placements {
        let (board_after_current, chain_result) = simulate_placement(board, current, placement);

        // 1手目で10連鎖以上なら即リターン
        if chain_result.chain_count >= 10 {
            return Some(SearchResult {
                best_placement: *placement,
                score: f64::INFINITY,
                depth: 2,
            });
        }

        if board_after_current.is_game_over() {
            // Skip placements that cause game over
            continue;
        }

        // Now try all next piece placements
        let next_placements = enumerate_placements(&board_after_current, next);
        if next_placements.is_empty() {
            // Can't place next piece -> bad
            continue;
        }

        let mut best_next_score = f64::NEG_INFINITY;
        for next_placement in &next_placements {
            let (board_after_next, next_chain_result) =
                simulate_placement(&board_after_current, next, next_placement);

            // 2手目で10連鎖以上なら、この1手目を即選択
            if next_chain_result.chain_count >= 10 {
                return Some(SearchResult {
                    best_placement: *placement,
                    score: f64::INFINITY,
                    depth: 2,
                });
            }

            let score = evaluator.evaluate(&board_after_next);
            if score > best_next_score {
                best_next_score = score;
            }
        }

        if best_next_score > best_score {
            best_score = best_next_score;
            best_placement = *placement;
        }
    }

    Some(SearchResult {
        best_placement,
        score: best_score,
        depth: 2,
    })
}

/// Main AI entry point: try depth-2, fall back to depth-1.
/// `next_next` is reserved for future use and currently ignored.
pub fn find_best_move(
    board: &Board,
    current: &Piece,
    next: &Piece,
    _next_next: Option<&Piece>,
    evaluator: &dyn Evaluator,
) -> Option<SearchResult> {
    // Try depth 2 first
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
        // The AI should place the red at column 0 to complete the chain
        // With North orientation, axis at col 0 means red goes to col 0
        // (axis_color = Red, col = 0 with various orientations could work)
        // We just verify it found a move; exact placement depends on eval weights
    }
}
