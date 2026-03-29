use crate::board::{Board, COLS, NUM_COLORS, ROWS};
use crate::piece::Piece;

pub use crate::config::{CONTEXT_TENSOR_SIZE, NUM_CHANNELS, PIECE_TENSOR_SIZE, TENSOR_SIZE};

/// MCTS/AI用の軽量ゲーム状態。
/// GameStateからUI関連（FallingPiece, phase等）を除いた純粋な盤面+ピースキュー。
#[derive(Clone)]
pub struct PuyoState {
    pub board: Board,
    pub current: Piece,
    pub next: Piece,
    pub next_next: Piece,
}

/// Board を [channel][row][col] のフラット f32 配列に変換する。
///
/// チャンネル:
/// - 0..(NUM_COLORS-1): 色ごとのone-hot
/// - NUM_COLORS: occupancy map (ぷよの有無)
/// - NUM_COLORS+1: adjacency map (同色隣接数 / 4.0)
pub fn board_to_tensor_data(board: &Board) -> Vec<f32> {
    let mut data = vec![0.0f32; TENSOR_SIZE];

    let occ_offset = NUM_COLORS * ROWS * COLS;
    let adj_offset = (NUM_COLORS + 1) * ROWS * COLS;

    for col in 0..COLS {
        for row in 0..ROWS {
            let color = board.get(col, row);
            if color.is_color() {
                let ch = color as u8 as usize - 1;
                data[ch * ROWS * COLS + row * COLS + col] = 1.0;

                data[occ_offset + row * COLS + col] = 1.0;

                let count = [
                    col > 0 && board.get(col - 1, row) == color,
                    col + 1 < COLS && board.get(col + 1, row) == color,
                    row > 0 && board.get(col, row - 1) == color,
                    row + 1 < ROWS && board.get(col, row + 1) == color,
                ]
                .iter()
                .filter(|&&b| b)
                .count();
                data[adj_offset + row * COLS + col] = count as f32 / 4.0;
            }
        }
    }

    data
}

/// 3ピース（current, next, next_next）をフラット f32 配列に変換する。
/// 各ピースの axis_color, satellite_color を NUM_COLORS 次元 one-hot でエンコード。
pub fn pieces_to_tensor_data(current: &Piece, next: &Piece, next_next: &Piece) -> Vec<f32> {
    let mut data = vec![0.0f32; PIECE_TENSOR_SIZE];
    let pieces = [current, next, next_next];
    for (i, piece) in pieces.iter().enumerate() {
        let base = i * (2 * NUM_COLORS);
        let axis_ch = piece.axis_color as u8 as usize - 1;
        data[base + axis_ch] = 1.0;
        let sat_ch = piece.satellite_color as u8 as usize - 1;
        data[base + NUM_COLORS + sat_ch] = 1.0;
    }
    data
}

/// コンテキストテンソル（FiLM条件付け用）に変換する。
pub fn context_to_tensor_data(
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
) -> Vec<f32> {
    pieces_to_tensor_data(current, next, next_next)
}
