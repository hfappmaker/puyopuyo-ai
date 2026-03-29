use serde::{Deserialize, Serialize};

use puyo_core::board::{COLS, NUM_COLORS, ROWS};
use puyo_core::state::PIECE_TENSOR_SIZE;

// ─── Color Permutation Data Augmentation ───

/// 色の置換テーブル。perm[old_ch] = new_ch
pub type ColorPermutation = [usize; NUM_COLORS];

const PLANE_SIZE: usize = ROWS * COLS;

/// NUM_COLORS! 通りの全色置換を返す（恒等置換を含む）。
pub fn all_color_permutations() -> Vec<ColorPermutation> {
    let mut result = Vec::new();
    let mut current = [0usize; NUM_COLORS];
    let mut used = [false; NUM_COLORS];
    generate_perms(&mut result, &mut current, &mut used, 0);
    result
}

fn generate_perms(
    result: &mut Vec<ColorPermutation>,
    current: &mut [usize; NUM_COLORS],
    used: &mut [bool; NUM_COLORS],
    depth: usize,
) {
    if depth == NUM_COLORS {
        result.push(*current);
        return;
    }
    for i in 0..NUM_COLORS {
        if !used[i] {
            used[i] = true;
            current[depth] = i;
            generate_perms(result, current, used, depth + 1);
            used[i] = false;
        }
    }
}

/// board_data にインプレースで色置換を適用する。
/// board_data は [NUM_CHANNELS ch][ROWS][COLS] floats。ch0..(NUM_COLORS-1)を入れ替え、残りは不変。
pub fn apply_color_perm_board(board_data: &mut [f32], perm: &[usize]) {
    let mut color_planes = [0.0f32; NUM_COLORS * PLANE_SIZE];
    color_planes.copy_from_slice(&board_data[..NUM_COLORS * PLANE_SIZE]);

    for old_ch in 0..NUM_COLORS {
        let new_ch = perm[old_ch];
        let src_start = old_ch * PLANE_SIZE;
        let dst_start = new_ch * PLANE_SIZE;
        board_data[dst_start..dst_start + PLANE_SIZE]
            .copy_from_slice(&color_planes[src_start..src_start + PLANE_SIZE]);
    }
}

/// context_data にインプレースで色置換を適用する。
/// context_data は 6つのNUM_COLORS要素one-hotブロック = PIECE_TENSOR_SIZE floats。
pub fn apply_color_perm_context(context_data: &mut [f32], perm: &[usize]) {
    for block_start in (0..PIECE_TENSOR_SIZE).step_by(NUM_COLORS) {
        let mut tmp = [0.0f32; NUM_COLORS];
        tmp.copy_from_slice(&context_data[block_start..block_start + NUM_COLORS]);
        for old_ch in 0..NUM_COLORS {
            context_data[block_start + perm[old_ch]] = tmp[old_ch];
        }
    }
}

/// A single training sample: board state + context info + chosen action + value estimate.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sample {
    /// Encoded board state (TENSOR_SIZE floats).
    pub board_data: Vec<f32>,
    /// Encoded context (PIECE_TENSOR_SIZE floats).
    pub context_data: Vec<f32>,
    /// Action index chosen by the teacher evaluator.
    pub action_index: u8,
    /// Value target: score estimate from the teacher evaluator.
    pub value_target: f32,
}

pub use az_framework::data::{AlphaZeroDataset, AlphaZeroSample};

/// Dataset of training samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: Dataset = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::state::TENSOR_SIZE;

    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    fn identity_perm() -> ColorPermutation {
        let mut p = [0usize; NUM_COLORS];
        for i in 0..NUM_COLORS { p[i] = i; }
        p
    }

    #[test]
    fn test_all_permutations_count_and_unique() {
        let perms = all_color_permutations();
        let expected_count = factorial(NUM_COLORS);
        assert_eq!(perms.len(), expected_count);
        for i in 0..perms.len() {
            let mut sorted = perms[i];
            sorted.sort();
            assert_eq!(sorted, identity_perm());
            for j in (i + 1)..perms.len() {
                assert_ne!(perms[i], perms[j]);
            }
        }
    }

    #[test]
    fn test_identity_permutation_exists() {
        let perms = all_color_permutations();
        assert!(perms.iter().any(|p| *p == identity_perm()));
    }

    #[test]
    fn test_board_perm_identity() {
        let mut data = vec![0.0f32; TENSOR_SIZE];
        data[0] = 1.0;
        let original = data.clone();
        apply_color_perm_board(&mut data, &identity_perm());
        assert_eq!(data, original);
    }

    #[test]
    fn test_board_perm_swap_first_two() {
        let mut data = vec![0.0f32; TENSOR_SIZE];
        data[0] = 1.0; // ch0, position 0
        let mut perm = identity_perm();
        perm.swap(0, 1);
        apply_color_perm_board(&mut data, &perm);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[PLANE_SIZE], 1.0);
    }

    #[test]
    fn test_board_perm_preserves_non_color_channels() {
        let mut data = vec![0.0f32; TENSOR_SIZE];
        let occ_offset = NUM_COLORS * PLANE_SIZE;
        let adj_offset = (NUM_COLORS + 1) * PLANE_SIZE;
        data[occ_offset] = 1.0;
        data[adj_offset] = 0.5;
        let mut perm = identity_perm();
        perm.reverse();
        apply_color_perm_board(&mut data, &perm);
        assert_eq!(data[occ_offset], 1.0);
        assert_eq!(data[adj_offset], 0.5);
    }

    #[test]
    fn test_context_perm_swap() {
        let nc = NUM_COLORS;
        let mut ctx = vec![0.0f32; PIECE_TENSOR_SIZE];
        ctx[0] = 1.0;
        ctx[nc + 2 % nc] = 1.0;
        let mut perm = identity_perm();
        perm.swap(0, 1);
        apply_color_perm_context(&mut ctx, &perm);
        assert_eq!(ctx[0], 0.0);
        assert_eq!(ctx[1], 1.0);
    }

}
