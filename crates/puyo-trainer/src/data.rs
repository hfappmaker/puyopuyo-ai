use serde::{Deserialize, Serialize};

// ─── Color Permutation Data Augmentation ───

/// 4色の置換テーブル。perm[old_ch] = new_ch (0=Red, 1=Green, 2=Blue, 3=Yellow)
pub type ColorPermutation = [usize; 4];

const PLANE_SIZE: usize = 14 * 6; // ROWS * COLS = 84

/// 24通りの全色置換を返す（恒等置換を含む）。
pub fn all_color_permutations() -> [ColorPermutation; 24] {
    let mut perms = [[0usize; 4]; 24];
    let mut idx = 0;
    for a in 0..4 {
        for b in 0..4 {
            if b == a { continue; }
            for c in 0..4 {
                if c == a || c == b { continue; }
                let d = 6 - a - b - c; // 0+1+2+3=6
                perms[idx] = [a, b, c, d];
                idx += 1;
            }
        }
    }
    perms
}

/// board_data にインプレースで色置換を適用する。
/// board_data は [6ch][ROWS][COLS] = 504 floats。ch0-3を入れ替え、ch4,5は不変。
pub fn apply_color_perm_board(board_data: &mut [f32], perm: &ColorPermutation) {
    let mut color_planes = [0.0f32; 4 * PLANE_SIZE];
    color_planes.copy_from_slice(&board_data[..4 * PLANE_SIZE]);

    for old_ch in 0..4 {
        let new_ch = perm[old_ch];
        let src_start = old_ch * PLANE_SIZE;
        let dst_start = new_ch * PLANE_SIZE;
        board_data[dst_start..dst_start + PLANE_SIZE]
            .copy_from_slice(&color_planes[src_start..src_start + PLANE_SIZE]);
    }
}

/// context_data にインプレースで色置換を適用する。
/// context_data は 6つの4要素one-hotブロック = 24 floats。
pub fn apply_color_perm_context(context_data: &mut [f32], perm: &ColorPermutation) {
    for block_start in (0..24).step_by(4) {
        let mut tmp = [0.0f32; 4];
        tmp.copy_from_slice(&context_data[block_start..block_start + 4]);
        for old_ch in 0..4 {
            context_data[block_start + perm[old_ch]] = tmp[old_ch];
        }
    }
}

/// A single training sample: board state + context info + chosen action + value estimate.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sample {
    /// Encoded board state (NUM_CHANNELS × 14 rows × 6 cols floats).
    pub board_data: Vec<f32>,
    /// Encoded context (3 pieces one-hot = 24 floats).
    pub context_data: Vec<f32>,
    /// Action index chosen by the teacher evaluator (0-23).
    pub action_index: u8,
    /// Value target: score estimate from the teacher evaluator.
    pub value_target: f32,
}

/// A single AlphaZero self-play sample.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlphaZeroSample {
    /// Encoded board state.
    pub board_data: Vec<f32>,
    /// Encoded context (3 pieces one-hot = 24 floats).
    pub context_data: Vec<f32>,
    /// MCTS improved policy (24 floats, sums to 1.0).
    pub mcts_policy: Vec<f32>,
    /// Value target: discounted cumulative reward from this step.
    pub value_target: f32,
}

/// AlphaZero dataset of self-play samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlphaZeroDataset {
    pub samples: Vec<AlphaZeroSample>,
}

impl Default for AlphaZeroDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl AlphaZeroDataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: AlphaZeroDataset = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(dataset)
    }

    /// Load and merge multiple dataset files into one.
    pub fn load_multiple(paths: &[String]) -> std::io::Result<Self> {
        let mut merged = Self::new();
        for path in paths {
            let ds = Self::load(path)?;
            println!("  Loaded {} samples from {}", ds.samples.len(), path);
            merged.samples.extend(ds.samples);
        }
        Ok(merged)
    }
}

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

    #[test]
    fn test_all_permutations_count_and_unique() {
        let perms = all_color_permutations();
        assert_eq!(perms.len(), 24);
        for i in 0..24 {
            let mut sorted = perms[i];
            sorted.sort();
            assert_eq!(sorted, [0, 1, 2, 3]);
            for j in (i + 1)..24 {
                assert_ne!(perms[i], perms[j]);
            }
        }
    }

    #[test]
    fn test_identity_permutation_exists() {
        let perms = all_color_permutations();
        assert!(perms.iter().any(|p| *p == [0, 1, 2, 3]));
    }

    #[test]
    fn test_board_perm_identity() {
        let mut data = vec![0.0f32; 504];
        data[0] = 1.0;
        let original = data.clone();
        apply_color_perm_board(&mut data, &[0, 1, 2, 3]);
        assert_eq!(data, original);
    }

    #[test]
    fn test_board_perm_swap_red_green() {
        let mut data = vec![0.0f32; 504];
        data[0] = 1.0; // Red ch0, position 0
        apply_color_perm_board(&mut data, &[1, 0, 2, 3]);
        assert_eq!(data[0], 0.0);   // ch0 now has Green's data (empty)
        assert_eq!(data[84], 1.0);  // ch1 now has Red's data
    }

    #[test]
    fn test_board_perm_preserves_ch4_ch5() {
        let mut data = vec![0.0f32; 504];
        data[336] = 1.0; // ch4
        data[420] = 0.5; // ch5
        apply_color_perm_board(&mut data, &[3, 2, 1, 0]);
        assert_eq!(data[336], 1.0);
        assert_eq!(data[420], 0.5);
    }

    #[test]
    fn test_context_perm_swap() {
        let mut ctx = vec![0.0f32; 24];
        ctx[0] = 1.0; // current_axis = Red
        ctx[6] = 1.0; // current_sat = Blue
        apply_color_perm_context(&mut ctx, &[1, 0, 3, 2]);
        assert_eq!(ctx[0], 0.0);
        assert_eq!(ctx[1], 1.0); // Red -> Green
        assert_eq!(ctx[6], 0.0);
        assert_eq!(ctx[7], 1.0); // Blue -> Yellow
    }

}
