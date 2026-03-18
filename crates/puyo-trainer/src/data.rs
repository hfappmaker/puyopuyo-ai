use serde::{Deserialize, Serialize};

/// A single training sample: board state + context info + chosen action.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sample {
    /// Encoded board state (NUM_CHANNELS × 14 rows × 6 cols floats).
    pub board_data: Vec<f32>,
    /// Encoded context (3 pieces one-hot 24 + remaining turns ratio 1 = 25 floats).
    pub context_data: Vec<f32>,
    /// Action index chosen by the teacher evaluator (0-23).
    pub action_index: u8,
}

/// A single AlphaZero self-play sample.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlphaZeroSample {
    /// Encoded board state.
    pub board_data: Vec<f32>,
    /// Encoded context (3 pieces one-hot 24 + remaining turns ratio 1 = 25 floats).
    pub context_data: Vec<f32>,
    /// MCTS visit-count policy (24 floats, sums to 1.0).
    pub mcts_policy: Vec<f32>,
    /// Value target: discounted cumulative reward from this step.
    pub value_target: f32,
}

/// AlphaZero dataset of self-play samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct AlphaZeroDataset {
    pub samples: Vec<AlphaZeroSample>,
}

impl AlphaZeroDataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).expect("Failed to serialize dataset");
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: AlphaZeroDataset = bincode::deserialize(&bytes).expect("Failed to deserialize dataset");
        Ok(dataset)
    }
}

/// Dataset of training samples.
#[derive(Serialize, Deserialize, Debug)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).expect("Failed to serialize dataset");
        std::fs::write(path, bytes)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let dataset: Dataset = bincode::deserialize(&bytes).expect("Failed to deserialize dataset");
        Ok(dataset)
    }
}
