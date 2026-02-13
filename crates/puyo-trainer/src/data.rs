use serde::{Deserialize, Serialize};

/// A single training sample: board state + target chain count.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sample {
    /// One-hot encoded board state (5 channels × 13 rows × 6 cols = 390 floats).
    pub board_data: Vec<f32>,
    /// Target value: chain count achieved from this board state.
    pub target: f32,
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
