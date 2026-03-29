pub mod eval;
#[cfg(feature = "nn")]
pub mod nn_eval;
pub mod puyo_game;

pub use puyo_core::placement;

// Re-export game-ai types for convenience
pub use game_ai::eval::Evaluator;

#[cfg(feature = "nn")]
pub use game_ai::{inference_server, mcts, model::GameModel};
#[cfg(feature = "nn")]
pub use game_ai::nn_eval::{DirectInference, MctsConfig};
