use burn::prelude::*;

/// Trait abstracting a neural network model for batched inference.
/// Implementations provide the forward pass and value post-processing.
pub trait GameModel<B: Backend>: Send + 'static {
    /// Board tensor shape: (channels, height, width).
    fn board_shape(&self) -> (usize, usize, usize);
    /// Context tensor size.
    fn context_size(&self) -> usize;
    /// Number of actions (policy output size).
    fn num_actions(&self) -> usize;
    /// Run forward pass. Returns (logits [batch, actions], values [batch, 1]).
    fn forward(&self, board: Tensor<B, 4>, context: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);
    /// Post-process raw value scalar from the network (e.g., inverse transform).
    fn postprocess_value(&self, raw: f32) -> f32;
}
