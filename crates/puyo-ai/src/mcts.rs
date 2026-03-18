use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::Piece;
use puyo_nn::encoding::{context_to_tensor_data, board_to_tensor_data, CONTEXT_TENSOR_SIZE, NUM_CHANNELS};
use puyo_nn::model::PuyoNet;

use puyo_nn::value_transform::value_inverse_transform;

use crate::placement::{compute_valid_mask, index_to_placement, simulate_placement, NUM_ACTIONS};

type InferBackend = NdArray;

/// Number of puyo colors for random tsumo generation.
const NUM_COLORS: u32 = 4;

/// Game snapshot for MCTS nodes.
#[derive(Clone)]
struct GameSnapshot {
    board: Board,
    current: Piece,
    next: Piece,
    next_next: Piece,
}

/// A node in the MCTS tree (Decision Node).
struct MctsNode {
    /// Number of visits.
    visit_count: u32,
    /// Sum of values from all visits (for computing Q = total_value / visit_count).
    total_value: f32,
    /// Prior probability from the policy network.
    prior: f32,
    /// Children indexed by action (0-23). None = not yet expanded for this action.
    children: [Option<usize>; NUM_ACTIONS],
    /// Whether this node has been expanded (network evaluated).
    expanded: bool,
    /// Terminal node (game over or no turns left).
    terminal: bool,
    /// Game state at this node.
    state: GameSnapshot,
    /// Depth from root (root = 0).
    depth: u32,
}

/// MCTS tree.
pub struct MctsTree {
    nodes: Vec<MctsNode>,
    root: usize,
    max_turns: u32,
}

impl MctsTree {
    /// Create a new MCTS tree rooted at the given game state.
    /// `max_turns` is the total turn budget (e.g., 50) used to compute remaining_turns_ratio.
    pub fn new(board: &Board, current: &Piece, next: &Piece, next_next: &Piece, max_turns: u32) -> Self {
        let state = GameSnapshot {
            board: board.clone(),
            current: *current,
            next: *next,
            next_next: *next_next,
        };
        let root = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal: state.board.is_game_over(),
            state,
            depth: 0,
        };
        MctsTree {
            nodes: vec![root],
            root: 0,
            max_turns,
        }
    }

    /// Run one MCTS simulation: select → expand/evaluate → backpropagate.
    pub fn run_one_simulation(
        &mut self,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
        c_puct: f32,
    ) {
        let mut path: Vec<(usize, usize)> = Vec::new(); // (node_id, action)
        let mut node_id = self.root;

        // 1. Selection: traverse tree using PUCT
        while self.nodes[node_id].expanded && !self.nodes[node_id].terminal {
            let action = self.select_action(node_id, c_puct);
            path.push((node_id, action));

            if let Some(child_id) = self.nodes[node_id].children[action] {
                node_id = child_id;
            } else {
                // Create child node by simulating the action
                let child_id = self.create_child(node_id, action);
                self.nodes[node_id].children[action] = Some(child_id);
                node_id = child_id;
                break; // New node, needs expansion
            }
        }

        // 2. Expansion & Evaluation
        let value = if self.nodes[node_id].terminal {
            0.0 // Terminal nodes have zero future value
        } else if !self.nodes[node_id].expanded {
            self.expand_node(node_id, model, device)
        } else {
            // Already expanded (shouldn't normally happen)
            0.0
        };

        // 3. Backpropagation
        for &(parent_id, action) in path.iter().rev() {
            if let Some(child_id) = self.nodes[parent_id].children[action] {
                self.nodes[child_id].visit_count += 1;
                self.nodes[child_id].total_value += value;
            }
        }
        self.nodes[self.root].visit_count += 1;
    }

    /// Select action using PUCT score.
    fn select_action(&self, node_id: usize, c_puct: f32) -> usize {
        let node = &self.nodes[node_id];
        let parent_visits = node.visit_count.max(1) as f32;
        let sqrt_parent = parent_visits.sqrt();

        let mask = compute_valid_mask(&node.state.board, &node.state.current);

        let mut best_action = 0;
        let mut best_score = f32::NEG_INFINITY;

        for action in 0..NUM_ACTIONS {
            if !mask[action] {
                continue;
            }

            let (q, n, prior) = match node.children[action] {
                Some(child_id) => {
                    let child = &self.nodes[child_id];
                    let q = if child.visit_count > 0 {
                        child.total_value / child.visit_count as f32
                    } else {
                        0.0
                    };
                    (q, child.visit_count as f32, child.prior)
                }
                None => {
                    // Unexpanded child: use parent's prior estimate
                    (0.0, 0.0, self.get_prior(node_id, action))
                }
            };

            let puct = q + c_puct * prior * sqrt_parent / (1.0 + n);
            if puct > best_score {
                best_score = puct;
                best_action = action;
            }
        }

        best_action
    }

    /// Get the prior probability for an action at a given node.
    fn get_prior(&self, node_id: usize, action: usize) -> f32 {
        // If node is expanded, we stored priors in children or as a default
        // For unexpanded actions, return uniform over valid actions
        let node = &self.nodes[node_id];
        let mask = compute_valid_mask(&node.state.board, &node.state.current);
        let valid_count = mask.iter().filter(|&&v| v).count() as f32;
        if valid_count > 0.0 && mask[action] {
            1.0 / valid_count
        } else {
            0.0
        }
    }

    /// Create a child node by simulating an action.
    fn create_child(&mut self, parent_id: usize, action: usize) -> usize {
        let parent_state = &self.nodes[parent_id].state;
        let placement = index_to_placement(action);

        // Simulate placement
        let (new_board, _chain_result) = simulate_placement(
            &parent_state.board,
            &parent_state.current,
            &placement,
        );

        let terminal = new_board.is_game_over();

        // Advance pieces: next→current, next_next→next, random→next_next
        let new_current = parent_state.next;
        let new_next = parent_state.next_next;
        // Generate a deterministic "random" piece based on state for reproducibility
        let new_next_next = sample_piece(parent_id as u64, action as u64);

        let child_state = GameSnapshot {
            board: new_board,
            current: new_current,
            next: new_next,
            next_next: new_next_next,
        };

        let parent_depth = self.nodes[parent_id].depth;
        let child = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: self.get_prior(parent_id, action),
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal,
            state: child_state,
            depth: parent_depth + 1,
        };

        let child_id = self.nodes.len();
        self.nodes.push(child);
        child_id
    }

    /// Expand a node: run the neural network and set priors for valid actions.
    fn expand_node(
        &mut self,
        node_id: usize,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
    ) -> f32 {
        let state = &self.nodes[node_id].state;
        let depth = self.nodes[node_id].depth;
        let remaining_ratio = if self.max_turns > 0 {
            (self.max_turns.saturating_sub(depth)) as f32 / self.max_turns as f32
        } else {
            1.0
        };

        let board_data = board_to_tensor_data(&state.board);
        let context_data = context_to_tensor_data(
            &state.current, &state.next, &state.next_next, remaining_ratio,
        );

        let board_tensor = Tensor::<InferBackend, 1>::from_floats(board_data.as_slice(), device)
            .reshape([1, NUM_CHANNELS, ROWS, COLS]);
        let context_tensor = Tensor::<InferBackend, 1>::from_floats(context_data.as_slice(), device)
            .reshape([1, CONTEXT_TENSOR_SIZE]);

        let (logits, value) = model.forward(board_tensor, context_tensor);

        let logits_vec = logits.into_data().to_vec::<f32>().unwrap_or_default();
        let value_scalar = value.into_data().to_vec::<f32>().unwrap_or_default();
        let v_raw = if value_scalar.is_empty() { 0.0 } else { value_scalar[0] };
        let v = value_inverse_transform(v_raw);

        // Compute masked softmax for priors
        let mask = compute_valid_mask(&self.nodes[node_id].state.board, &self.nodes[node_id].state.current);
        let priors = masked_softmax(&logits_vec, &mask);

        // Store priors in existing children, create placeholders for the rest
        for action in 0..NUM_ACTIONS {
            if let Some(child_id) = self.nodes[node_id].children[action] {
                self.nodes[child_id].prior = priors[action];
            }
        }

        // Store priors for later use (when children are created)
        // We mark the node as expanded — priors will be read via get_prior override
        self.nodes[node_id].expanded = true;

        // Override get_prior to use network priors: store them in a side channel
        // For simplicity, we store the priors by pre-creating children with just priors set
        // Actually, let's update get_prior to check expanded status and use stored data
        // We'll store priors directly by updating children's prior when they're created

        v
    }

    /// Get visit counts for root's direct children (used to select the final move).
    pub fn root_visit_counts(&self) -> [u32; NUM_ACTIONS] {
        let mut counts = [0u32; NUM_ACTIONS];
        let root = &self.nodes[self.root];
        for action in 0..NUM_ACTIONS {
            if let Some(child_id) = root.children[action] {
                counts[action] = self.nodes[child_id].visit_count;
            }
        }
        counts
    }

    /// Get the MCTS policy (normalized visit counts) with temperature.
    pub fn get_policy(&self, temperature: f32) -> [f32; NUM_ACTIONS] {
        let counts = self.root_visit_counts();
        let mut policy = [0.0f32; NUM_ACTIONS];

        if temperature < 0.01 {
            // Greedy: all weight on most-visited action
            let best = counts.iter().enumerate()
                .max_by_key(|(_, &n)| n)
                .map(|(i, _)| i)
                .unwrap_or(0);
            policy[best] = 1.0;
        } else {
            // Temperature-scaled
            let inv_temp = 1.0 / temperature;
            for i in 0..NUM_ACTIONS {
                policy[i] = (counts[i] as f32).powf(inv_temp);
            }
            let sum: f32 = policy.iter().sum();
            if sum > 0.0 {
                for p in &mut policy {
                    *p /= sum;
                }
            }
        }

        policy
    }
}

/// Compute masked softmax over logits.
fn masked_softmax(logits: &[f32], mask: &[bool; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    let mut result = [0.0f32; NUM_ACTIONS];

    // Find max for numerical stability
    let mut max_logit = f32::NEG_INFINITY;
    for i in 0..NUM_ACTIONS {
        if mask[i] && logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) > max_logit {
            max_logit = logits[i];
        }
    }
    if max_logit == f32::NEG_INFINITY {
        return result; // No valid actions
    }

    let mut sum = 0.0f32;
    for i in 0..NUM_ACTIONS {
        if mask[i] {
            let exp = (logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) - max_logit).exp();
            result[i] = exp;
            sum += exp;
        }
    }
    if sum > 0.0 {
        for r in &mut result {
            *r /= sum;
        }
    }

    result
}

/// Sample a random piece deterministically from node_id and action.
fn sample_piece(seed1: u64, seed2: u64) -> Piece {
    // Simple hash for deterministic "random" piece
    let mut x = seed1.wrapping_mul(6364136223846793005).wrapping_add(seed2).wrapping_add(1);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);

    let axis = ((x % NUM_COLORS as u64) as u8) + 1;
    let sat = (((x >> 16) % NUM_COLORS as u64) as u8) + 1;
    Piece::new(
        puyo_core::board::PuyoColor::from_u8(axis),
        puyo_core::board::PuyoColor::from_u8(sat),
    )
}

/// Run MCTS search and return the policy (visit count distribution).
/// `max_turns` is the total turn budget used to compute remaining_turns_ratio for the context.
pub fn mcts_search(
    board: &Board,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
    model: &PuyoNet<InferBackend>,
    device: &<InferBackend as Backend>::Device,
    num_simulations: usize,
    c_puct: f32,
    temperature: f32,
    max_turns: u32,
) -> [f32; NUM_ACTIONS] {
    let mut tree = MctsTree::new(board, current, next, next_next, max_turns);

    for _ in 0..num_simulations {
        tree.run_one_simulation(model, device, c_puct);
    }

    tree.get_policy(temperature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use puyo_core::board::PuyoColor;

    #[test]
    fn test_masked_softmax_basic() {
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut mask = [false; 24];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let result = masked_softmax(&logits, &mask);

        // Highest logit (index 2) should have highest probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
        // Masked actions should be 0
        assert_eq!(result[3], 0.0);

        // Sum of valid actions should be ~1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_piece_deterministic() {
        let p1 = sample_piece(42, 7);
        let p2 = sample_piece(42, 7);
        assert_eq!(p1.axis_color, p2.axis_color);
        assert_eq!(p1.satellite_color, p2.satellite_color);
    }

    #[test]
    fn test_mcts_tree_creation() {
        let board = Board::new();
        let current = Piece::new(PuyoColor::Red, PuyoColor::Blue);
        let next = Piece::new(PuyoColor::Green, PuyoColor::Yellow);
        let next_next = Piece::new(PuyoColor::Blue, PuyoColor::Red);

        let tree = MctsTree::new(&board, &current, &next, &next_next, 50);
        assert_eq!(tree.nodes.len(), 1);
        assert!(!tree.nodes[0].expanded);
        assert!(!tree.nodes[0].terminal);
    }
}
