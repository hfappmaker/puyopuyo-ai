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
    /// NN policy priors for each action (set when expanded).
    priors: [f32; NUM_ACTIONS],
    /// Children indexed by action (0-23). None = not yet expanded for this action.
    children: [Option<usize>; NUM_ACTIONS],
    /// Whether this node has been expanded (network evaluated).
    expanded: bool,
    /// Terminal node (game over or no turns left).
    terminal: bool,
    /// Immediate reward received when transitioning TO this node (chain score).
    immediate_reward: f32,
    /// Game state at this node.
    state: GameSnapshot,
}

/// MCTS tree.
pub struct MctsTree {
    nodes: Vec<MctsNode>,
    root: usize,
    /// Discount factor for future rewards.
    gamma: f32,
    /// Minimum Q value observed in the tree (for Min-Max normalization).
    min_value: f32,
    /// Maximum Q value observed in the tree (for Min-Max normalization).
    max_value: f32,
}

impl MctsTree {
    /// Create a new MCTS tree rooted at the given game state.
    /// `gamma` is the discount factor for future rewards.
    pub fn new(board: &Board, current: &Piece, next: &Piece, next_next: &Piece, gamma: f32) -> Self {
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
            priors: [0.0; NUM_ACTIONS],
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal: state.board.is_game_over(),
            immediate_reward: 0.0,
            state,
        };
        MctsTree {
            nodes: vec![root],
            root: 0,
            gamma,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
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

        // 3. Backpropagation with discounted rewards: backup = r + gamma * backup
        let mut backup = value;
        for &(parent_id, action) in path.iter().rev() {
            if let Some(child_id) = self.nodes[parent_id].children[action] {
                let reward = self.nodes[child_id].immediate_reward;
                backup = reward + self.gamma * backup;
                self.min_value = self.min_value.min(backup);
                self.max_value = self.max_value.max(backup);
                self.nodes[child_id].visit_count += 1;
                self.nodes[child_id].total_value += backup;
            }
        }
        self.nodes[self.root].visit_count += 1;
    }

    /// Normalize a Q value to [0, 1] using min-max normalization (MuZero Reanalyze).
    /// Returns 0.5 when min == max (no information yet).
    fn normalize_q(&self, q: f32) -> f32 {
        let range = self.max_value - self.min_value;
        if range > f32::EPSILON {
            ((q - self.min_value) / range).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Select action using PUCT score with Min-Max normalized Q values.
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

            let prior = node.priors[action];
            let (q, n) = match node.children[action] {
                Some(child_id) => {
                    let child = &self.nodes[child_id];
                    let q = if child.visit_count > 0 {
                        child.total_value / child.visit_count as f32
                    } else {
                        0.0
                    };
                    (q, child.visit_count as f32)
                }
                None => (0.0, 0.0),
            };

            let q_normalized = self.normalize_q(q);
            let puct = q_normalized + c_puct * prior * sqrt_parent / (1.0 + n);
            if puct > best_score {
                best_score = puct;
                best_action = action;
            }
        }

        best_action
    }

    /// Create a child node by simulating an action.
    fn create_child(&mut self, parent_id: usize, action: usize) -> usize {
        let parent_state = &self.nodes[parent_id].state;
        let placement = index_to_placement(action);

        // Simulate placement
        let (new_board, chain_result) = simulate_placement(
            &parent_state.board,
            &parent_state.current,
            &placement,
        );

        let terminal = new_board.is_game_over();
        let immediate_reward = chain_result.score as f32;

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

        let child = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: self.nodes[parent_id].priors[action],
            priors: [0.0; NUM_ACTIONS],
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal,
            immediate_reward,
            state: child_state,
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

        let board_data = board_to_tensor_data(&state.board);
        let context_data = context_to_tensor_data(
            &state.current, &state.next, &state.next_next,
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

        // Store priors in the node for use by select_action and create_child
        self.nodes[node_id].priors = priors;

        // Update priors of already-created children
        for action in 0..NUM_ACTIONS {
            if let Some(child_id) = self.nodes[node_id].children[action] {
                self.nodes[child_id].prior = priors[action];
            }
        }

        self.nodes[node_id].expanded = true;

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

    /// Get Q values (average cumulative reward) for root's direct children.
    pub fn root_q_values(&self) -> [f32; NUM_ACTIONS] {
        let mut q_values = [0.0f32; NUM_ACTIONS];
        let root = &self.nodes[self.root];
        for action in 0..NUM_ACTIONS {
            if let Some(child_id) = root.children[action] {
                let child = &self.nodes[child_id];
                if child.visit_count > 0 {
                    q_values[action] = child.total_value / child.visit_count as f32;
                }
            }
        }
        q_values
    }

    /// Apply Dirichlet noise to root node priors for exploration diversity.
    /// `P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)`
    pub fn apply_root_dirichlet_noise(&mut self, alpha: f32, epsilon: f32, seed: u64) {
        let root = self.root;
        if !self.nodes[root].expanded {
            return;
        }
        let mask = compute_valid_mask(
            &self.nodes[root].state.board,
            &self.nodes[root].state.current,
        );
        let num_valid = mask.iter().filter(|&&v| v).count();
        if num_valid == 0 {
            return;
        }

        let noise = sample_dirichlet(alpha, num_valid, seed);
        let mut noise_idx = 0;
        for action in 0..NUM_ACTIONS {
            if mask[action] {
                let p = self.nodes[root].priors[action];
                self.nodes[root].priors[action] =
                    (1.0 - epsilon) * p + epsilon * noise[noise_idx];
                noise_idx += 1;
            }
        }
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

/// Sample from a Dirichlet distribution with concentration parameter `alpha`.
/// Returns a vector of `n` values summing to 1.0.
/// Uses Marsaglia-Tsang method for Gamma sampling with a simple xorshift64 PRNG.
fn sample_dirichlet(alpha: f32, n: usize, seed: u64) -> Vec<f32> {
    let mut rng_state = seed.wrapping_add(1); // avoid 0

    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let g = sample_gamma(alpha, &mut rng_state, i as u64);
        samples.push(g);
    }
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Fallback: uniform
        let u = 1.0 / n as f32;
        for s in &mut samples {
            *s = u;
        }
    }
    samples
}

/// Sample from Gamma(alpha, 1) using Marsaglia-Tsang method.
/// For alpha < 1, uses the boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha).
fn sample_gamma(alpha: f32, rng: &mut u64, extra_seed: u64) -> f32 {
    let alpha = alpha as f64;
    let (alpha_use, boost) = if alpha < 1.0 {
        let u = xorshift64_f64(rng, extra_seed);
        (alpha + 1.0, u.powf(1.0 / alpha))
    } else {
        (alpha, 1.0)
    };

    let d = alpha_use - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        // Generate normal using Box-Muller
        let u1 = xorshift64_f64(rng, 0).max(1e-15);
        let u2 = xorshift64_f64(rng, 1);
        let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        let v = (1.0 + c * n).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = xorshift64_f64(rng, 2);
        // Acceptance test
        if u < 1.0 - 0.0331 * n * n * n * n
            || u.ln() < 0.5 * n * n + d * (1.0 - v + v.ln())
        {
            return (d * v * boost) as f32;
        }
    }
}

/// Simple xorshift64-based PRNG returning f64 in [0, 1).
fn xorshift64_f64(state: &mut u64, mix: u64) -> f64 {
    let mut s = (*state).wrapping_add(mix.wrapping_mul(2654435761));
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s >> 11) as f64 / ((1u64 << 53) as f64)
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

/// Dirichlet noise configuration for root exploration.
pub struct DirichletConfig {
    pub alpha: f32,
    pub epsilon: f32,
}

/// Run MCTS search and return the policy (visit count distribution) and Q values.
/// `gamma` is the discount factor for future rewards (e.g. 0.99).
/// `dirichlet` adds Dirichlet noise to root priors for exploration (used in self-play).
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
    gamma: f32,
    dirichlet: Option<&DirichletConfig>,
) -> ([f32; NUM_ACTIONS], [f32; NUM_ACTIONS]) {
    let mut tree = MctsTree::new(board, current, next, next_next, gamma);

    // Run first simulation to expand root node
    if num_simulations > 0 {
        tree.run_one_simulation(model, device, c_puct);
    }

    // Apply Dirichlet noise to root priors after root expansion
    if let Some(dir) = dirichlet {
        // Use hash of all column heights for diverse seeds
        let mut seed = 0u64;
        for col in 0..COLS {
            seed = seed.wrapping_mul(6364136223846793005)
                .wrapping_add(board.columns[col].len() as u64);
        }
        tree.apply_root_dirichlet_noise(dir.alpha, dir.epsilon, seed);
    }

    // Remaining simulations
    for _ in 1..num_simulations {
        tree.run_one_simulation(model, device, c_puct);
    }

    (tree.get_policy(temperature), tree.root_q_values())
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

        let tree = MctsTree::new(&board, &current, &next, &next_next, 0.99);
        assert_eq!(tree.nodes.len(), 1);
        assert!(!tree.nodes[0].expanded);
        assert!(!tree.nodes[0].terminal);
    }
}
