use burn::backend::ndarray::NdArray;
use burn::prelude::*;

use puyo_core::board::{Board, COLS, ROWS};
use puyo_core::piece::Piece;
use puyo_nn::encoding::{context_to_tensor_data, board_to_tensor_data, CONTEXT_TENSOR_SIZE, NUM_CHANNELS};
use puyo_nn::model::PuyoNet;

use puyo_nn::value_transform::value_inverse_transform;

use crate::nn_eval::MctsConfig;
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
    /// Raw NN logits before softmax (set when expanded).
    logits: [f32; NUM_ACTIONS],
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
    /// Cached valid action mask (derived from board + current piece, immutable per node).
    valid_mask: [bool; NUM_ACTIONS],
    /// Depth from root (root = 0).
    depth: u32,
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
    /// Value head output for the root node (used for completed Q-values).
    root_value: f32,
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
        let valid_mask = compute_valid_mask(&state.board, &state.current);
        let root = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: 1.0,
            priors: [0.0; NUM_ACTIONS],
            logits: [0.0; NUM_ACTIONS],
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal: state.board.is_game_over(),
            immediate_reward: 0.0,
            state,
            valid_mask,
            depth: 0,
        };
        MctsTree {
            nodes: vec![root],
            root: 0,
            gamma,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            root_value: 0.0,
        }
    }

    /// Expand the root node and return the value estimate.
    fn expand_root(
        &mut self,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
    ) -> f32 {
        let v = self.expand_node(self.root, model, device);
        self.root_value = v;
        self.nodes[self.root].visit_count += 1;
        v
    }

    /// Run one simulation forcing a specific action at the root, then PUCT for the rest.
    fn simulate_from_root_action(
        &mut self,
        action: usize,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
        c_puct: f32,
    ) {
        let child_id = self.get_or_create_child(self.root, action);
        let mut path: Vec<(usize, usize)> = vec![(self.root, action)];
        let mut node_id = child_id;

        // From child onward, use standard PUCT selection
        while self.nodes[node_id].expanded && !self.nodes[node_id].terminal {
            let act = self.select_action(node_id, c_puct);
            path.push((node_id, act));
            node_id = self.get_or_create_child(node_id, act);
        }

        // Expand leaf
        let value = if self.nodes[node_id].terminal {
            0.0
        } else if !self.nodes[node_id].expanded {
            self.expand_node(node_id, model, device)
        } else {
            0.0
        };

        // Backpropagate: compute cumulative values (immutable)
        let backups: Vec<f32> = path
            .iter()
            .rev()
            .scan(value, |acc, &(parent_id, act)| {
                if let Some(cid) = self.nodes[parent_id].children[act] {
                    *acc = self.nodes[cid].immediate_reward + self.gamma * *acc;
                }
                Some(*acc)
            })
            .collect();

        // Backpropagate: apply updates (mutable)
        for (&(parent_id, act), backup) in path.iter().rev().zip(&backups) {
            if let Some(cid) = self.nodes[parent_id].children[act] {
                self.min_value = self.min_value.min(*backup);
                self.max_value = self.max_value.max(*backup);
                self.nodes[cid].visit_count += 1;
                self.nodes[cid].total_value += *backup;
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

        node.valid_mask.iter()
            .enumerate()
            .filter(|(_, &is_valid)| is_valid)
            .map(|(action, _)| {
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
                (action, puct)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(action, _)| action)
            .unwrap_or(0)
    }

    /// Get existing child or create a new one by simulating the action.
    fn get_or_create_child(&mut self, parent_id: usize, action: usize) -> usize {
        if let Some(child_id) = self.nodes[parent_id].children[action] {
            return child_id;
        }
        let child_id = self.create_child(parent_id, action);
        self.nodes[parent_id].children[action] = Some(child_id);
        child_id
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
        let board_h = board_hash(&new_board);
        let new_next_next = sample_piece(parent_id as u64, action as u64, board_h);

        let child_state = GameSnapshot {
            board: new_board,
            current: new_current,
            next: new_next,
            next_next: new_next_next,
        };

        let parent_depth = self.nodes[parent_id].depth;
        let valid_mask = compute_valid_mask(&child_state.board, &child_state.current);
        let child = MctsNode {
            visit_count: 0,
            total_value: 0.0,
            prior: self.nodes[parent_id].priors[action],
            priors: [0.0; NUM_ACTIONS],
            logits: [0.0; NUM_ACTIONS],
            children: [None; NUM_ACTIONS],
            expanded: false,
            terminal,
            immediate_reward,
            state: child_state,
            valid_mask,
            depth: parent_depth + 1,
        };

        let child_id = self.nodes.len();
        self.nodes.push(child);
        child_id
    }

    /// Expand a node: run the neural network and set priors + logits for valid actions.
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
        let priors = masked_softmax(&logits_vec, &self.nodes[node_id].valid_mask);

        // Store logits and priors
        let mut stored_logits = [0.0f32; NUM_ACTIONS];
        for (i, logit) in logits_vec.iter().enumerate().take(NUM_ACTIONS) {
            stored_logits[i] = *logit;
        }
        self.nodes[node_id].logits = stored_logits;
        self.nodes[node_id].priors = priors;

        // Update priors of already-created children
        for (action, &prior) in priors.iter().enumerate() {
            if let Some(child_id) = self.nodes[node_id].children[action] {
                self.nodes[child_id].prior = prior;
            }
        }

        self.nodes[node_id].expanded = true;

        v
    }

    /// Get Q values (average cumulative reward) for root's direct children.
    pub fn root_q_values(&self) -> [f32; NUM_ACTIONS] {
        let root = &self.nodes[self.root];
        std::array::from_fn(|action| {
            root.children[action]
                .map(|child_id| &self.nodes[child_id])
                .filter(|child| child.visit_count > 0)
                .map(|child| child.total_value / child.visit_count as f32)
                .unwrap_or(0.0)
        })
    }

    /// Run Sequential Halving over the considered actions, then spend remaining budget.
    /// Returns the final completed Q-values to avoid redundant recomputation.
    fn sequential_halving(
        &mut self,
        considered: &mut Vec<usize>,
        scores: &mut [f32; NUM_ACTIONS],
        gumbels: &[f32; NUM_ACTIONS],
        remaining_budget: usize,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
        config: &MctsConfig,
    ) -> [f32; NUM_ACTIONS] {
        let root_logits = self.nodes[self.root].logits;
        let mask = self.nodes[self.root].valid_mask;

        if remaining_budget == 0 {
            return compute_completed_q(self, &mask);
        }

        let num_phases = {
            let mut phases = 0u32;
            let mut n = considered.len();
            while n > 1 {
                n = (n + 1) / 2;
                phases += 1;
            }
            phases.max(1) as usize
        };

        let mut budget_used = 0usize;

        for phase in 0..num_phases {
            if considered.len() <= 1 {
                break;
            }

            let n_actions = considered.len();
            let budget_remaining = remaining_budget.saturating_sub(budget_used);
            let phases_left = num_phases - phase;
            let sims_per_action = (budget_remaining / (phases_left * n_actions)).max(1);

            budget_used += self.run_simulations(considered, budget_used, remaining_budget, sims_per_action, model, device, config.c_puct);

            let q_completed = compute_completed_q(self, &mask);
            let sigma_bar = compute_sigma_bar(self, &q_completed, &mask, considered, config.c_visit);

            for &a in considered.iter() {
                scores[a] = gumbels[a] + root_logits[a] + sigma_bar[a];
            }

            let keep = (n_actions + 1) / 2;
            considered.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
            considered.truncate(keep);
        }

        // Spend remaining budget on surviving action(s)
        self.run_simulations(considered, budget_used, remaining_budget, usize::MAX, model, device, config.c_puct);

        compute_completed_q(self, &mask)
    }

    /// Run simulations on the considered actions, up to `sims_per_action` each,
    /// respecting the total budget. Returns the number of simulations run.
    fn run_simulations(
        &mut self,
        considered: &[usize],
        budget_used: usize,
        total_budget: usize,
        sims_per_action: usize,
        model: &PuyoNet<InferBackend>,
        device: &<InferBackend as Backend>::Device,
        c_puct: f32,
    ) -> usize {
        let mut count = 0usize;
        for &a in considered {
            for _ in 0..sims_per_action {
                if budget_used + count >= total_budget {
                    return count;
                }
                self.simulate_from_root_action(a, model, device, c_puct);
                count += 1;
            }
        }
        count
    }
}

/// Compute masked softmax over logits.
fn masked_softmax(logits: &[f32], mask: &[bool; NUM_ACTIONS]) -> [f32; NUM_ACTIONS] {
    // Find max for numerical stability
    let max_logit = (0..NUM_ACTIONS)
        .filter(|&i| mask[i])
        .filter_map(|i| logits.get(i).copied())
        .fold(f32::NEG_INFINITY, f32::max);

    if max_logit == f32::NEG_INFINITY {
        return [0.0f32; NUM_ACTIONS]; // No valid actions
    }

    let mut result: [f32; NUM_ACTIONS] = std::array::from_fn(|i| {
        if mask[i] {
            (logits.get(i).copied().unwrap_or(f32::NEG_INFINITY) - max_logit).exp()
        } else {
            0.0
        }
    });
    let sum: f32 = result.iter().sum();
    if sum > 0.0 {
        result.iter_mut().for_each(|r| *r /= sum);
    }

    result
}

/// Simple xorshift64-based PRNG returning f64 in [0, 1).
fn xorshift64_f64(state: &mut u64, mix: u64) -> f64 {
    let mut s = (*state).wrapping_add(mix.wrapping_mul(2654435761));
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // Guard against zero state (xorshift gets stuck at 0)
    if s == 0 {
        s = 0x5a17a453cc79b7d1;
    }
    *state = s;
    (s >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Sample from the standard Gumbel(0,1) distribution: g = -log(-log(u)).
fn sample_gumbel(state: &mut u64, mix: u64) -> f32 {
    let u = xorshift64_f64(state, mix).clamp(1e-20, 1.0 - 1e-10);
    -((-(u.ln())).ln()) as f32
}

/// Compute a simple FNV-1a hash of the board state.
pub fn board_hash(board: &Board) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for col in 0..COLS {
        for row in 0..ROWS {
            h ^= board.columns[col][row] as u8 as u64;
            h = h.wrapping_mul(0x00000100000001B3);
        }
    }
    h
}

/// Sample a random piece deterministically from node_id, action, and board hash.
fn sample_piece(seed1: u64, seed2: u64, seed3: u64) -> Piece {
    let mut x = seed1
        .wrapping_mul(6364136223846793005)
        .wrapping_add(seed2)
        .wrapping_add(1)
        .wrapping_add(seed3.wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);

    let axis = ((x % NUM_COLORS as u64) as u8) + 1;
    // Re-mix for independent satellite color
    x = (x ^ (x >> 30)).wrapping_mul(0x517cc1b727220a95);
    x = x ^ (x >> 27);
    let sat = ((x % NUM_COLORS as u64) as u8) + 1;
    Piece::new(
        puyo_core::board::PuyoColor::from_u8(axis),
        puyo_core::board::PuyoColor::from_u8(sat),
    )
}

/// Compute completed Q-values for all valid root actions.
/// Visited actions use actual Q from tree; unvisited actions use root value estimate.
fn compute_completed_q(
    tree: &MctsTree,
    mask: &[bool; NUM_ACTIONS],
) -> [f32; NUM_ACTIONS] {
    let root = &tree.nodes[tree.root];
    std::array::from_fn(|a| {
        if !mask[a] {
            return 0.0;
        }
        match root.children[a] {
            Some(child_id) if tree.nodes[child_id].visit_count > 0 => {
                tree.nodes[child_id].total_value / tree.nodes[child_id].visit_count as f32
            }
            _ => tree.root_value,
        }
    })
}

/// Compute the improved policy target from logits and completed Q-values.
/// π_improved(a) ∝ π(a) · exp(advantage(a) · c_visit)
fn compute_improved_policy(
    logits: &[f32; NUM_ACTIONS],
    q_completed: &[f32; NUM_ACTIONS],
    mask: &[bool; NUM_ACTIONS],
    c_visit: f32,
    c_scale: f32,
) -> [f32; NUM_ACTIONS] {
    // Compute V_mixed: prior-weighted sum of completed Q-values
    let priors = masked_softmax(logits, mask);
    let v_mixed: f32 = (0..NUM_ACTIONS)
        .filter(|&a| mask[a])
        .map(|a| priors[a] * q_completed[a])
        .sum();

    // Compute improved logits: logit(a) + advantage(a) * c_visit / c_scale
    let mut improved_logits = [f32::NEG_INFINITY; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        if mask[a] {
            let advantage = q_completed[a] - v_mixed;
            improved_logits[a] = logits[a] + advantage * c_visit / c_scale;
        }
    }

    masked_softmax(&improved_logits, mask)
}

/// Compute sigma_bar for Sequential Halving score updates.
/// sigma_bar(a) = (c_visit + N_max) * q_normalized(a)
/// where q_normalized is min-max normalized completed Q-value.
fn compute_sigma_bar(
    tree: &MctsTree,
    q_completed: &[f32; NUM_ACTIONS],
    mask: &[bool; NUM_ACTIONS],
    considered: &[usize],
    c_visit: f32,
) -> [f32; NUM_ACTIONS] {
    // Find max visit count among root children
    let root = &tree.nodes[tree.root];
    let n_max: f32 = considered.iter()
        .filter_map(|&a| root.children[a].map(|cid| tree.nodes[cid].visit_count as f32))
        .fold(0.0f32, f32::max);

    // Min-max normalize completed Q-values
    let mut min_q = f32::INFINITY;
    let mut max_q = f32::NEG_INFINITY;
    for &a in considered {
        if mask[a] {
            min_q = min_q.min(q_completed[a]);
            max_q = max_q.max(q_completed[a]);
        }
    }
    let q_range = max_q - min_q;

    std::array::from_fn(|a| {
        if !mask[a] {
            return 0.0;
        }
        let q_norm = if q_range > f32::EPSILON {
            (q_completed[a] - min_q) / q_range
        } else {
            0.5
        };
        (c_visit + n_max) * q_norm
    })
}

/// Run Gumbel MCTS search using Sequential Halving with Gumbel noise.
///
/// Returns (improved_policy, q_values).
/// `seed` is used for deterministic Gumbel noise sampling.
pub fn mcts_search(
    board: &Board,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
    model: &PuyoNet<InferBackend>,
    device: &<InferBackend as Backend>::Device,
    config: &MctsConfig,
    seed: u64,
) -> ([f32; NUM_ACTIONS], [f32; NUM_ACTIONS]) {
    let mut tree = MctsTree::new(board, current, next, next_next, config.gamma);

    // 1. Expand root (1 NN evaluation)
    if config.num_simulations == 0 || tree.nodes[tree.root].terminal {
        let mask = tree.nodes[tree.root].valid_mask;
        return (masked_softmax(&[0.0f32; NUM_ACTIONS], &mask), [0.0; NUM_ACTIONS]);
    }
    tree.expand_root(model, device);

    let mask = tree.nodes[tree.root].valid_mask;
    let root_logits = tree.nodes[tree.root].logits;

    // Collect valid actions
    let valid_actions: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| mask[a]).collect();
    if valid_actions.is_empty() {
        return ([0.0; NUM_ACTIONS], [0.0; NUM_ACTIONS]);
    }
    if valid_actions.len() == 1 {
        let mut policy = [0.0f32; NUM_ACTIONS];
        policy[valid_actions[0]] = 1.0;
        return (policy, tree.root_q_values());
    }

    // 2. Sample Gumbel noise and compute initial scores: g(a) + logit(a)
    let mut rng_state = seed.wrapping_add(0xdeadbeef);
    let mut gumbels = [0.0f32; NUM_ACTIONS];
    let mut scores = [f32::NEG_INFINITY; NUM_ACTIONS];
    for &a in &valid_actions {
        let g = sample_gumbel(&mut rng_state, a as u64);
        gumbels[a] = g;
        scores[a] = g + root_logits[a];
    }

    // 3. Select top-m actions by initial score
    let m = config.m.min(valid_actions.len());
    let mut considered = valid_actions;
    considered.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    considered.truncate(m);

    // 4. Sequential Halving + spend remaining budget
    let remaining_budget = config.num_simulations.saturating_sub(1); // root expansion used 1
    let q_completed = tree.sequential_halving(
        &mut considered,
        &mut scores,
        &gumbels,
        remaining_budget,
        model,
        device,
        config,
    );

    // 5. Compute improved policy target
    let improved_policy = compute_improved_policy(&root_logits, &q_completed, &mask, config.c_visit, config.c_scale);

    (improved_policy, tree.root_q_values())
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
        let p1 = sample_piece(42, 7, 0);
        let p2 = sample_piece(42, 7, 0);
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

    #[test]
    fn test_sample_gumbel_finite() {
        let mut state = 12345u64;
        for i in 0..100 {
            let g = sample_gumbel(&mut state, i);
            assert!(g.is_finite(), "Gumbel sample should be finite");
        }
    }

    #[test]
    fn test_compute_improved_policy_normalized() {
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let q = [10.0, 20.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut mask = [false; 24];
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;

        let policy = compute_improved_policy(&logits, &q, &mask, 50.0, 1.0);

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Improved policy should sum to 1.0, got {}", sum);
        // Action 1 has highest Q, should have highest improved probability
        assert!(policy[1] > policy[0]);
        assert!(policy[1] > policy[2]);
    }
}
