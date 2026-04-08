/// Mirror of WasmGame interface for type safety.
export interface WasmGame {
  board_cols(): number;
  board_rows(): number;
  board_visible_rows(): number;
  num_colors(): number;
  get_board(): Uint8Array;
  get_current_piece(): Uint8Array;
  get_next_piece(): Uint8Array;
  get_next_next_piece(): Uint8Array;
  get_score(): number;
  get_max_chain(): number;
  get_phase(): number;
  get_total_pieces(): number;
  move_left(): boolean;
  move_right(): boolean;
  rotate_cw(): boolean;
  rotate_ccw(): boolean;
  hard_drop(): number;
  soft_drop(): boolean;
  ai_best_move(): Uint8Array;
  ai_play_move(): number;
  apply_placement_direct(col: number, ori: number): number;
  enumerate_placements(): Uint8Array;
  load_nn_model(model_bytes: Uint8Array): void;
  load_nn_model_with_mcts(model_bytes: Uint8Array, num_simulations: number): void;
  set_mcts_simulations(num_simulations: number): void;
  use_heuristic(): void;
  restart(): void;
  free(): void;
}

export interface WasmModule {
  WasmGame: {
    new(cols: number, rows: number, num_colors: number): WasmGame;
  };
}
