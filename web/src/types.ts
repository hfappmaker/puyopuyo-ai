/// Mirror of WasmGame interface for type safety.
export interface WasmGame {
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
  load_nn_model(model_bytes: Uint8Array, mean: number, std_dev: number): void;
  use_heuristic(): void;
  restart(seed: bigint): void;
  free(): void;
}

export interface WasmModule {
  WasmGame: {
    new(seed: bigint): WasmGame;
  };
}
