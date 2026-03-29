export const CELL_SIZE = 64; // pixels per cell

// Board configuration (populated from WASM at startup)
export interface BoardConfig {
  cols: number;
  rows: number;
  visibleRows: number;
  numColors: number;
  boardWidth: number;   // cols * CELL_SIZE
  boardHeight: number;  // rows * CELL_SIZE
}

// Puyo colors (matching Rust PuyoColor repr)
export const COLOR_EMPTY = 0;
export const COLOR_RED = 1;
export const COLOR_GREEN = 2;
export const COLOR_BLUE = 3;
export const COLOR_YELLOW = 4;

// Rendering colors for each PuyoColor value
export const PUYO_COLORS: Record<number, string> = {
  [COLOR_RED]: "#e94560",
  [COLOR_GREEN]: "#4ade80",
  [COLOR_BLUE]: "#60a5fa",
  [COLOR_YELLOW]: "#fbbf24",
};

// Lighter shades for the "eye" highlight
export const PUYO_HIGHLIGHT: Record<number, string> = {
  [COLOR_RED]: "#ff7b93",
  [COLOR_GREEN]: "#86efac",
  [COLOR_BLUE]: "#93c5fd",
  [COLOR_YELLOW]: "#fcd34d",
};

// Game phases
export const PHASE_GAME_OVER = 2;

// Orientation offsets (matching Rust)
export const ORIENTATION_OFFSETS: [number, number][] = [
  [0, 1],   // North: satellite above
  [1, 0],   // East: satellite right
  [0, -1],  // South: satellite below
  [-1, 0],  // West: satellite left
];
