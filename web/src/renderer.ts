import {
  CELL_SIZE,
  PUYO_COLORS,
  PUYO_HIGHLIGHT,
  COLOR_EMPTY,
  ORIENTATION_OFFSETS,
} from "./constants";
import type { BoardConfig } from "./constants";
import type { WasmGame } from "./types";

const PUYO_RADIUS = CELL_SIZE * 0.42;
const EYE_RADIUS = 3;
const EYE_OFFSET_X = 5;
const EYE_OFFSET_Y = -4;
const PUPIL_RADIUS = 1.5;
const PUPIL_OFFSET_X = 1;

// Transparency levels
const HIDDEN_AREA_ALPHA = 0.5;
const CURRENT_PIECE_ALPHA = 0.9;
const GHOST_ALPHA = 0.25;
const PLACEMENT_PREVIEW_ALPHA = 0.4;

// Board background colors
const BG_HIDDEN = "#080812";
const BG_VISIBLE = "#0f0f23";
const GRID_COLOR = "#1a1a3e";
const GRID_LINE_WIDTH = 0.5;
const BOUNDARY_COLOR = "#e94560";
const BOUNDARY_LINE_WIDTH = 2;
const BOUNDARY_DASH = [6, 4];

// Next piece preview sizes
const NEXT_PREVIEW_SIZE = 80;
const NEXT_NEXT_PREVIEW_SIZE = 60;
const NEXT_NEXT_SCALE = 0.75;

// Preview canvas background
const PREVIEW_BG = "#16213e";

/** ピースデータ配列の各フィールド */
interface PieceData {
  axisColor: number;
  satColor: number;
  col: number;
  row: number;
  orientation: number;
}

function parsePieceData(data: Uint8Array): PieceData | null {
  if (data.length < 6) return null;
  return {
    axisColor: data[0],
    satColor: data[1],
    col: data[2],
    row: data[3] + data[4] / 100,
    orientation: data[5],
  };
}

/** 盤面座標をキャンバスピクセル座標に変換 */
function cellCenterX(col: number): number {
  return col * CELL_SIZE + CELL_SIZE / 2;
}

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private nextCtx: CanvasRenderingContext2D;
  private nextNextCtx: CanvasRenderingContext2D;
  private config: BoardConfig;

  constructor(boardCanvas: HTMLCanvasElement, nextCanvas: HTMLCanvasElement, nextNextCanvas: HTMLCanvasElement, config: BoardConfig) {
    this.ctx = boardCanvas.getContext("2d")!;
    this.nextCtx = nextCanvas.getContext("2d")!;
    this.nextNextCtx = nextNextCanvas.getContext("2d")!;
    this.config = config;
  }

  private cellCenterY(row: number): number {
    return (this.config.rows - 1 - row) * CELL_SIZE + CELL_SIZE / 2;
  }

  render(game: WasmGame): void {
    this.drawBoard(game);
    this.drawCurrentPiece(game);
    this.drawNext(game);
    this.drawNextNext(game);
  }

  private drawBoard(game: WasmGame): void {
    const ctx = this.ctx;
    const board = game.get_board();
    const { cols, rows, visibleRows } = this.config;
    const hiddenRows = rows - visibleRows;

    // Clear: hidden area (darker) + visible area
    ctx.fillStyle = BG_HIDDEN;
    ctx.fillRect(0, 0, cols * CELL_SIZE, hiddenRows * CELL_SIZE);
    ctx.fillStyle = BG_VISIBLE;
    ctx.fillRect(0, hiddenRows * CELL_SIZE, cols * CELL_SIZE, visibleRows * CELL_SIZE);

    // Grid lines
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = GRID_LINE_WIDTH;
    for (let col = 0; col <= cols; col++) {
      ctx.beginPath();
      ctx.moveTo(col * CELL_SIZE, 0);
      ctx.lineTo(col * CELL_SIZE, rows * CELL_SIZE);
      ctx.stroke();
    }
    for (let row = 0; row <= rows; row++) {
      ctx.beginPath();
      ctx.moveTo(0, row * CELL_SIZE);
      ctx.lineTo(cols * CELL_SIZE, row * CELL_SIZE);
      ctx.stroke();
    }

    // Boundary line between hidden and visible area
    ctx.strokeStyle = BOUNDARY_COLOR;
    ctx.lineWidth = BOUNDARY_LINE_WIDTH;
    ctx.setLineDash(BOUNDARY_DASH);
    ctx.beginPath();
    ctx.moveTo(0, hiddenRows * CELL_SIZE);
    ctx.lineTo(cols * CELL_SIZE, hiddenRows * CELL_SIZE);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw puyos (board is column-major, bottom=row0)
    for (let col = 0; col < cols; col++) {
      for (let row = 0; row < rows; row++) {
        const color = board[col * rows + row];
        if (color !== COLOR_EMPTY) {
          if (row >= visibleRows) {
            ctx.globalAlpha = HIDDEN_AREA_ALPHA;
          }
          this.drawPuyo(ctx, cellCenterX(col), this.cellCenterY(row), color, 1.0);
          ctx.globalAlpha = 1.0;
        }
      }
    }
  }

  private drawCurrentPiece(game: WasmGame): void {
    const piece = parsePieceData(game.get_current_piece());
    if (!piece) return;

    const [dc, dr] = ORIENTATION_OFFSETS[piece.orientation];

    this.drawPuyo(this.ctx, cellCenterX(piece.col), this.cellCenterY(piece.row), piece.axisColor, CURRENT_PIECE_ALPHA);
    this.drawPuyo(this.ctx, cellCenterX(piece.col + dc), this.cellCenterY(piece.row + dr), piece.satColor, CURRENT_PIECE_ALPHA);

    this.drawGhost(game, piece);
  }

  private drawGhost(game: WasmGame, piece: PieceData): void {
    const board = game.get_board();
    const [dc, dr] = ORIENTATION_OFFSETS[piece.orientation];
    const { cols, rows } = this.config;

    const getHeight = (c: number): number => {
      for (let r = 0; r < rows; r++) {
        if (board[c * rows + r] === COLOR_EMPTY) return r;
      }
      return rows;
    };

    const satCol = piece.col + dc;
    if (satCol < 0 || satCol >= cols) return;

    let axisRow: number;
    let satRow: number;

    if (dc === 0) {
      const h = getHeight(piece.col);
      if (dr > 0) {
        axisRow = h;
        satRow = h + 1;
      } else {
        satRow = h;
        axisRow = h + 1;
      }
    } else {
      axisRow = getHeight(piece.col);
      satRow = getHeight(satCol);
    }

    const ctx = this.ctx;
    ctx.globalAlpha = GHOST_ALPHA;
    this.drawPuyo(ctx, cellCenterX(piece.col), this.cellCenterY(axisRow), piece.axisColor, 1.0);
    this.drawPuyo(ctx, cellCenterX(satCol), this.cellCenterY(satRow), piece.satColor, 1.0);
    ctx.globalAlpha = 1.0;
  }

  private drawNext(game: WasmGame): void {
    const ctx = this.nextCtx;
    const nextData = game.get_next_piece();
    if (nextData.length < 2) return;

    ctx.fillStyle = PREVIEW_BG;
    ctx.fillRect(0, 0, NEXT_PREVIEW_SIZE, NEXT_PREVIEW_SIZE);

    const cx = NEXT_PREVIEW_SIZE / 2;
    this.drawPuyo(ctx, cx, 20, nextData[1], 1.0);
    this.drawPuyo(ctx, cx, 56, nextData[0], 1.0);
  }

  private drawNextNext(game: WasmGame): void {
    const ctx = this.nextNextCtx;
    const data = game.get_next_next_piece();
    if (data.length < 2) return;

    ctx.fillStyle = PREVIEW_BG;
    ctx.fillRect(0, 0, NEXT_NEXT_PREVIEW_SIZE, NEXT_NEXT_PREVIEW_SIZE);

    const cx = NEXT_NEXT_PREVIEW_SIZE / 2;
    this.drawPuyo(ctx, cx, 15, data[1], NEXT_NEXT_SCALE);
    this.drawPuyo(ctx, cx, 42, data[0], NEXT_NEXT_SCALE);
  }

  renderWithPlacementPreview(game: WasmGame, col: number, orientation: number): void {
    this.render(game);
    this.drawPlacementPreview(game, col, orientation);
  }

  private drawPlacementPreview(game: WasmGame, col: number, orientation: number): void {
    const piece = parsePieceData(game.get_current_piece());
    if (!piece) return;

    const board = game.get_board();
    const [dc, dr] = ORIENTATION_OFFSETS[orientation];
    const { cols, rows } = this.config;

    const getHeight = (c: number): number => {
      for (let r = 0; r < rows; r++) {
        if (board[c * rows + r] === COLOR_EMPTY) return r;
      }
      return rows;
    };

    const satCol = col + dc;
    if (satCol < 0 || satCol >= cols) return;

    let axisRow: number;
    let satRow: number;

    if (dc === 0) {
      const h = getHeight(col);
      if (dr > 0) {
        axisRow = h;
        satRow = h + 1;
      } else {
        satRow = h;
        axisRow = h + 1;
      }
    } else {
      axisRow = getHeight(col);
      satRow = getHeight(satCol);
    }

    const ctx = this.ctx;
    ctx.globalAlpha = PLACEMENT_PREVIEW_ALPHA;
    this.drawPuyo(ctx, cellCenterX(col), this.cellCenterY(axisRow), piece.axisColor, 1.0);
    this.drawPuyo(ctx, cellCenterX(satCol), this.cellCenterY(satRow), piece.satColor, 1.0);
    ctx.globalAlpha = 1.0;

    // Draw outline to distinguish from normal ghost
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cellCenterX(col), this.cellCenterY(axisRow), PUYO_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cellCenterX(satCol), this.cellCenterY(satRow), PUYO_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
  }

  private drawPuyo(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    color: number,
    scale: number
  ): void {
    const fillColor = PUYO_COLORS[color];
    const highlight = PUYO_HIGHLIGHT[color];
    if (!fillColor) return;

    const r = PUYO_RADIUS * scale;

    // Main circle
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Highlight gradient
    const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
    grad.addColorStop(0, highlight + "88");
    grad.addColorStop(1, "transparent");
    ctx.fillStyle = grad;
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(x - EYE_OFFSET_X, y + EYE_OFFSET_Y, EYE_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + EYE_OFFSET_X, y + EYE_OFFSET_Y, EYE_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    // Pupils
    ctx.fillStyle = "#000";
    ctx.beginPath();
    ctx.arc(x - EYE_OFFSET_X + PUPIL_OFFSET_X, y + EYE_OFFSET_Y, PUPIL_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + EYE_OFFSET_X + PUPIL_OFFSET_X, y + EYE_OFFSET_Y, PUPIL_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  }
}
