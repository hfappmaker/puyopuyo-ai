import {
  COLS,
  VISIBLE_ROWS,
  ROWS,
  CELL_SIZE,
  PUYO_COLORS,
  PUYO_HIGHLIGHT,
  COLOR_EMPTY,
  ORIENTATION_OFFSETS,
} from "./constants";
import type { WasmGame } from "./types";

const PUYO_RADIUS = CELL_SIZE * 0.42;
const EYE_RADIUS = 3;
const EYE_OFFSET_X = 5;
const EYE_OFFSET_Y = -4;

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private nextCtx: CanvasRenderingContext2D;
  private nextNextCtx: CanvasRenderingContext2D;

  constructor(boardCanvas: HTMLCanvasElement, nextCanvas: HTMLCanvasElement, nextNextCanvas: HTMLCanvasElement) {
    this.ctx = boardCanvas.getContext("2d")!;
    this.nextCtx = nextCanvas.getContext("2d")!;
    this.nextNextCtx = nextNextCanvas.getContext("2d")!;
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
    const hiddenRows = ROWS - VISIBLE_ROWS;

    // Clear: hidden area (darker) + visible area
    ctx.fillStyle = "#080812";
    ctx.fillRect(0, 0, COLS * CELL_SIZE, hiddenRows * CELL_SIZE);
    ctx.fillStyle = "#0f0f23";
    ctx.fillRect(0, hiddenRows * CELL_SIZE, COLS * CELL_SIZE, VISIBLE_ROWS * CELL_SIZE);

    // Grid lines
    ctx.strokeStyle = "#1a1a3e";
    ctx.lineWidth = 0.5;
    for (let col = 0; col <= COLS; col++) {
      ctx.beginPath();
      ctx.moveTo(col * CELL_SIZE, 0);
      ctx.lineTo(col * CELL_SIZE, ROWS * CELL_SIZE);
      ctx.stroke();
    }
    for (let row = 0; row <= ROWS; row++) {
      ctx.beginPath();
      ctx.moveTo(0, row * CELL_SIZE);
      ctx.lineTo(COLS * CELL_SIZE, row * CELL_SIZE);
      ctx.stroke();
    }

    // Boundary line between hidden and visible area
    ctx.strokeStyle = "#e94560";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, hiddenRows * CELL_SIZE);
    ctx.lineTo(COLS * CELL_SIZE, hiddenRows * CELL_SIZE);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw puyos (board is column-major, bottom=row0)
    for (let col = 0; col < COLS; col++) {
      for (let row = 0; row < ROWS; row++) {
        const color = board[col * ROWS + row];
        if (color !== COLOR_EMPTY) {
          const x = col * CELL_SIZE + CELL_SIZE / 2;
          const y = (ROWS - 1 - row) * CELL_SIZE + CELL_SIZE / 2;
          // Hidden area puyos are dimmed
          if (row >= VISIBLE_ROWS) {
            ctx.globalAlpha = 0.5;
          }
          this.drawPuyo(ctx, x, y, color, 1.0);
          ctx.globalAlpha = 1.0;
        }
      }
    }
  }

  private drawCurrentPiece(game: WasmGame): void {
    const pieceData = game.get_current_piece();
    if (pieceData.length === 0) return;

    const axisColor = pieceData[0];
    const satColor = pieceData[1];
    const col = pieceData[2];
    const rowInt = pieceData[3];
    const rowFrac = pieceData[4] / 100;
    const orientation = pieceData[5];

    const row = rowInt + rowFrac;
    const [dc, dr] = ORIENTATION_OFFSETS[orientation];

    // Draw axis puyo
    const axisX = col * CELL_SIZE + CELL_SIZE / 2;
    const axisY = (ROWS - 1 - row) * CELL_SIZE + CELL_SIZE / 2;
    this.drawPuyo(this.ctx, axisX, axisY, axisColor, 0.9);

    // Draw satellite puyo
    const satX = (col + dc) * CELL_SIZE + CELL_SIZE / 2;
    const satY = (ROWS - 1 - (row + dr)) * CELL_SIZE + CELL_SIZE / 2;
    this.drawPuyo(this.ctx, satX, satY, satColor, 0.9);

    // Draw ghost (hard drop preview)
    this.drawGhost(game, col, orientation, axisColor, satColor);
  }

  private drawGhost(
    game: WasmGame,
    col: number,
    orientation: number,
    axisColor: number,
    satColor: number
  ): void {
    // Simple ghost: show where piece would land
    const board = game.get_board();
    const [dc, dr] = ORIENTATION_OFFSETS[orientation];

    const getHeight = (c: number): number => {
      for (let r = ROWS - 1; r >= 0; r--) {
        if (board[c * ROWS + r] !== COLOR_EMPTY) return r + 1;
      }
      return 0;
    };

    const satCol = col + dc;
    if (satCol < 0 || satCol >= COLS) return;

    let axisRow: number;
    let satRow: number;

    if (dc === 0) {
      // Vertical: both in same column
      const h = getHeight(col);
      if (dr > 0) {
        // North: axis below, satellite above
        axisRow = h;
        satRow = h + 1;
      } else {
        // South: satellite below, axis above
        satRow = h;
        axisRow = h + 1;
      }
    } else {
      // Horizontal: side by side
      axisRow = getHeight(col);
      satRow = getHeight(satCol);
    }

    const ctx = this.ctx;
    ctx.globalAlpha = 0.25;
    this.drawPuyo(
      ctx,
      col * CELL_SIZE + CELL_SIZE / 2,
      (ROWS - 1 - axisRow) * CELL_SIZE + CELL_SIZE / 2,
      axisColor,
      1.0
    );
    this.drawPuyo(
      ctx,
      satCol * CELL_SIZE + CELL_SIZE / 2,
      (ROWS - 1 - satRow) * CELL_SIZE + CELL_SIZE / 2,
      satColor,
      1.0
    );
    ctx.globalAlpha = 1.0;
  }

  private drawNext(game: WasmGame): void {
    const ctx = this.nextCtx;
    const nextData = game.get_next_piece();
    if (nextData.length < 2) return;

    const axisColor = nextData[0];
    const satColor = nextData[1];

    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, 80, 80);

    // Draw satellite above axis (North orientation)
    this.drawPuyo(ctx, 40, 20, satColor, 1.0);
    this.drawPuyo(ctx, 40, 56, axisColor, 1.0);
  }

  private drawNextNext(game: WasmGame): void {
    const ctx = this.nextNextCtx;
    const data = game.get_next_next_piece();
    if (data.length < 2) return;

    const axisColor = data[0];
    const satColor = data[1];

    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, 60, 60);

    // Draw satellite above axis (North orientation), scaled down
    this.drawPuyo(ctx, 30, 15, satColor, 0.75);
    this.drawPuyo(ctx, 30, 42, axisColor, 0.75);
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
    ctx.arc(x - EYE_OFFSET_X + 1, y + EYE_OFFSET_Y, 1.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x + EYE_OFFSET_X + 1, y + EYE_OFFSET_Y, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
}
