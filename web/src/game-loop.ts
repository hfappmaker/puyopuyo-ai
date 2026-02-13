import type { WasmGame } from "./types";
import { Renderer } from "./renderer";
import type { InputAction } from "./input";
import { InputHandler } from "./input";
import { UI } from "./ui";
import { PHASE_GAME_OVER, DROP_SPEED, AUTO_PLAY_DELAY } from "./constants";

const enum AnimState {
  IDLE,
  DROPPING,
}

export class GameLoop {
  private game: WasmGame;
  private renderer: Renderer;
  private input: InputHandler;
  private ui: UI;
  private createGame: (seed: bigint) => WasmGame;

  private animState: AnimState = AnimState.IDLE;
  private autoPlay: boolean = false;
  private gameOverShown: boolean = false;

  private rafId: number = 0;
  private lastTimestamp: number = 0;
  private autoPlayDelayTimer: number = 0;

  constructor(
    game: WasmGame,
    renderer: Renderer,
    ui: UI,
    createGame: (seed: bigint) => WasmGame
  ) {
    this.game = game;
    this.renderer = renderer;
    this.input = new InputHandler();
    this.ui = ui;
    this.createGame = createGame;

    this.input.onAction = (action) => this.handleAction(action);
  }

  start(): void {
    this.ui.update(this.game);
    this.renderer.render(this.game);
  }

  private handleAction(action: InputAction): void {
    switch (action) {
      case "next_move":
        this.triggerNextMove();
        break;
      case "toggle_auto":
        this.toggleAutoPlay();
        break;
      case "restart":
        this.restart();
        break;
    }
  }

  private triggerNextMove(): void {
    if (this.animState !== AnimState.IDLE) return;

    if (this.game.get_phase() === PHASE_GAME_OVER) {
      if (!this.gameOverShown) {
        this.ui.showGameOver(this.game);
        this.gameOverShown = true;
      }
      this.stopAutoPlay();
      return;
    }

    const bestMove = this.game.ai_best_move();
    if (bestMove.length === 0) return;

    const targetCol = bestMove[0];
    const targetOrientation = bestMove[1];

    this.positionPiece(targetCol, targetOrientation);

    this.animState = AnimState.DROPPING;
    this.lastTimestamp = 0;
    this.rafId = requestAnimationFrame((ts) => this.animationLoop(ts));
  }

  private positionPiece(targetCol: number, targetOrientation: number): void {
    const pieceData = this.game.get_current_piece();
    if (pieceData.length === 0) return;

    const currentOrientation = pieceData[5];

    // Rotate to target orientation (shortest path)
    const diff = (targetOrientation - currentOrientation + 4) % 4;
    if (diff === 1) {
      this.game.rotate_cw();
    } else if (diff === 2) {
      this.game.rotate_cw();
      this.game.rotate_cw();
    } else if (diff === 3) {
      this.game.rotate_ccw();
    }

    // Re-read col after rotation (wall kicks may have shifted it)
    const afterRotate = this.game.get_current_piece();
    if (afterRotate.length === 0) return;
    const colAfterRotate = afterRotate[2];

    // Move to target column
    const colDiff = targetCol - colAfterRotate;
    if (colDiff > 0) {
      for (let i = 0; i < colDiff; i++) this.game.move_right();
    } else if (colDiff < 0) {
      for (let i = 0; i < -colDiff; i++) this.game.move_left();
    }

    this.renderer.render(this.game);
  }

  private animationLoop(timestamp: number): void {
    if (this.lastTimestamp === 0) {
      this.lastTimestamp = timestamp;
      this.renderer.render(this.game);
      this.rafId = requestAnimationFrame((ts) => this.animationLoop(ts));
      return;
    }

    const deltaMs = Math.min(timestamp - this.lastTimestamp, 100);
    this.lastTimestamp = timestamp;

    const gravity = DROP_SPEED * (deltaMs / 1000);

    const totalBefore = this.game.get_total_pieces();
    this.game.tick(gravity);
    this.renderer.render(this.game);

    const totalAfter = this.game.get_total_pieces();
    if (totalAfter > totalBefore || this.game.get_phase() === PHASE_GAME_OVER) {
      this.onDropComplete();
      return;
    }

    this.rafId = requestAnimationFrame((ts) => this.animationLoop(ts));
  }

  private onDropComplete(): void {
    this.animState = AnimState.IDLE;
    this.rafId = 0;
    this.lastTimestamp = 0;

    this.ui.update(this.game);
    this.renderer.render(this.game);

    if (this.game.get_phase() === PHASE_GAME_OVER) {
      this.ui.showGameOver(this.game);
      this.gameOverShown = true;
      this.stopAutoPlay();
      return;
    }

    if (this.autoPlay) {
      this.autoPlayDelayTimer = window.setTimeout(() => {
        this.autoPlayDelayTimer = 0;
        this.triggerNextMove();
      }, AUTO_PLAY_DELAY);
    }
  }

  private toggleAutoPlay(): void {
    if (this.autoPlay) {
      this.stopAutoPlay();
    } else {
      this.startAutoPlay();
    }
  }

  private startAutoPlay(): void {
    if (this.autoPlay) return;
    this.autoPlay = true;
    this.ui.setAutoPlayStatus(true);

    if (this.animState === AnimState.IDLE) {
      this.triggerNextMove();
    }
  }

  private stopAutoPlay(): void {
    if (!this.autoPlay) return;
    this.autoPlay = false;
    this.ui.setAutoPlayStatus(false);

    if (this.autoPlayDelayTimer) {
      window.clearTimeout(this.autoPlayDelayTimer);
      this.autoPlayDelayTimer = 0;
    }
  }

  private restart(): void {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
    this.lastTimestamp = 0;
    this.animState = AnimState.IDLE;

    this.stopAutoPlay();

    const seed = BigInt(Date.now());
    this.game.free();
    this.game = this.createGame(seed);
    this.gameOverShown = false;
    this.ui.hideGameOver();
    this.ui.update(this.game);
    this.renderer.render(this.game);
  }
}
