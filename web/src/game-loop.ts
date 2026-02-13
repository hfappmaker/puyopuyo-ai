import type { WasmGame } from "./types";
import { Renderer } from "./renderer";
import type { InputAction } from "./input";
import { InputHandler } from "./input";
import { UI } from "./ui";
import { PHASE_GAME_OVER } from "./constants";

export class GameLoop {
  private game: WasmGame;
  private renderer: Renderer;
  private input: InputHandler;
  private ui: UI;
  private createGame: (seed: bigint) => WasmGame;

  private gameOver: boolean = false;
  private aiPreviewing: boolean = false;

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
    if (this.aiPreviewing && action !== "ai_move" && action !== "restart") {
      return;
    }
    switch (action) {
      case "move_left":
        this.handleMoveLeft();
        break;
      case "move_right":
        this.handleMoveRight();
        break;
      case "rotate":
        this.handleRotate();
        break;
      case "soft_drop":
        this.handleSoftDrop();
        break;
      case "hard_drop":
        this.handleHardDrop();
        break;
      case "ai_move":
        this.handleAiMove();
        break;
      case "restart":
        this.restart();
        break;
    }
  }

  private handleMoveLeft(): void {
    if (this.gameOver) return;
    this.game.move_left();
    this.renderer.render(this.game);
  }

  private handleMoveRight(): void {
    if (this.gameOver) return;
    this.game.move_right();
    this.renderer.render(this.game);
  }

  private handleRotate(): void {
    if (this.gameOver) return;
    this.game.rotate_cw();
    this.renderer.render(this.game);
  }

  private handleSoftDrop(): void {
    if (this.gameOver) return;
    const moved = this.game.soft_drop();
    if (!moved) {
      this.game.hard_drop();
      this.onPieceLanded();
      return;
    }
    this.renderer.render(this.game);
  }

  private handleHardDrop(): void {
    if (this.gameOver) return;
    this.game.hard_drop();
    this.onPieceLanded();
  }

  private handleAiMove(): void {
    if (this.gameOver) return;

    if (this.aiPreviewing) {
      this.game.hard_drop();
      this.aiPreviewing = false;
      this.onPieceLanded();
      return;
    }

    const bestMove = this.game.ai_best_move();
    if (bestMove.length === 0) return;

    const targetCol = bestMove[0];
    const targetOri = bestMove[1];

    const currentPiece = this.game.get_current_piece();
    if (currentPiece.length === 0) return;
    const currentOri = currentPiece[5];

    this.rotateTo(currentOri, targetOri);

    const afterRotate = this.game.get_current_piece();
    const currentCol = afterRotate[2];
    this.moveTo(currentCol, targetCol);

    this.renderer.render(this.game);
    this.aiPreviewing = true;
  }

  private rotateTo(currentOri: number, targetOri: number): void {
    if (currentOri === targetOri) return;
    const cwSteps = (targetOri - currentOri + 4) % 4;
    const ccwSteps = (currentOri - targetOri + 4) % 4;
    if (cwSteps <= ccwSteps) {
      for (let i = 0; i < cwSteps; i++) this.game.rotate_cw();
    } else {
      for (let i = 0; i < ccwSteps; i++) this.game.rotate_ccw();
    }
  }

  private moveTo(currentCol: number, targetCol: number): void {
    while (currentCol < targetCol) {
      this.game.move_right();
      currentCol++;
    }
    while (currentCol > targetCol) {
      this.game.move_left();
      currentCol--;
    }
  }

  private onPieceLanded(): void {
    this.ui.update(this.game);
    this.renderer.render(this.game);

    if (this.game.get_phase() === PHASE_GAME_OVER) {
      this.gameOver = true;
      this.ui.showGameOver(this.game);
      return;
    }
  }

  private restart(): void {
    this.aiPreviewing = false;

    const seed = BigInt(Date.now());
    this.game.free();
    this.game = this.createGame(seed);
    this.gameOver = false;
    this.ui.hideGameOver();
    this.ui.update(this.game);
    this.renderer.render(this.game);
  }
}
