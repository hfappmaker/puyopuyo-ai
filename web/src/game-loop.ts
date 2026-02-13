import type { WasmGame } from "./types";
import { Renderer } from "./renderer";
import { InputHandler } from "./input";
import { UI } from "./ui";
import { PHASE_GAME_OVER } from "./constants";

const AUTO_PLAY_INTERVAL = 500; // ms between auto-steps

export class GameLoop {
  private game: WasmGame;
  private renderer: Renderer;
  private input: InputHandler;
  private ui: UI;
  private autoPlay: boolean = false;
  private autoPlayTimer: number = 0;
  private gameOverShown: boolean = false;
  private createGame: (seed: bigint) => WasmGame;

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

  private handleAction(action: string): void {
    switch (action) {
      case "step":
        this.step();
        break;
      case "toggle_auto":
        this.toggleAutoPlay();
        break;
      case "restart":
        this.restart();
        break;
    }
  }

  private step(): void {
    if (this.game.get_phase() === PHASE_GAME_OVER) {
      if (!this.gameOverShown) {
        this.ui.showGameOver(this.game);
        this.gameOverShown = true;
      }
      this.stopAutoPlay();
      return;
    }

    this.game.ai_play_move();

    if (this.game.get_phase() === PHASE_GAME_OVER) {
      this.ui.showGameOver(this.game);
      this.gameOverShown = true;
      this.stopAutoPlay();
    }

    this.ui.update(this.game);
    this.renderer.render(this.game);
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
    this.autoPlayTimer = window.setInterval(() => this.step(), AUTO_PLAY_INTERVAL);
  }

  private stopAutoPlay(): void {
    if (!this.autoPlay) return;
    this.autoPlay = false;
    this.ui.setAutoPlayStatus(false);
    window.clearInterval(this.autoPlayTimer);
    this.autoPlayTimer = 0;
  }

  private restart(): void {
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
