import type { WasmGame } from "./types";
import { Renderer } from "./renderer";
import { InputHandler } from "./input";
import { UI } from "./ui";
import {
  GRAVITY_NORMAL,
  GRAVITY_FAST,
  AI_PLAY_INTERVAL,
  PHASE_FALLING,
  PHASE_GAME_OVER,
} from "./constants";

export class GameLoop {
  private game: WasmGame;
  private renderer: Renderer;
  private input: InputHandler;
  private ui: UI;
  private aiMode: boolean = false;
  private lastAiTime: number = 0;
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
  }

  start(): void {
    const loop = (timestamp: number) => {
      this.update(timestamp);
      this.render();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private update(timestamp: number): void {
    const phase = this.game.get_phase();

    // Handle input
    const actions = this.input.consumeJustPressed();
    for (const action of actions) {
      switch (action) {
        case "toggle_ai":
          this.aiMode = !this.aiMode;
          this.ui.setAIStatus(this.aiMode);
          break;
        case "restart":
          this.restart();
          return;
      }

      if (phase === PHASE_GAME_OVER) continue;

      if (!this.aiMode) {
        switch (action) {
          case "move_left":
            this.game.move_left();
            break;
          case "move_right":
            this.game.move_right();
            break;
          case "rotate_cw":
            this.game.rotate_cw();
            break;
          case "rotate_ccw":
            this.game.rotate_ccw();
            break;
          case "hard_drop":
            this.game.hard_drop();
            break;
        }
      }
    }

    if (phase === PHASE_GAME_OVER) {
      if (!this.gameOverShown) {
        this.ui.showGameOver(this.game);
        this.gameOverShown = true;
      }
      return;
    }

    // AI mode
    if (this.aiMode && phase === PHASE_FALLING) {
      if (timestamp - this.lastAiTime > AI_PLAY_INTERVAL) {
        this.game.ai_play_move();
        this.lastAiTime = timestamp;
      }
    }

    // Gravity
    if (!this.aiMode && phase === PHASE_FALLING) {
      const gravity = this.input.isHeld("soft_drop")
        ? GRAVITY_FAST
        : GRAVITY_NORMAL;
      this.game.tick(gravity);
    }

    this.ui.update(this.game);
  }

  private render(): void {
    this.renderer.render(this.game);
  }

  private restart(): void {
    const seed = BigInt(Date.now());
    this.game.free();
    this.game = this.createGame(seed);
    this.gameOverShown = false;
    this.ui.hideGameOver();
    this.ui.update(this.game);
  }
}
