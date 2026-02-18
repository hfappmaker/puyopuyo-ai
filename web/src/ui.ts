import type { WasmGame } from "./types";

export class UI {
  private scoreEl: HTMLElement;
  private chainEl: HTMLElement;
  private piecesEl: HTMLElement;
  private gameOverOverlay: HTMLElement;
  private finalScoreEl: HTMLElement;
  private finalChainEl: HTMLElement;

  constructor() {
    this.scoreEl = document.getElementById("score-display")!;
    this.chainEl = document.getElementById("chain-display")!;
    this.piecesEl = document.getElementById("pieces-display")!;
    this.gameOverOverlay = document.getElementById("game-over-overlay")!;
    this.finalScoreEl = document.getElementById("final-score")!;
    this.finalChainEl = document.getElementById("final-chain")!;
  }

  update(game: WasmGame): void {
    this.scoreEl.textContent = game.get_score().toLocaleString();
    this.chainEl.textContent = game.get_max_chain().toString();
    this.piecesEl.textContent = game.get_total_pieces().toString();
  }

  showGameOver(game: WasmGame): void {
    this.finalScoreEl.textContent = game.get_score().toLocaleString();
    this.finalChainEl.textContent = game.get_max_chain().toString();
    this.gameOverOverlay.classList.add("show");
  }

  hideGameOver(): void {
    this.gameOverOverlay.classList.remove("show");
  }
}
