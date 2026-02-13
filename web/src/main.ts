import { loadWasm } from "./wasm";
import { Renderer } from "./renderer";
import { GameLoop } from "./game-loop";
import { UI } from "./ui";
import { loadNnModel } from "./model-loader";

async function main() {
  const wasm = await loadWasm();

  const boardCanvas = document.getElementById("board-canvas") as HTMLCanvasElement;
  const nextCanvas = document.getElementById("next-canvas") as HTMLCanvasElement;
  const nextNextCanvas = document.getElementById("next-next-canvas") as HTMLCanvasElement;

  const renderer = new Renderer(boardCanvas, nextCanvas, nextNextCanvas);
  const ui = new UI();

  const createGame = (seed: bigint) => new wasm.WasmGame(seed);
  const game = createGame(BigInt(Date.now()));

  const gameLoop = new GameLoop(game, renderer, ui, createGame);
  gameLoop.start();

  // AI mode toggle
  const aiModeSelect = document.getElementById("ai-mode-select") as HTMLSelectElement;
  const aiModeStatus = document.getElementById("ai-mode-status") as HTMLElement;

  aiModeSelect.addEventListener("change", async () => {
    const mode = aiModeSelect.value;
    if (mode === "nn") {
      aiModeStatus.textContent = "モデル読み込み中...";
      const currentGame = gameLoop.getGame();
      const success = await loadNnModel(currentGame);
      if (success) {
        aiModeStatus.textContent = "NN AI 有効";
      } else {
        aiModeStatus.textContent = "モデル未配置";
        aiModeSelect.value = "heuristic";
      }
    } else {
      const currentGame = gameLoop.getGame();
      currentGame.use_heuristic();
      aiModeStatus.textContent = "";
    }
  });
}

main().catch(console.error);
