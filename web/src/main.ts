import { loadWasm } from "./wasm";
import { Renderer } from "./renderer";
import { GameLoop } from "./game-loop";
import { UI } from "./ui";
import { loadNnModel, loadNnModelWithMcts } from "./model-loader";
import { CELL_SIZE } from "./constants";
import type { BoardConfig } from "./constants";

function setupAiModeToggle(gameLoop: GameLoop): void {
  const aiModeSelect = document.getElementById("ai-mode-select") as HTMLSelectElement;
  const aiModeStatus = document.getElementById("ai-mode-status") as HTMLElement;
  const mctsOptions = document.getElementById("mcts-options") as HTMLElement;
  const mctsSimInput = document.getElementById("mcts-simulations") as HTMLInputElement;

  const applyAiMode = async (game: ReturnType<typeof gameLoop.getGame>) => {
    const mode = aiModeSelect.value;
    if (mode === "nn") {
      await loadNnModel(game);
    } else if (mode === "nn-mcts") {
      const numSim = parseInt(mctsSimInput.value, 10) || 50;
      await loadNnModelWithMcts(game, numSim);
    }
  };

  aiModeSelect.addEventListener("change", async () => {
    const mode = aiModeSelect.value;
    mctsOptions.style.display = mode === "nn-mcts" ? "" : "none";

    if (mode === "nn") {
      aiModeStatus.textContent = "モデル読み込み中...";
      const success = await loadNnModel(gameLoop.getGame());
      if (success) {
        aiModeStatus.textContent = "NN AI 有効";
      } else {
        aiModeStatus.textContent = "モデル未配置";
        aiModeSelect.value = "heuristic";
      }
    } else if (mode === "nn-mcts") {
      const numSim = parseInt(mctsSimInput.value, 10) || 50;
      aiModeStatus.textContent = `MCTS読み込み中 (${numSim}sim)...`;
      const success = await loadNnModelWithMcts(gameLoop.getGame(), numSim);
      if (success) {
        aiModeStatus.textContent = `NN MCTS 有効 (${numSim}sim)`;
      } else {
        aiModeStatus.textContent = "モデル未配置";
        aiModeSelect.value = "heuristic";
        mctsOptions.style.display = "none";
      }
    } else {
      gameLoop.getGame().use_heuristic();
      aiModeStatus.textContent = "";
    }
  });

  mctsSimInput.addEventListener("change", () => {
    if (aiModeSelect.value === "nn-mcts") {
      const numSim = parseInt(mctsSimInput.value, 10) || 50;
      gameLoop.getGame().set_mcts_simulations(numSim);
      aiModeStatus.textContent = `NN MCTS 有効 (${numSim}sim)`;
    }
  });

  gameLoop.setOnRestart(applyAiMode);
}

async function main() {
  const wasm = await loadWasm();

  const COLS = 6;
  const ROWS = 14;
  const NUM_COLORS = 4;

  const createGame = () => new wasm.WasmGame(COLS, ROWS, NUM_COLORS);
  const game = createGame();

  const config: BoardConfig = {
    cols: game.board_cols(),
    rows: game.board_rows(),
    visibleRows: game.board_visible_rows(),
    numColors: game.num_colors(),
    boardWidth: game.board_cols() * CELL_SIZE,
    boardHeight: game.board_rows() * CELL_SIZE,
  };

  const boardCanvas = document.getElementById("board-canvas") as HTMLCanvasElement;
  const nextCanvas = document.getElementById("next-canvas") as HTMLCanvasElement;
  const nextNextCanvas = document.getElementById("next-next-canvas") as HTMLCanvasElement;

  boardCanvas.width = config.boardWidth;
  boardCanvas.height = config.boardHeight;

  const renderer = new Renderer(boardCanvas, nextCanvas, nextNextCanvas, config);
  const ui = new UI();

  const gameLoop = new GameLoop(game, renderer, ui, createGame);
  gameLoop.start();

  setupAiModeToggle(gameLoop);
}

main().catch(console.error);
