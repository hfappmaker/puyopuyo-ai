import { loadWasm } from "./wasm";
import { Renderer } from "./renderer";
import { GameLoop } from "./game-loop";
import { UI } from "./ui";

async function main() {
  const wasm = await loadWasm();

  const boardCanvas = document.getElementById("board-canvas") as HTMLCanvasElement;
  const nextCanvas = document.getElementById("next-canvas") as HTMLCanvasElement;
  const nextNextCanvas = document.getElementById("next-next-canvas") as HTMLCanvasElement;

  const renderer = new Renderer(boardCanvas, nextCanvas, nextNextCanvas);
  const ui = new UI();

  const createGame = (seed: bigint) => new wasm.WasmGame(seed);
  const game = createGame(BigInt(Date.now()));

  const loop = new GameLoop(game, renderer, ui, createGame);
  loop.start();
}

main().catch(console.error);
