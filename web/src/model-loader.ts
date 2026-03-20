import type { WasmGame } from "./types";

const MODEL_URL = "/models/puyo_model.bin";

export async function loadNnModel(game: WasmGame): Promise<boolean> {
  try {
    const modelResponse = await fetch(MODEL_URL);

    if (!modelResponse.ok) {
      console.warn("NN model file not found. Using heuristic AI.");
      return false;
    }

    const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());

    game.load_nn_model(modelBytes);
    console.log("NN policy model loaded");
    return true;
  } catch (e) {
    console.warn("Failed to load NN model:", e);
    return false;
  }
}
