import type { WasmGame } from "./types";

const MODEL_URL = "/models/puyo_model.bin";
const NORM_PARAMS_URL = "/models/norm_params.txt";

export async function loadNnModel(game: WasmGame): Promise<boolean> {
  try {
    const [modelResponse, normResponse] = await Promise.all([
      fetch(MODEL_URL),
      fetch(NORM_PARAMS_URL),
    ]);

    if (!modelResponse.ok || !normResponse.ok) {
      console.warn("NN model files not found. Using heuristic AI.");
      return false;
    }

    const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
    const normText = await normResponse.text();
    const [meanStr, stdStr] = normText.trim().split("\n");
    const mean = parseFloat(meanStr);
    const stdDev = parseFloat(stdStr);

    game.load_nn_model(modelBytes, mean, stdDev);
    console.log(`NN model loaded (mean=${mean}, std=${stdDev})`);
    return true;
  } catch (e) {
    console.warn("Failed to load NN model:", e);
    return false;
  }
}
