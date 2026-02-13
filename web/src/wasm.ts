import type { WasmModule } from "./types";

let wasmModule: WasmModule | null = null;

export async function loadWasm(): Promise<WasmModule> {
  if (wasmModule) return wasmModule;

  const mod = await import("../wasm-pkg/puyo_wasm.js");
  await mod.default();
  wasmModule = mod as unknown as WasmModule;
  return wasmModule;
}
