export type InputAction = "next_move" | "toggle_auto" | "restart";

export class InputHandler {
  onAction: ((action: InputAction) => void) | null = null;

  constructor() {
    window.addEventListener("keydown", (e) => this.onKeyDown(e));
  }

  private onKeyDown(e: KeyboardEvent): void {
    const action = this.mapKey(e);
    if (action) {
      e.preventDefault();
      this.onAction?.(action);
    }
  }

  private mapKey(e: KeyboardEvent): InputAction | null {
    switch (e.code) {
      case "Space":
      case "ArrowRight":
        return "next_move";
      case "KeyA":
        return "toggle_auto";
      case "KeyR":
        return "restart";
      default:
        return null;
    }
  }
}
