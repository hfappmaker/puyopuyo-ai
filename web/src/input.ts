export type InputAction =
  | "move_left"
  | "move_right"
  | "rotate"
  | "soft_drop"
  | "hard_drop"
  | "ai_move"
  | "restart";

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
      case "ArrowLeft":
        return "move_left";
      case "ArrowRight":
        return "move_right";
      case "ArrowUp":
        return "rotate";
      case "ArrowDown":
        return "soft_drop";
      case "Space":
        return "ai_move";
      case "KeyR":
        return "restart";
      default:
        return null;
    }
  }
}
