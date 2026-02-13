export type InputAction =
  | "move_left"
  | "move_right"
  | "rotate_cw"
  | "rotate_ccw"
  | "soft_drop"
  | "hard_drop"
  | "toggle_ai"
  | "restart";

export class InputHandler {
  private actions: Set<InputAction> = new Set();
  private justPressed: Set<InputAction> = new Set();
  private held: Set<string> = new Set();

  constructor() {
    window.addEventListener("keydown", (e) => this.onKeyDown(e));
    window.addEventListener("keyup", (e) => this.onKeyUp(e));
  }

  private onKeyDown(e: KeyboardEvent): void {
    if (this.held.has(e.code)) return; // ignore repeat
    this.held.add(e.code);

    const action = this.mapKey(e);
    if (action) {
      e.preventDefault();
      this.justPressed.add(action);
      this.actions.add(action);
    }
  }

  private onKeyUp(e: KeyboardEvent): void {
    this.held.delete(e.code);
    const action = this.mapKey(e);
    if (action) {
      this.actions.delete(action);
    }
  }

  private mapKey(e: KeyboardEvent): InputAction | null {
    switch (e.code) {
      case "ArrowLeft":
        return "move_left";
      case "ArrowRight":
        return "move_right";
      case "ArrowUp":
      case "KeyX":
        return "rotate_cw";
      case "KeyZ":
        return "rotate_ccw";
      case "ArrowDown":
        return "soft_drop";
      case "Space":
        return "hard_drop";
      case "KeyA":
        return "toggle_ai";
      case "KeyR":
        return "restart";
      default:
        return null;
    }
  }

  /// Consume all just-pressed actions for this frame.
  consumeJustPressed(): InputAction[] {
    const result = [...this.justPressed];
    this.justPressed.clear();
    return result;
  }

  /// Check if an action is currently held.
  isHeld(action: InputAction): boolean {
    return this.actions.has(action);
  }
}
