# Architecture & Choices (Step 0)

## Tech Stack
- **HTML5 Canvas**: Chosen for high-performance 2D rendering. It provides immediate mode rendering which is ideal for physics simulations where objects move every frame.
- **Vanilla JS (ES Modules)**: No build step required initially. Easier to debug and host on GitHub Pages.
- **Docker + live-server**: standardized dev environment. `live-server` injects a script to auto-reload the page when files change, speeding up UI iteration.

## Extensibility Plan

### State Variables
Currently, `state` is a flat object in `main.js`.
*   **Future**: We will create a `PhysicsWorld` class that holds `Rocket` and `Pad` instances.
*   The `Rocket` class will hold `position` (Vector2), `velocity` (Vector2), `angle`, `angularVelocity`, `fuel`, etc.

### Physics Updates
Currently, the `draw()` loop just renders static coordinates.
*   **Future**: We will add an `update(dt)` function called before `draw()`.
*   `update(dt)` will integrate equations of motion: `velocity += acceleration * dt`, `position += velocity * dt`.
*   We will implement a Fixed Timestep loop to ensure deterministic physics stability.

### Control Inputs
*   **Future**: A `InputHandler` class will listen for `keydown` / `keyup` events.
*   Maps keys (W/A/D or Arrows) to `rocket.engineOn` or `rocket.gimbalLeft`/`Right` booleans.
