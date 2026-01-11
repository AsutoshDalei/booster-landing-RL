/**
 * controls.js
 * Handles user input and updates state control variables.
 */

// Key mapping
const KEYS = {
    THROTTLE_UP: 'w',
    THROTTLE_DOWN: 's',
    GIMBAL_LEFT: 'a',
    GIMBAL_RIGHT: 'd',
    RCS_LEFT: 'q',
    RCS_RIGHT: 'e',
    CUT_ENGINE: 'x',
    RESET: 'r',
    AUTOPILOT: 'p'
};

const THROTTLE_SPEED = 0.5; // Change per second
const GIMBAL_SPEED = 2.0;   // Radians per second
const MAX_GIMBAL = 0.5;     // Max gimbal angle (radians)

// Input State
const keyState = {};

export function initControls(state, resetCallback) {
    window.addEventListener('keydown', (e) => {
        keyState[e.key.toLowerCase()] = true;

        // Instant actions
        if (e.key.toLowerCase() === KEYS.CUT_ENGINE) {
            state.rocket.throttle = 0;
        }
        if (e.key.toLowerCase() === KEYS.RESET) {
            resetCallback();
        }
        if (e.key.toLowerCase() === KEYS.AUTOPILOT) {
            state.autopilotEnabled = !state.autopilotEnabled;
            console.log("Autopilot:", state.autopilotEnabled);
        }
    });

    window.addEventListener('keyup', (e) => {
        keyState[e.key.toLowerCase()] = false;
    });
}

export function updateControls(state, dt) {
    const r = state.rocket;

    // Throttle Control (Continuous Ramp)
    if (keyState[KEYS.THROTTLE_UP]) {
        r.throttle += THROTTLE_SPEED * dt;
    }
    if (keyState[KEYS.THROTTLE_DOWN]) {
        r.throttle -= THROTTLE_SPEED * dt;
    }
    // Clamp Throttle
    r.throttle = Math.max(0, Math.min(1.0, r.throttle));

    // Gimbal Control (Continuous Move)
    if (keyState[KEYS.GIMBAL_LEFT]) {
        // 'A' moves nozzle Left relative to rocket (pushes nose Left? No.)
        // Visual: If I press Left, I want to steer Left.
        // To steer Left, I need Torque CCW.
        // To get Torque CCW, Gimbal should point RIGHT? (Thrust pushes Left).
        // Let's assume 'A' maps to negative angle, 'D' maps to positive.
        // We will adjust visually.
        r.engineGimbal -= GIMBAL_SPEED * dt;
    }
    if (keyState[KEYS.GIMBAL_RIGHT]) {
        r.engineGimbal += GIMBAL_SPEED * dt;
    }
    // Clamp Gimbal
    r.engineGimbal = Math.max(-MAX_GIMBAL, Math.min(MAX_GIMBAL, r.engineGimbal));

    // RCS Control (Discrete Instant)
    // Q -> Rotate Left (CCW). 
    // E -> Rotate Right (CW).
    r.rcsLeft = !!keyState[KEYS.RCS_LEFT];
    r.rcsRight = !!keyState[KEYS.RCS_RIGHT];
}
