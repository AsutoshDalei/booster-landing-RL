/**
 * state.js
 * Defines the formal state model for the simulation.
 */

// Formal State Model as requested
export const RocketState = {
    // Spatial
    x: 0,
    y: 0,
    angle: 0,        // radians
    width: 24,       // visual dimensions (Slimmer)
    height: 100,

    // Physics (Placeholder for now, but explicit in state)
    vx: 0,
    vy: 0,
    angularVelocity: 0,

    // Control/System State
    throttle: 0.0,   // 0.0 to 1.0
    engineGimbal: 0, // radians, 0 is straight down relative to rocket
    rcsLeft: false,  // Thruster firing to push nose Left
    rcsRight: false, // Thruster firing to push nose Right
    fuel: 100.0,     // %

    // Contact State
    leg1Contact: false,
    leg2Contact: false,
    groundContact: false
};

// Global Simulation State
export const SimState = {
    rocket: { ...RocketState },
    pad: {
        x: 0,
        y: 0,
        width: 120,
        height: 10
    },
    // Simple particle system for exhaust
    particles: [],
    // Particle structure: { x, y, vx, vy, life (0-1), color }
    autopilotEnabled: true,

    guidance: {
        targetVy: 0,
        targetAngle: 0,
        phase: 'COAST', // COAST, BURN, LANDED
        engineEnabled: true, // Master switch for guidance
        landingResult: null // null, 'SUCCESS', 'FAILURE'
    },

    tuning: {
        ignitionAltitude: 400,
        gimbalKp: 2.5,
        gimbalKd: 1.5,
        rcsDeadband: 0.02
    }
};

export function resetRocket(state, canvasWidth, canvasHeight) {
    // Random Start Conditions
    const randomX = (Math.random() - 0.5) * 200; // +/- 100px (Narrower)
    // Random initial Angle: 3 to 8 degrees, either direction
    // 3 deg = 0.052 rad, 8 deg = 0.14 rad
    const angleMag = (Math.random() * 5 + 3) * (Math.PI / 180);
    const randomAngle = angleMag * (Math.random() < 0.5 ? 1 : -1);

    state.rocket.x = canvasWidth / 2 + randomX;
    state.rocket.y = canvasHeight * 0.1; // Start high
    state.rocket.angle = randomAngle;
    state.rocket.vx = 0;
    state.rocket.vy = 50; // Initial drop speed
    state.rocket.throttle = 0; // Engine OFF
    state.rocket.engineGimbal = 0;
    state.rocket.leg1Contact = false;
    state.rocket.leg2Contact = false;
    state.rocket.groundContact = false;

    // Reset Guidance State
    state.guidance.phase = 'COAST';
    state.guidance.targetVy = 0;
    state.guidance.targetAngle = 0;
    state.guidance.landingResult = null;

    // Reset Pad
    state.pad.x = canvasWidth / 2;
    state.pad.y = canvasHeight - 50;
    state.particles = [];
}
