/**
 * guidance.js
 * High-level guidance computer. Determines trajectory and burn timing.
 */

// Physics Constants (Duplicate from physics.js for estimation - simplified)
const G = 100.0; // Gravity
const T_ACCEL_MAX = 300.0; // Max Thrust Accel (Thrust/Mass)
const NET_ACCEL = T_ACCEL_MAX - G; // Max Upward Accel

/**
 * Updates the Guidance State Machine.
 * @param {Object} state - Global Simulation State
 * @param {number} dt - Timestep
 */
export const Guidance = {
    update(state, dt) {
        if (!state.autopilotEnabled) return;

        const g = state.guidance;
        const r = state.rocket;
        const pad = state.pad;

        // 0. Screen Boundary Check (Fail if out of bounds)
        // Allow some headroom at top for spawning, but side/bottom bounds are strict
        const canvasWidth = 800; // Hardcoded or passed in state? Ideally passed. 
        // We'll assume typical width or check if state has it. 
        // state.js doesn't store canvas dims.
        // Let's use loose bounds or rely on render clipping?
        // User rq: "Rocket center leaves visible screen bounds"
        if (r.y > 1000 || r.x < -50 || r.x > 850) { // Approx bounds
            if (g.landingResult === null) {
                g.landingResult = 'FAILURE';
                g.engineEnabled = false;
                g.phase = 'LANDED';
                console.log("GUIDANCE: FAILURE - Out of bounds");
            }
            return;
        }

        // Altitude Calculation
        const padTop = pad.y - pad.height / 2;
        const rocketBottom = r.y + r.height / 2;
        const altitude = Math.max(0, padTop - rocketBottom);

        // Phase Logic
        switch (g.phase) {
            case 'COAST':
                handleCoast(state, g, r, altitude);
                break;
            case 'BURN':
                handleBurn(state, g, r, altitude, pad);
                break;
            case 'LANDED':
                handleLanded(state, g, r);
                break;
        }
    }
};

function handleCoast(state, g, r, altitude) {
    g.engineEnabled = false; // Cut engine in coast

    // Switch to BURN if stopping distance > altitude
    // Stopping Distance Calculation:
    // v^2 = u^2 + 2as. Final v=0. u=Current Vy. 
    // s = v^2 / 2a.
    // Accel a = NET_ACCEL.
    // Safety Factor: 1.3 (Earlier ignition)

    // Note: r.vy is Positive Down (Descent).
    if (r.vy > 10) { // Only if falling meaningfully
        // User Override: Default ignition altitude check
        const userIgnition = state.tuning.ignitionAltitude || 400;

        // Physics Safety Check (still calc stopping dist just in case user sets it absurdly low? 
        // No, current logic is "Safety Factor 1.3". 
        // Let's rely on user setting mainly, or MAX of both?
        // Prompt says: "When set, the suicide-burn logic uses this altitude as the ignition trigger"
        // Let's use the USER value as the primary trigger.

        if (altitude <= userIgnition) {
            console.log("GUIDANCE: IGNITION - User Trigger");
            g.phase = 'BURN';
            g.engineEnabled = true;
        }
    }

    g.targetVy = 0; // Irrelevant if engine cut
    g.targetAngle = 0; // Keep upright
}

function handleBurn(state, g, r, altitude, pad) {
    g.engineEnabled = true;

    // 1. Vertical Guidance (Trajectory)

    let desiredVy = 0;

    if (altitude > 0) {
        // Simple Square Root profile
        // V = Sqrt(2 * a * h)
        // Use a_target slightly less than max to ensure we can always follow it
        const a_target = NET_ACCEL * 0.8;
        desiredVy = Math.sqrt(2 * a_target * altitude);

        // Clamp minimum descent speed when close
        // Ensure strictly positive (Descent)
        if (altitude < 50) desiredVy = Math.max(desiredVy, 10);
        else desiredVy = Math.max(desiredVy, 20);
    } else {
        desiredVy = 5;
    }

    // Force Descent: targetVy must be >= 0
    g.targetVy = Math.max(5, desiredVy);

    // 2. Horizontal Guidance (Steering)
    const xError = r.x - pad.x;
    const K_STEER = 0.003;
    let cmdAngle = -xError * K_STEER;
    const MAX_TILT = 0.25;
    cmdAngle = Math.max(-MAX_TILT, Math.min(MAX_TILT, cmdAngle));
    g.targetAngle = cmdAngle;

    // 3. Touchdown Detection
    if (r.groundContact) {
        // Only set result if we haven't already finished
        if (g.phase !== 'LANDED') {
            g.phase = 'LANDED';

            // Hard Crash Check (High Velocity)
            if (Math.abs(r.vy) > 10.0) {
                console.log(`GUIDANCE: CRASH (Speed ${r.vy.toFixed(1)})`);
                g.landingResult = 'FAILURE';
                return;
            }

            // Soft Landing Analysis
            const angleDeg = Math.abs(r.angle * 180 / Math.PI);
            const isUpright = angleDeg < 5.0; // Strict threshold
            const onPad = Math.abs(r.x - pad.x) < (pad.width / 2 + r.width);

            if (!isUpright) {
                console.log(`GUIDANCE: FAILURE - TIPPED (Angle: ${angleDeg.toFixed(1)}°)`);
                g.landingResult = 'FAILURE';
            } else if (!onPad) {
                console.log(`GUIDANCE: FAILURE - MISSED PAD (Dist: ${Math.abs(r.x - pad.x).toFixed(0)})`);
                g.landingResult = 'FAILURE';
            } else {
                console.log("GUIDANCE: SUCCESS - PERFECT LANDING");
                g.landingResult = 'SUCCESS';
            }
        }
    }
}

function handleLanded(state, g, r) {
    g.engineEnabled = false;
    g.targetVy = 0;
    g.targetAngle = 0;
}
