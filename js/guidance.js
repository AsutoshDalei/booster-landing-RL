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
    // Safety buffer: 1.1x

    // Note: r.vy is Positive Down (Descent).
    if (r.vy > 0) { // Only if falling
        const stoppingDist = (r.vy * r.vy) / (2 * NET_ACCEL);

        // Trigger if we are close to the limit
        // Use a safety margin (e.g. 10% or 50px)
        if (altitude <= stoppingDist * 1.05 + 50) {
            console.log("GUIDANCE: IGNITION - Starting Suicide Burn");
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
    // Desired Velocity at this altitude

    let desiredVy = 0;

    if (altitude > 0) {
        // Simple Square Root profile
        const a_target = NET_ACCEL * 0.85;
        desiredVy = Math.sqrt(2 * a_target * altitude);

        // Clamp minimum descent speed when close to ensure touchdown
        // We want to force it to LAND, not hover.
        // If altitude is very low (<10m), target speed should be at least 15 m/s
        if (altitude < 20) desiredVy = Math.max(desiredVy, 15);
        else desiredVy = Math.max(desiredVy, 10);
    } else {
        desiredVy = 5; // Positive = Descent. 
    }

    // Crucial: Never target Ascent (Negative Vy).
    // The physics/PID might still ascend if throttle is too high, 
    // but the setpoint must validly request descent.
    g.targetVy = Math.max(0, desiredVy);

    // 2. Horizontal Guidance (Steering)
    const xError = r.x - pad.x;
    const K_STEER = 0.003;
    let cmdAngle = -xError * K_STEER;

    const MAX_TILT = 0.25; // ~15 degrees
    cmdAngle = Math.max(-MAX_TILT, Math.min(MAX_TILT, cmdAngle));

    g.targetAngle = cmdAngle;

    // 3. Touchdown Detection
    if (r.groundContact) {
        // Determine Success
        // Velocity must be low (< 40)
        // Angle must be low (< 0.1 rad roughly 5 deg)
        // Both legs contact ideally, but let's just check stability

        // Check if stopped
        if (Math.abs(r.vy) < 5) {
            console.log("GUIDANCE: TOUCHDOWN");
            g.phase = 'LANDED';

            const angleDeg = Math.abs(r.angle * 180 / Math.PI);
            const isUpright = angleDeg < 5;
            const isSlow = Math.abs(r.vy) < 40; // Impact speed check (already happened on physics contact technically)
            const onPad = Math.abs(r.x - pad.x) < (pad.width / 2 + r.width);

            if (isUpright && onPad) {
                g.landingResult = 'SUCCESS';
            } else {
                g.landingResult = 'FAILURE';
            }
        }
    }
}

function handleLanded(state, g, r) {
    g.engineEnabled = false;
    g.targetVy = 0;
    g.targetAngle = 0;
}
