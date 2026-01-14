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
                handleCoast(state, g, r, altitude, pad);
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

function handleCoast(state, g, r, altitude, pad) {
    g.engineEnabled = false; // Cut engine in coast

    // Switch to BURN if stopping distance > altitude
    // Stopping Distance Calculation:
    // v^2 = u^2 + 2as. Final v=0. u=Current Vy. 
    // s = v^2 / 2a.
    // Accel a = NET_ACCEL.
    // Safety Factor: 1.3 (Earlier ignition)

    // Ignition Logic
    // User Override: Default ignition altitude check
    const userIgnition = state.tuning.ignitionAltitude || 400;

    // "Side Engines" Logic / Early Alignment:
    // If we are far from the pad, ignite EARLIER to allow time for traversal.
    // Heuristic: Add buffer to ignition altitude based on distance.
    const distToPad = Math.abs(r.x - pad.x);
    // If dist is 300px, add say 200px to ignition height
    const extraHeight = distToPad * 0.8;

    const triggerAlt = userIgnition + extraHeight;

    // Check altitude regardless of velocity (removed r.vy > 10 requirement)
    // This ensures ignition happens at the correct altitude even if falling slowly
    if (altitude <= triggerAlt) {
        console.log(`GUIDANCE: IGNITION - Altitude: ${altitude.toFixed(0)}, Trigger: ${triggerAlt.toFixed(0)}, Dist: ${distToPad.toFixed(0)}`);
        g.phase = 'BURN';
        g.engineEnabled = true;
    }

    g.targetVy = 0; // Irrelevant if engine cut
    g.targetAngle = 0; // Keep upright
}

function handleBurn(state, g, r, altitude, pad) {
    // Engine can turn off mid-flight if conditions warrant, then turn back on
    // This allows for more flexible trajectory control (coast phases)

    // Check if we should cut engine (transition back to COAST)
    // Conditions to cut engine:
    // 1. Ascending too fast (vy < -5) - we're going up, cut engine
    // 2. Very high altitude and slow descent - can coast to save fuel
    const isAscending = r.vy < -5.0;
    const isHighAndSlow = altitude > 200 && r.vy < 10.0;

    if (isAscending || isHighAndSlow) {
        // Transition back to COAST - engine can turn back on later
        g.phase = 'COAST';
        g.engineEnabled = false;
        console.log(`GUIDANCE: ENGINE CUT - Coasting (Alt: ${altitude.toFixed(0)}, Vy: ${r.vy.toFixed(1)})`);
        return; // Exit early, will re-enter in COAST phase next frame
    }

    // Otherwise, keep engine on
    g.engineEnabled = true;

    // 1. Vertical Guidance (Trajectory)
    // Use a velocity profile that slows down more aggressively near the ground

    let desiredVy = 0;

    if (altitude > 100) {
        // High altitude: use square root profile for efficient braking
        const a_target = NET_ACCEL * 0.75;
        desiredVy = Math.sqrt(2 * a_target * altitude);
        // Don't exceed current velocity - always slow down
        desiredVy = Math.min(desiredVy, r.vy);
        // Minimum to maintain control
        desiredVy = Math.max(desiredVy, 40);
    } else if (altitude > 30) {
        // Medium altitude: more aggressive slowdown
        const a_target = NET_ACCEL * 0.7;
        desiredVy = Math.sqrt(2 * a_target * altitude);
        desiredVy = Math.min(desiredVy, r.vy);
        desiredVy = Math.max(desiredVy, 25);
    } else if (altitude > 10) {
        // Low altitude: aggressive slowdown
        const a_target = NET_ACCEL * 0.6;
        desiredVy = Math.sqrt(2 * a_target * altitude);
        desiredVy = Math.min(desiredVy, r.vy);
        desiredVy = Math.max(desiredVy, 12);
    } else if (altitude > 3) {
        // Very low: slow approach for soft landing
        const a_target = NET_ACCEL * 0.5;
        desiredVy = Math.sqrt(2 * a_target * altitude);
        desiredVy = Math.min(desiredVy, r.vy);
        desiredVy = Math.max(desiredVy, 5);
    } else {
        // Almost touching: minimal descent speed
        desiredVy = Math.max(2, Math.min(r.vy, 8));
    }

    // Ensure target is reasonable (not faster than current, not too slow)
    g.targetVy = Math.max(2, Math.min(desiredVy, r.vy + 5));

    // 2. Horizontal Guidance (Steering)
    const xError = r.x - pad.x;

    // When very close to ground, prioritize being upright over horizontal positioning
    // Transition smoothly: use steering at high altitude, straighten out when low
    let cmdAngle = 0;

    if (altitude > 30) {
        // High enough: allow steering to correct horizontal position
        // Increased steering gain and max tilt for better angle adjustment
        const K_STEER = 0.004;  // Increased from 0.003
        cmdAngle = -xError * K_STEER;
        const MAX_TILT = 0.3;  // Increased from 0.25 for more aggressive steering
        cmdAngle = Math.max(-MAX_TILT, Math.min(MAX_TILT, cmdAngle));
    } else if (altitude > 15) {
        // Medium altitude: reduce steering gain, start straightening
        const K_STEER = 0.002;
        cmdAngle = -xError * K_STEER;
        const MAX_TILT = 0.15;
        cmdAngle = Math.max(-MAX_TILT, Math.min(MAX_TILT, cmdAngle));
    } else {
        // Low altitude: prioritize upright, minimal steering
        const K_STEER = 0.001;
        cmdAngle = -xError * K_STEER;
        const MAX_TILT = 0.08;
        cmdAngle = Math.max(-MAX_TILT, Math.min(MAX_TILT, cmdAngle));
    }

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

            // Strict Landing: Both legs must be on the pad
            // This means the center of the rocket must be close enough to the center of the pad
            // such that the rocket's half-width fits within the pad's half-width.
            const maxDist = (pad.width / 2) - (r.width / 2);
            const onPad = Math.abs(r.x - pad.x) <= maxDist;

            if (!isUpright) {
                console.log(`GUIDANCE: FAILURE - TIPPED (Angle: ${angleDeg.toFixed(1)}°)`);
                g.landingResult = 'FAILURE';
            } else if (!onPad) {
                console.log(`GUIDANCE: FAILURE - MISSED PAD (Dist: ${Math.abs(r.x - pad.x).toFixed(0)}, Max: ${maxDist.toFixed(0)})`);
                g.landingResult = 'FAILURE';
            } else {
                console.log("GUIDANCE: SUCCESS - PERFECT LANDING");
                g.landingResult = 'SUCCESS';
            }
        }
    }
}

function handleLanded(state, g, r) {
    // Once landed, engine is permanently disabled
    // Cannot transition back to COAST or BURN
    g.engineEnabled = false;
    g.targetVy = 0;
    g.targetAngle = 0;
}
