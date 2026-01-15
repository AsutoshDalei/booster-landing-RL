/**
 * physics.js
 * Deterministic 2D Rigid Body Physics for Rocket Landing
 */

// --- Constants ---
// Tuned for pixel-space "game feel" (1m approx 10px? Not strictly defined yet)
// Y is positive DOWN
const GRAVITY = 100.0;           // px/s^2 down
const THRUST_POWER = 300.0;      // px/s^2 max acceleration (TWR ~3.0)
const MASS = 1.0;                // Normalized mass
const MOMENT_OF_INERTIA = 1000.0;// Resistance to rotation
const DRAG_COEFFICIENT = 0.05;   // Simple linear drag (air resistance)
const ANGULAR_DRAG = 2.0;       // Damping for rotation
const RCS_THRUST = 100.0;        // Force applied by RCS (updated for force-based application)
const MIN_THROTTLE = 0.4;       // Minimum throttle when engine is on (realistic rocket behavior)

// Collision Parameters
const RESTITUTION = 0.0;         // Bounciness (0 = no bounce, reliable landing)
const FRICTION = 0.5;            // Ground friction
const STOP_THRESHOLD = 0.5;      // Velocity below which we force stop

/**
 * Steps the physics simulation forward by dt seconds.
 * @param {Object} state - The complete simulation state (modified in place)
 * @param {number} dt - Timestep in seconds
 */
export function stepPhysics(state, dt) {
    const r = state.rocket;
    const pad = state.pad;

    // --- 1. Force Accumulation ---
    let fx = 0;
    let fy = 0;
    let torque = 0;

    // A. Gravity
    fy += GRAVITY * MASS;

    // B. Thrust
    // Applied only if we have fuel (Fuel logic not strictly requested yet, but good practice)
    // IMPORTANT: For RL mode, we need a deadband to prevent tiny throttle values from triggering MIN_THROTTLE
    // RL models can output very small values that shouldn't activate the engine
    const THROTTLE_DEADBAND = 0.05; // Minimum throttle to actually turn on engine
    
    if (r.throttle > THROTTLE_DEADBAND) {
        // Calculate power with minimum throttle (realistic rocket behavior)
        // When throttle > deadband, engine power is MIN_THROTTLE + throttle * (1 - MIN_THROTTLE)
        // This ensures engine cannot go below minimum throttle when on
        const power = MIN_THROTTLE + r.throttle * (1 - MIN_THROTTLE);
        const thrustMag = power * THRUST_POWER * MASS;

        // Direction
        // Rocket Angle 0 is Up (-Y in screen space? No, usually 0 is Right in Canvas 0 radian)
        // Let's standardize: 
        // In render.js: ctx.rotate(rocket.angle)
        // If angle=0, rect is drawn. If rect is drawn upright, then 0 is UP.
        // Wait, render.js usually draws rect centered. 
        // Let's Assume: Angle 0 = Up (-Y). 
        // Vector for Angle 0: [0, -1]
        // Vector for Angle theta: [sin(theta), -cos(theta)]

        // The thrust force is in the direction the engine is pointing.
        // Engine Gimbal is relative to rocket body.
        // Total angle of thrust vector = rocket.angle + rocket.engineGimbal
        const thrustAngle = r.angle + r.engineGimbal;

        // Thrust vector pushes the rocket.
        // If rocket points UP (0), thrust pushes UP (-Y).
        // fx += thrustMag * Math.sin(thrustAngle);
        // fy += thrustMag * -Math.cos(thrustAngle);
        // Wait, standard math: 0 is Right (+X). -PI/2 is Up (-Y).
        // Let's stick to the convention used in Render.js 
        // In Render.js: ctx.rotate(rocket.angle). 
        // If I pass 0, and drawRect(-w/2, -h/2, w, h), it draws a vertical box?
        // No, standard rect is horizontal? No, depends on w vs h.
        // Rocket width=30, height=100. It's a tall rect. 
        // So at angle 0, it is upright (vertical).
        // Therefore at angle 0, "Forward" is Up (-Y).

        const tx = Math.sin(thrustAngle); // Right component
        const ty = -Math.cos(thrustAngle); // Up component

        fx += tx * thrustMag;
        fy += ty * thrustMag;

        // C. Torque
        // Torque = r x F
        // Force is applied at the bottom of the rocket (engine).
        // COM is center. Distance is height/2.
        // Lever arm vector is from COM to Engine: [0, height/2] (rotated)
        // But simpler: Torque is caused by the *tangential* component of thrust relative to COM.
        // Gimbal angle creates a sideways component of thrust relative to the rocket axis.
        // Tangential Force = Thrust * sin(gimbal)
        // Torque = (Height / 2) * (Thrust * sin(gimbal))
        // Note: Gimbal positive -> rotates nozzle ?? 
        // If gimbal is + (Right), nozzle points Left? Or Right?
        // Let's assume standard: Gimbal +0.1 means nozzle points right.
        // Thrust pushes LEft. Torque rotates CW?
        // Let's stick to the visual: 
        // In render.js: rotate(gimbal). Positive rotation is CW.
        // So Nozzle points Right-ish.
        // Thrust vector points Left-ish (Opposite to flow).
        // Push on bottom-left -> Rotates CCW (Negative Torque).
        // So T = - (r * F_tangential).

        const leverArm = r.height / 2;
        // The component of thrust perpendicular to rocket axis triggers rotation
        // Perpendicular Force = ThrustMag * sin(gimbal)
        // Increase torque strength for better angle control (multiply by 1.5 for more responsiveness)
        torque += -leverArm * thrustMag * Math.sin(r.engineGimbal) * 1.5;
    }

    // C2. RCS Force Application (at thruster position)
    // RCS thrusters are at the top of the rocket and apply force perpendicular to rocket axis
    // This creates both torque (rotation) and linear velocity change (translation)
    if (r.rcsLeft || r.rcsRight) {
        // Calculate thruster position (top of rocket, 86% of height from center)
        // This matches gym-rocketlander's THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.86
        const THRUSTER_HEIGHT = r.height * 0.86;
        
        // Thruster position relative to rocket center
        const thrusterOffsetX = Math.sin(r.angle) * THRUSTER_HEIGHT;
        const thrusterOffsetY = -Math.cos(r.angle) * THRUSTER_HEIGHT;
        
        // Force direction: perpendicular to rocket axis
        // rcsLeft pushes rocket right (positive X), rcsRight pushes rocket left (negative X)
        const forceDir = r.rcsLeft ? 1 : -1;
        
        // Force components perpendicular to rocket axis
        // When rocket is upright (angle=0), force is horizontal
        const forceX = forceDir * Math.cos(r.angle) * RCS_THRUST;
        const forceY = forceDir * Math.sin(r.angle) * RCS_THRUST;
        
        // Apply force at center of mass (simplified - in reality force is at thruster position)
        fx += forceX;
        fy += forceY;
        
        // Force at thruster position also creates torque
        // Torque = lever_arm × force_perpendicular
        // Lever arm is the distance from COM to thruster
        const leverArm = THRUSTER_HEIGHT;
        // Perpendicular component of force relative to rocket axis
        const forcePerpendicular = forceDir * RCS_THRUST;
        // Torque sign: rcsLeft (forceDir=1) creates CCW rotation (negative torque)
        // rcsRight (forceDir=-1) creates CW rotation (positive torque)
        torque += -leverArm * forcePerpendicular;
    }

    // D. Drag (Simple linear damping)
    fx -= r.vx * DRAG_COEFFICIENT;
    fy -= r.vy * DRAG_COEFFICIENT;
    torque -= r.angularVelocity * ANGULAR_DRAG;

    // --- 2. Integration (Semi-Implicit Euler) ---
    // Update Velocities
    r.vx += (fx / MASS) * dt;
    r.vy += (fy / MASS) * dt;
    r.angularVelocity += (torque / MOMENT_OF_INERTIA) * dt;

    // Update Positions
    r.x += r.vx * dt;
    r.y += r.vy * dt;
    r.angle += r.angularVelocity * dt;

    // --- 3. Collision Constraints ---

    // Simple Ground Plane Collision
    // Ground is at Pad Y (minus half pad height? or just use a fixed number)
    // Pad Y in State is `canvas.height - 50`
    // Rocket Y is center. Bottom is Y + height/2.
    // Collision when (r.y + r.height/2) > groundY

    const groundY = pad.y - pad.height / 2; // Top surface of pad
    const rocketBottom = r.y + r.height / 2;

    r.groundContact = false;
    r.leg1Contact = false;
    r.leg2Contact = false;

    if (rocketBottom >= groundY) {
        // Penetration detected
        r.groundContact = true;

        // 1. Position Collection (Clamp)
        r.y = groundY - r.height / 2;

        // 2. Velocity Response
        if (r.vy > 0) {
            // Apply impact friction to horizontal
            r.vx *= FRICTION;

            // Kill rotation on landing (simple stabilization)
            r.angularVelocity *= 0.8;

            // Restitution (Bounce)
            if (r.vy < STOP_THRESHOLD) {
                r.vy = 0;
                // If angle is small, assume legs are touching
                if (Math.abs(r.angle) < 0.2) {
                    r.leg1Contact = true;
                    r.leg2Contact = true;
                }
            } else {
                r.vy = -r.vy * RESTITUTION;
            }
        }
    }
}
