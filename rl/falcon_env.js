/**
 * falcon_env.js
 * Headless simulation environment for RL Training.
 * Ports logic from physics.js, state.js, and guidance.js
 */

// --- Constants (From physics.js) ---
const GRAVITY = 100.0;
const THRUST_POWER = 300.0;
const MASS = 1.0;
const MOMENT_OF_INERTIA = 1000.0;
const DRAG_COEFFICIENT = 0.05;
const ANGULAR_DRAG = 2.0;
const RCS_THRUST = 50.0;
const FRICTION = 0.5;
const RESTITUTION = 0.0;
const STOP_THRESHOLD = 0.5;

// --- Sim Constants ---
const DT = 1 / 60;
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const PAD_X = 400; // Center
const PAD_Y = CANVAS_HEIGHT - 50;
const PAD_WIDTH = 120; // Approx
const ROCKET_WIDTH = 24;
const ROCKET_HEIGHT = 80; // Approx

class FalconEnv {
    constructor() {
        this.reset();
    }

    reset() {
        // Randomize Initial State (Simulating 'resetRocket' from state.js)
        this.state = {
            x: PAD_X + (Math.random() - 0.5) * 200, // Random X +/- 100
            y: 50 + Math.random() * 100,            // Start high
            vx: (Math.random() - 0.5) * 20,         // Random drift
            vy: 0,
            angle: (Math.random() - 0.5) * 0.2,     // Small random angle (+/- ~6 deg)
            angularVelocity: 0,
            throttle: 0,
            engineGimbal: 0,
            fuel: 100,
            groundContact: false,
            landingResult: null, // 'SUCCESS', 'FAILURE', or null
            done: false,
            stepCount: 0
        };

        return this._get_obs();
    }

    // Step the simulation
    step(action) {
        // Action: [throttle (0-1), gimbal (-1 to 1), rcs (-1 to 1)]
        let [throttle, gimbal, rcs] = action;

        // Clamp Inputs
        throttle = Math.max(0, Math.min(1, throttle));
        gimbal = Math.max(-1, Math.min(1, gimbal));
        rcs = Math.max(-1, Math.min(1, rcs));

        // Apply Control Actions
        this.state.throttle = throttle;
        this.state.engineGimbal = gimbal * 0.25; // Max 0.25 rad

        // RCS logic (Discrete-ish)
        let torqueInput = 0;
        if (rcs < -0.3) torqueInput = -10000;
        else if (rcs > 0.3) torqueInput = 10000;

        // --- Physics Step (Ported from physics.js) ---
        const r = this.state;

        // Forces
        let fx = 0;
        let fy = GRAVITY * MASS;
        let torque = torqueInput;

        // Thrust
        if (r.throttle > 0) {
            const thrustMag = r.throttle * THRUST_POWER * MASS;
            const thrustAngle = r.angle + r.engineGimbal;
            fx += Math.sin(thrustAngle) * thrustMag;
            fy -= Math.cos(thrustAngle) * thrustMag; // Standard: Thust pushes UP (-Y)

            // Torque from Gimbal
            const leverArm = ROCKET_HEIGHT / 2;
            torque += -leverArm * thrustMag * Math.sin(r.engineGimbal);
        }

        // Drag
        fx -= r.vx * DRAG_COEFFICIENT;
        fy -= r.vy * DRAG_COEFFICIENT;
        torque -= r.angularVelocity * ANGULAR_DRAG;

        // Integration
        r.vx += (fx / MASS) * DT;
        r.vy += (fy / MASS) * DT;
        r.angularVelocity += (torque / MOMENT_OF_INERTIA) * DT;

        r.x += r.vx * DT;
        r.y += r.vy * DT;
        r.angle += r.angularVelocity * DT;

        // Collision (Ported from physics.js + fix)
        const groundY = PAD_Y - 10; // Approx pad top
        const rocketBottom = r.y + ROCKET_HEIGHT / 2;

        if (rocketBottom >= groundY) {
            r.groundContact = true; // FIXED: Logic 81
            r.y = groundY - ROCKET_HEIGHT / 2;

            if (r.vy > 0) {
                r.vx *= FRICTION;
                r.angularVelocity *= 0.8;
                if (r.vy < STOP_THRESHOLD) {
                    r.vy = 0;
                } else {
                    r.vy = -r.vy * RESTITUTION;
                }
            }
        }

        // --- Outcome Detection (Ported from guidance.js) ---
        // Out of Bounds
        if (r.x < 0 || r.x > CANVAS_WIDTH || r.y > CANVAS_HEIGHT + 100 || r.y < -500) {
            r.landingResult = 'FAILURE';
            r.done = true;
        }

        // Landing Logic (Strict)
        if (r.groundContact && !r.done) {
            // Hard Crash
            if (Math.abs(r.vy) > 10.0) {
                r.landingResult = 'FAILURE';
                r.done = true;
            } else {
                // Soft Landing Check
                // We need to wait for it to settle? Or instant fail if tipped?
                // Logic says: if angle > 5 deg -> FAIL
                const angleDeg = Math.abs(r.angle * 180 / Math.PI);
                const isUpright = angleDeg < 5.0;
                const onPad = Math.abs(r.x - PAD_X) < (PAD_WIDTH / 2 + ROCKET_WIDTH);

                if (!isUpright) {
                    r.landingResult = 'FAILURE';
                    r.done = true;
                } else if (!onPad) {
                    r.landingResult = 'FAILURE';
                    r.done = true;
                } else {
                    // Success condition: Speed low, upright, on pad.
                    // Wait until speed is very low to declare success?
                    if (Math.abs(r.vy) < 1.0 && Math.abs(r.vx) < 1.0) {
                        r.landingResult = 'SUCCESS';
                        r.done = true;
                    }
                }
            }
        }

        // Timeout (fuel or steps)
        r.stepCount++;
        if (r.stepCount > 1000) {
            r.landingResult = 'FAILURE'; // Ran out of time/fuel
            r.done = true;
        }

        const reward = this._compute_reward();

        return {
            state: this._get_obs(),
            reward: reward,
            done: r.done,
            info: { result: r.landingResult }
        };
    }

    _get_obs() {
        const r = this.state;
        // Normalize observations for RL (approximate ranges)
        return [
            (r.x - PAD_X) / 400.0,      // X Dist (norm)
            (PAD_Y - r.y) / 600.0,      // Altitude (norm)
            r.vx / 100.0,               // Vx
            r.vy / 100.0,               // Vy
            r.angle / 3.14,             // Angle (normalized PI)
            r.angularVelocity / 10.0    // AngVel
        ];
    }

    _compute_reward() {
        const r = this.state;

        if (r.landingResult === 'FAILURE') return -100;
        if (r.landingResult === 'SUCCESS') return 100;

        // Shaping Reward
        let reward = 0;

        // 1. Distance penalty
        const distToPad = Math.abs(r.x - PAD_X);
        reward -= distToPad * 0.01;

        // 2. Angle penalty (keep upright)
        reward -= Math.abs(r.angle) * 0.1;

        // 3. Velocity penalty (encourage slowing down as we get lower)
        // Only penalize high speed near ground?
        // Simple shaping: - |Vy| * 0.05
        reward -= Math.abs(r.vy) * 0.01;

        // 4. Survival bonus (small positive per step to encourage not crashing instantly?)
        // Or negative per step to encourage fast landing? 
        // Suicide burn means fast landing is efficient. 
        // Let's use small penalty per step to encourage efficiency.
        reward -= 0.05;

        return reward;
    }
}

module.exports = FalconEnv;
