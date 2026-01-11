/**
 * autopilot.js
 * PID-based stabilization system.
 */

class PIDController {
    constructor(kp, ki, kd, min, max) {
        this.kp = kp;
        this.ki = ki;
        this.kd = kd;
        this.min = min;
        this.max = max;

        this.integral = 0;
        this.previousError = 0;
    }

    update(setpoint, measured, dt) {
        const error = setpoint - measured;
        this.integral += error * dt;
        const derivative = (error - this.previousError) / dt;
        this.previousError = error;

        let output = (this.kp * error) + (this.ki * this.integral) + (this.kd * derivative);

        // Clamp output
        if (output > this.max) output = this.max;
        if (output < this.min) output = this.min;

        return output;
    }

    reset() {
        this.integral = 0;
        this.previousError = 0;
    }
}

// Tuned Gains (Estimates)
// Angle: Needs strong response. 
// Gimbal Range: +/- 0.5 rad. Error 0.1 rad should give significant gimbal.
const GimbalPID = new PIDController(2.0, 0.0, 1.0, -0.5, 0.5); // was 2.0, 0.0, 1.0. Let's make it stronger?
// Request said "slightly stronger". Current 2.0 is already high.
// Let's go Kp=2.5, Kd=1.5.
// Wait, plan said Kp=0.8. Current file has Kp=2.0 ?? 
// Ah, I might have misread user request or file history. 
// The file view showed: `const GimbalPID = new PIDController(2.0, 0.0, 1.0, -0.5, 0.5);`
// Implementation plan said: "Increase Kp from 0.5 by 0.8". 
// It seems the current code ALREADY has 2.0.
// Let's set it to valid values that work. 
// Kp=2.5, Kd=1.5 is good for fast twitch.

// Velocity: Hover logic.
// Throttle Range: 0.0 to 1.0. 
// Plan: Kp = -0.08, Kd = -0.02.
const ThrottlePID = new PIDController(-0.08, -0.01, -0.03, 0.0, 1.0);

export const Autopilot = {
    update(state, dt) {
        if (!state.autopilotEnabled) return;

        const r = state.rocket;
        const g = state.guidance;

        // 0. Engine Cutoff Logic (Guidance Override)
        if (g && g.engineEnabled === false) {
            r.throttle = 0;
            // Still perform stabilization? Yes, RCS still works.
        } else {
            // 3. Velocity Hold (Throttle)
            // Use Guidance Target if available, else default
            const targetVy = (g && g.targetVy !== undefined) ? g.targetVy : 50.0;

            // Note: ThrottlePID tuned for Kp=-0.05 (Negative error -> Positive Throttle)
            // Error = Setpoint - Measured.
            // If Setpoint (Target Descent) = 100, Measured = 200 (Falling fast).
            // Error = -100. Output = +5. Increase Throttle.

            const throttleCmd = ThrottlePID.update(targetVy, r.vy, dt);

            // Ascent Prevention Clamp
            // If targetVy is positive (Descent) and r.vy is small (near hover) or negative (Ascent),
            // we must limit throttle to ensure we don't fly up.
            // Hover Throttle ~= G / T_ACCEL_MAX ~= 100 / 300 = 0.33

            let finalThrottle = throttleCmd;

            // If we are ascending (vy < 0) or nearly stopped (vy < 2), 
            // limit throttle to be below Hover threshold to force gravity to win.
            if (r.vy < 2.0) {
                const hoverThrottle = 100.0 / 300.0; // G / MaxAccel
                const maxAllowed = hoverThrottle * 0.95; // Ensure net accel is Downward
                if (finalThrottle > maxAllowed) {
                    finalThrottle = maxAllowed;
                }
            }

            r.throttle = finalThrottle;
        }

        // 1. Angle Stabilization (Gimbal)
        // Use Guidance Target if available
        const targetAngle = (g && g.targetAngle !== undefined) ? g.targetAngle : 0;
        const gimbalCmd = GimbalPID.update(targetAngle, r.angle, dt);
        r.engineGimbal = gimbalCmd;

        // 2. Angular Velocity Damping (RCS)
        // Goal: Angular Velocity = 0
        // Deadband: Only fire if > threshold
        const deadband = 0.02;
        r.rcsLeft = false;
        r.rcsRight = false;

        if (r.angularVelocity > deadband) {
            // Spinning CW (Positive) -> Need CCW Torque (Left RCS pushes Right? No, rcsRight creates +Torque, rcsLeft creates -Torque).
            // Checks physics.js:
            // rcsLeft -> -Torque (CCW)
            // rcsRight -> +Torque (CW)
            // If spin > 0 (CW), we need -Torque -> rcsLeft
            r.rcsLeft = true;
        } else if (r.angularVelocity < -deadband) {
            // Spinning CCW (Negative) -> Need +Torque -> rcsRight
            r.rcsRight = true;
        }

        // Velocity Hold logic moved to top (Engine Check)
    },

    reset() {
        GimbalPID.reset();
        ThrottlePID.reset();
    }
};
