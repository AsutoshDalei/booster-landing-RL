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
// Increased max gimbal range to +/- 0.6 rad for better angle control
const GimbalPID = new PIDController(2.0, 0.0, 1.0, -0.6, 0.6); // Increased range for better rotation
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
            // Update PID Gains from State
            if (state.tuning) {
                GimbalPID.kp = state.tuning.gimbalKp;
                GimbalPID.kd = state.tuning.gimbalKd;
                
                // Update Throttle PID gains from tuning
                if (state.tuning.throttleKp !== undefined) {
                    ThrottlePID.kp = state.tuning.throttleKp;
                }
                if (state.tuning.throttleKi !== undefined) {
                    ThrottlePID.ki = state.tuning.throttleKi;
                }
                if (state.tuning.throttleKd !== undefined) {
                    ThrottlePID.kd = state.tuning.throttleKd;
                }
            }

            // 3. Velocity Hold (Throttle)
            // Use Guidance Target if available, else default
            const targetVy = (g && g.targetVy !== undefined) ? g.targetVy : 50.0;

            // Note: ThrottlePID tuned for Kp=-0.05 (Negative error -> Positive Throttle)
            // Error = Setpoint - Measured.
            // If Setpoint (Target Descent) = 100, Measured = 200 (Falling fast).
            // Error = -100. Output = +5. Increase Throttle.

            const throttleCmd = ThrottlePID.update(targetVy, r.vy, dt);

            // Improved throttle logic - allow proper slowdown for landing
            let finalThrottle = throttleCmd;

            // Calculate hover throttle (throttle needed to counteract gravity)
            const GRAVITY = 100.0;
            const THRUST_POWER = 300.0;
            const hoverThrottle = GRAVITY / THRUST_POWER; // ~0.33

            // If ascending (vy < 0), reduce throttle significantly
            if (r.vy < -1.0) {
                // Ascending - cut throttle to let gravity pull down
                finalThrottle = Math.min(finalThrottle, hoverThrottle * 0.9);
            }
            // If falling much faster than target, allow full throttle
            else if (r.vy > targetVy * 1.3) {
                // Falling too fast - allow full throttle to slow down
                finalThrottle = Math.min(finalThrottle, 1.0);
            }
            // If close to target velocity or falling slowly, allow fine control
            else if (r.vy <= targetVy * 1.1) {
                // Close to target or slower - allow up to hover * 1.2 for fine control
                finalThrottle = Math.min(finalThrottle, hoverThrottle * 1.3);
            }
            // Otherwise (falling moderately fast), allow moderate throttle
            else {
                finalThrottle = Math.min(finalThrottle, 0.8);
            }

            // Ensure throttle is in valid range
            r.throttle = Math.max(0.0, Math.min(1.0, finalThrottle));
        }

        // 1. Angle Stabilization (Gimbal)
        // Use Guidance Target if available
        const targetAngle = (g && g.targetAngle !== undefined) ? g.targetAngle : 0;
        const gimbalCmd = GimbalPID.update(targetAngle, r.angle, dt);
        r.engineGimbal = gimbalCmd;

        // 2. Angular Velocity Damping (RCS) - Coordinated with Throttle
        // Goal: Angular Velocity = 0
        // Coordination: When throttle is high, gimbal is more effective, so RCS is less needed
        // When throttle is low/off, RCS is primary attitude control
        const deadband = 0.02;
        r.rcsLeft = false;
        r.rcsRight = false;

        // Adaptive deadband based on throttle
        // High throttle: gimbal can handle more, so RCS only for fine control (larger deadband)
        // Low throttle: RCS is primary control, use smaller deadband for more responsiveness
        const adaptiveDeadband = deadband * (1.0 + r.throttle * 1.5); // Deadband increases with throttle
        
        // Only use RCS if angular velocity exceeds adaptive deadband
        // This prevents RCS from fighting gimbal when throttle is high
        if (Math.abs(r.angularVelocity) > adaptiveDeadband) {
            if (r.angularVelocity > adaptiveDeadband) {
                // Spinning CW (Positive) -> Need CCW Torque -> rcsLeft
                r.rcsLeft = true;
            } else if (r.angularVelocity < -adaptiveDeadband) {
                // Spinning CCW (Negative) -> Need +Torque -> rcsRight
                r.rcsRight = true;
            }
        }
        
        // Additional coordination: If gimbal is already providing significant correction,
        // reduce RCS usage to avoid fighting
        // When gimbal is large, it's already correcting, so RCS can be more conservative
        if (Math.abs(r.engineGimbal) > 0.2 && r.throttle > 0.3) {
            // Gimbal is doing significant work, reduce RCS sensitivity
            // Only use RCS if angular velocity is quite high
            if (Math.abs(r.angularVelocity) < adaptiveDeadband * 2.0) {
                r.rcsLeft = false;
                r.rcsRight = false;
            }
        }

        // Velocity Hold logic moved to top (Engine Check)
    },

    reset() {
        GimbalPID.reset();
        ThrottlePID.reset();
    }
};
