/**
 * main.js
 * Entry point. Orchestrates the render loop and state updates.
 * Step 1: Integrated State & Visuals (Mock Physics)
 */

import { SimState, resetRocket } from './state.js';
import * as keyRender from './render.js'; // naming to avoid collision if needed, but 'render' is fine

import { stepPhysics } from './physics.js';
import { initControls, updateControls } from './controls.js';
import { Autopilot } from './autopilot.js';
import { Guidance } from './guidance.js';
import { initUI, updateUI, initTuning, initOptimizer } from './ui.js';
import { ParameterOptimizer, setCanvasDimensions } from './paramOptimizer.js';

const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');

// --- Initialization ---
function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    resetRocket(SimState, canvas.width, canvas.height);
    setCanvasDimensions(canvas.width, canvas.height);
}

function resetGame() {
    resetRocket(SimState, canvas.width, canvas.height);
    Autopilot.reset();
}

window.addEventListener('resize', resize);
resize(); // Initial setup
initControls(SimState, resetGame);
initUI();
initTuning(SimState);

// Initialize parameter optimizer
// State factory: creates a deep copy of the state structure
function createStateCopy() {
    const copy = JSON.parse(JSON.stringify(SimState));
    // Ensure all nested objects are properly initialized
    copy.rocket = { ...SimState.rocket };
    copy.pad = { ...SimState.pad };
    copy.guidance = { ...SimState.guidance };
    copy.tuning = { ...SimState.tuning };
    copy.particles = [];
    return copy;
}

const optimizer = new ParameterOptimizer(
    createStateCopy,
    stepPhysics,
    Guidance,
    Autopilot,
    resetRocket
);

// Update canvas dimensions when window resizes
function updateOptimizerDimensions() {
    setCanvasDimensions(canvas.width, canvas.height);
}
updateOptimizerDimensions();
window.addEventListener('resize', updateOptimizerDimensions);

initOptimizer(SimState, optimizer);

// --- Physics Loop ---
let lastTime = 0;
const FIXED_DT = 1.0 / 60.0; // Fixed physics step
let accumulator = 0;

function loop(timestamp) {
    if (!lastTime) lastTime = timestamp;
    const dt = (timestamp - lastTime) / 1000; // Seconds
    lastTime = timestamp;

    // Fixed timestep integration for stability
    accumulator += dt;
    while (accumulator >= FIXED_DT) {
        updateControls(SimState, FIXED_DT);
        Guidance.update(SimState, FIXED_DT);
        Autopilot.update(SimState, FIXED_DT);
        stepPhysics(SimState, FIXED_DT);
        // Particle update (visuals can run at physics rate or frame rate, let's keep it simple here)
        updateParticles(FIXED_DT);
        accumulator -= FIXED_DT;
    }

    keyRender.drawScene(ctx, SimState, canvas.width, canvas.height);
    updateUI(SimState);

    requestAnimationFrame(loop);
}

function updateParticles(dt) {
    const r = SimState.rocket;
    // Spawn Particles if Throttling
    if (r.throttle > 0.01) {
        const cosA = Math.cos(r.angle);
        const sinA = Math.sin(r.angle);

        // Nozzle position (approximate)
        const baseX = r.x + sinA * (r.height / 2);
        const baseY = r.y + (-cosA) * (r.height / 2); // Wait, if angle 0 (Up), cos(0)=1. Bottom is +Y relative to center?
        // Standard: Y is Down. Angle 0 is Up.
        // Rocket Center (0,0). Bottom is (0, +height/2).
        // Rotated by Angle:
        // X' = x*cos - y*sin ?? No.
        // Vector (0, h/2). Rotated:
        // x_rot = - (h/2) * sin(angle)
        // y_rot = (h/2) * cos(angle)
        // Let's trust the logic: bottom is "behind" the specific direction vector.

        // Actually, simplest: Center + Vector(Down).rotated(angle)
        // Vector Down is (0, 1).
        // Rotated(theta): ( -sin(theta), cos(theta) ) ... wait, standard rotation matrix:
        // [cos -sin]
        // [sin  cos]
        // * [0, 1] = [-sin, cos].
        // So nozzle = center + (h/2)*[-sin(a), cos(a)].

        const nozzleX = r.x - (r.height / 2) * Math.sin(r.angle);
        const nozzleY = r.y + (r.height / 2) * Math.cos(r.angle);

        SimState.particles.push({
            x: nozzleX + (Math.random() - 0.5) * 5,
            y: nozzleY,
            vx: r.vx + (Math.random() - 0.5) * 50 - Math.sin(r.angle) * 100, // Eject backward
            vy: r.vy + (Math.random() * 50) + Math.cos(r.angle) * 100, // Eject downward
            life: 1.0,
            decay: 2.0 * dt, // Fast decay
            size: 3 + Math.random() * 5
        });
    }

    // Update Existing Particles
    for (let i = SimState.particles.length - 1; i >= 0; i--) {
        const p = SimState.particles[i];
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.life -= p.decay;
        if (p.life <= 0) {
            SimState.particles.splice(i, 1);
        }
    }
}

// Start
requestAnimationFrame(loop);
