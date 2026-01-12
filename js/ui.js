/**
 * ui.js
 * Handles the HUD / UI updates based on Simulation State.
 */

const UI = {
    alt: null,
    spd: null,
    fuel: null,
    ang: null,
    apStatus: null,
    landingOutcome: null,
    throttleBar: null,
    legLeft: null,
    legRight: null,
    gimbalArrow: null
};

export function initUI() {
    UI.alt = document.getElementById('alt-val');
    UI.spd = document.getElementById('spd-val');
    UI.fuel = document.getElementById('fuel-val');
    UI.ang = document.getElementById('ang-val');
    UI.apStatus = document.getElementById('autopilot-status');
    UI.landingOutcome = document.getElementById('landing-outcome');
    UI.throttleBar = document.getElementById('throttle-bar-fill');
    UI.legLeft = document.getElementById('leg-left');
    UI.legRight = document.getElementById('leg-right');
    UI.gimbalArrow = document.getElementById('gimbal-arrow');
}

export function initTuning(state) {
    const ids = [
        { sl: 'inp-ign', val: 'val-ign', key: 'ignitionAltitude' },
        { sl: 'inp-kp', val: 'val-kp', key: 'gimbalKp' },
        { sl: 'inp-kd', val: 'val-kd', key: 'gimbalKd' }
    ];

    ids.forEach(item => {
        const slider = document.getElementById(item.sl);
        const label = document.getElementById(item.val);

        if (slider && label) {
            // Set Initial
            slider.value = state.tuning[item.key];
            label.innerText = state.tuning[item.key];

            // Listen
            slider.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                state.tuning[item.key] = val;
                label.innerText = val;
            });
        }
    });
}

const GENERATIONS = 8;
const POPULATION_SIZE = 30;

export function initOptimizer(state, optimizerInstance) {
    const optimizeBtn = document.getElementById('optimize-btn');
    const stopBtn = document.getElementById('stop-optimize-btn');
    const statusEl = document.getElementById('optimizer-status');
    const progressBar = document.getElementById('optimizer-progress-bar');
    const resultsEl = document.getElementById('optimizer-results');

    if (!optimizeBtn || !stopBtn || !statusEl || !progressBar || !resultsEl) {
        console.warn('Optimizer UI elements not found');
        return;
    }

    let isRunning = false;

    optimizeBtn.addEventListener('click', async () => {
        if (isRunning) return;
        
        isRunning = true;
        optimizeBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        statusEl.textContent = 'Initializing...';
        progressBar.style.width = '0%';
        resultsEl.innerHTML = '';

        try {
            optimizerInstance.onProgress = (info) => {
                statusEl.textContent = `Gen ${info.generation}/${GENERATIONS}, Ind ${info.individual}/${POPULATION_SIZE}`;
                progressBar.style.width = `${info.progress}%`;
                
                if (info.bestParams) {
                    resultsEl.innerHTML = `
                        <strong>Best so far:</strong><br>
                        Ign: ${info.bestParams.ignitionAltitude.toFixed(0)} | 
                        Kp: ${info.bestParams.gimbalKp.toFixed(2)} | 
                        Kd: ${info.bestParams.gimbalKd.toFixed(2)}<br>
                        Throttle: Kp=${info.bestParams.throttleKp.toFixed(3)}, 
                        Ki=${info.bestParams.throttleKi.toFixed(3)}, 
                        Kd=${info.bestParams.throttleKd.toFixed(3)}<br>
                        Fitness: ${info.bestFitness.toFixed(1)}
                    `;
                }
            };

            const bestParams = await optimizerInstance.evolve(3);

            if (bestParams) {
                // Apply best parameters to state
                state.tuning.ignitionAltitude = bestParams.ignitionAltitude;
                state.tuning.gimbalKp = bestParams.gimbalKp;
                state.tuning.gimbalKd = bestParams.gimbalKd;
                state.tuning.throttleKp = bestParams.throttleKp;
                state.tuning.throttleKi = bestParams.throttleKi;
                state.tuning.throttleKd = bestParams.throttleKd;

                // Update UI sliders
                document.getElementById('inp-ign').value = bestParams.ignitionAltitude;
                document.getElementById('val-ign').textContent = bestParams.ignitionAltitude.toFixed(0);
                document.getElementById('inp-kp').value = bestParams.gimbalKp;
                document.getElementById('val-kp').textContent = bestParams.gimbalKp.toFixed(2);
                document.getElementById('inp-kd').value = bestParams.gimbalKd;
                document.getElementById('val-kd').textContent = bestParams.gimbalKd.toFixed(2);

                statusEl.textContent = 'Optimization complete!';
                progressBar.style.width = '100%';
                resultsEl.innerHTML = `
                    <strong>Optimal Parameters Found:</strong><br>
                    Ignition Altitude: ${bestParams.ignitionAltitude.toFixed(0)}m<br>
                    Gimbal Kp: ${bestParams.gimbalKp.toFixed(2)}<br>
                    Gimbal Kd: ${bestParams.gimbalKd.toFixed(2)}<br>
                    Throttle Kp: ${bestParams.throttleKp.toFixed(3)}<br>
                    Throttle Ki: ${bestParams.throttleKi.toFixed(3)}<br>
                    Throttle Kd: ${bestParams.throttleKd.toFixed(3)}<br>
                    <strong>Applied to simulation!</strong>
                `;
            } else {
                statusEl.textContent = 'Optimization failed';
            }
        } catch (error) {
            console.error('Optimization error:', error);
            statusEl.textContent = 'Error: ' + error.message;
        } finally {
            isRunning = false;
            optimizeBtn.style.display = 'block';
            stopBtn.style.display = 'none';
        }
    });

    stopBtn.addEventListener('click', () => {
        optimizerInstance.stop();
        statusEl.textContent = 'Stopped by user';
        isRunning = false;
        optimizeBtn.style.display = 'block';
        stopBtn.style.display = 'none';
    });
}

export function updateUI(state) {
    const r = state.rocket;
    const padY = state.pad.y; // Assumed ground level

    // Altitude (from pad surface)
    const rocketBottom = r.y + r.height / 2;
    const padTop = padY - state.pad.height / 2;
    const altitude = Math.max(0, padTop - rocketBottom);

    if (UI.alt) UI.alt.innerText = altitude.toFixed(1);

    // Speed (Positive = Descent)
    if (UI.spd) UI.spd.innerText = r.vy.toFixed(1);

    // Angle (Degrees)
    const angleDeg = (r.angle * 180 / Math.PI);
    if (UI.ang) UI.ang.innerText = angleDeg.toFixed(1);

    // Fuel (Mock)
    if (UI.fuel) UI.fuel.innerText = "100";

    // Throttle Bar
    if (UI.throttleBar) {
        const pct = (r.throttle * 100).toFixed(0);
        UI.throttleBar.style.height = `${pct}%`;
    }

    // Autopilot Status
    if (UI.apStatus) {
        if (state.autopilotEnabled) {
            UI.apStatus.innerText = "ON";
            UI.apStatus.className = "hud-value status-on";
        } else {
            UI.apStatus.innerText = "OFF";
            UI.apStatus.className = "hud-value status-off";
        }
    }

    // Leg Indicators
    if (UI.legLeft) {
        UI.legLeft.className = r.leg1Contact ? "leg-light leg-active" : "leg-light";
    }
    if (UI.legRight) {
        UI.legRight.className = r.leg2Contact ? "leg-light leg-active" : "leg-light";
    }

    // Gimbal Visual
    // Apply negative rotation because Gimbal angle is relative to Rocket, but visual arrow acts like the nozzle vector.
    // If Gimbal is +0.1 rad (points right), we rotate arrow right (+).
    // Just need to verify CSS transform origin.
    if (UI.gimbalArrow) {
        const gimbalDeg = r.engineGimbal * 180 / Math.PI;
        // Exaggerate for visibility (x2)
        UI.gimbalArrow.style.transform = `rotate(${-gimbalDeg * 2}deg)`;
    }

    // Landing Outcome
    if (UI.landingOutcome) {
        UI.landingOutcome.style.display = 'block';
        if (state.guidance && state.guidance.landingResult) {
            if (state.guidance.landingResult === 'SUCCESS') {
                UI.landingOutcome.innerText = "STATUS: LANDED (SUCCESS)";
                UI.landingOutcome.style.color = '#a3be8c'; // Green
                UI.landingOutcome.style.borderLeft = '4px solid #a3be8c';
            } else {
                UI.landingOutcome.innerText = "STATUS: FAILED";
                UI.landingOutcome.style.color = '#bf616a'; // Red
                UI.landingOutcome.style.borderLeft = '4px solid #bf616a';
            }
        } else {
            UI.landingOutcome.innerText = "STATUS: FLIGHT";
        }
    }
}
