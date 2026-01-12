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
