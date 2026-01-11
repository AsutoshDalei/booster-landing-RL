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
    throttleBar: null
};

export function initUI() {
    UI.alt = document.getElementById('alt-val');
    UI.spd = document.getElementById('spd-val');
    UI.fuel = document.getElementById('fuel-val');
    UI.ang = document.getElementById('ang-val');
    UI.apStatus = document.getElementById('autopilot-status');
    UI.landingOutcome = document.getElementById('landing-outcome');
    UI.throttleBar = document.getElementById('throttle-bar-fill');
}

export function updateUI(state) {
    const r = state.rocket;
    const padY = state.pad.y; // Assumed ground level

    // Altitude: Distance from bottom of rocket to pad top
    // Rocket Y is center. Bottom is Y + Height/2.
    // Pad Y is Center. Top is Y - Height/2.
    // But Pad is drawn at Pad Y. Let's assume Pad Y is Surface Y.
    // Actually in render.js/main.js logic:
    // Pad Top = pad.y - pad.height/2.
    // Rocket Bottom = rocket.y + rocket.height/2.
    // Alt = Pad Top - Rocket Bottom.
    // Note: Y increases Down. So Pad Top > Rocket Bottom when above.
    // Distance = PadTop - RocketBottom.

    const padTop = state.pad.y - state.pad.height / 2;
    const rocketBottom = r.y + r.height / 2;
    const altitude = Math.max(0, padTop - rocketBottom);

    // Velocity: Vy. Positive is Down (Falling). 
    // Display: Positive Up? Or Descent Rate?
    // Let's use standard aviation: +Climb, -Descend? 
    // Or Space: +Velocity = Speed?
    // Let's just show Vy inverted so + is Up (Ascent).
    const velocity = -r.vy;

    // Angle: Degrees.
    // r.angle is Radians.
    const deg = r.angle * (180 / Math.PI);

    // Update DOM
    if (UI.alt) UI.alt.innerText = (altitude / 10).toFixed(1); // Scale 10px = 1m approx
    if (UI.spd) UI.spd.innerText = (velocity / 10).toFixed(1);
    if (UI.fuel) UI.fuel.innerText = r.fuel.toFixed(0);
    if (UI.ang) UI.ang.innerText = deg.toFixed(1);

    // Autopilot Status
    if (UI.apStatus) {
        if (state.autopilotEnabled) {
            UI.apStatus.innerText = "AUTOPILOT: ON";
            UI.apStatus.className = "status-on";
        } else {
            UI.apStatus.innerText = "AUTOPILOT: OFF";
            UI.apStatus.className = "status-off";
        }
    }

    // Landing Outcome
    if (UI.landingOutcome) {
        if (state.guidance && state.guidance.landingResult) {
            if (state.guidance.landingResult === 'SUCCESS') {
                UI.landingOutcome.innerText = "Successful Landing";
                UI.landingOutcome.style.color = '#00ff00';
            } else {
                UI.landingOutcome.innerText = "Unsuccessful Landing";
                UI.landingOutcome.style.color = '#ff0000';
            }
        } else {
            UI.landingOutcome.innerText = "";
        }
    }

    // Throttle Bar
    if (UI.throttleBar) {
        UI.throttleBar.style.height = (r.throttle * 100) + '%';
        // Color Change: Orange to Red if high
        if (r.throttle > 0.9) UI.throttleBar.style.style = 'background-color: #ff0000';
        else UI.throttleBar.style.backgroundColor = '#ff9900';
    }
}
