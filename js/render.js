/**
 * render.js
 * Pure rendering logic. Decoupled from physics and state management.
 */

const COLORS = {
    bg: '#0b0d10',
    ground: '#1a1d23',
    pad: '#4CAF50',
    rocket: '#e0e0e0',
    leg: '#888888',
    legContact: '#00ff00', // Green when touching
    flameCore: '#ffffff',
    flameCore: '#ffffff',
    flameOuter: '#ff9900'
};

const VISUAL_SCALE = 0.6;

export function drawScene(ctx, state, canvasWidth, canvasHeight) {
    // 1. Clear Screen
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // 2. Draw Ground
    ctx.fillStyle = COLORS.ground;
    ctx.fillRect(0, canvasHeight - 40, canvasWidth, 40);

    // 3. Draw Pad
    drawPad(ctx, state.pad);

    // 4. Draw Particles (Behind rocket)
    drawParticles(ctx, state.particles);

    // 5. Draw Rocket
    drawRocket(ctx, state.rocket);
}

function drawPad(ctx, pad) {
    ctx.save();
    ctx.translate(pad.x, pad.y);
    ctx.fillStyle = COLORS.pad;
    ctx.fillRect(-pad.width / 2, -pad.height / 2, pad.width, pad.height);
    ctx.restore();
}

function drawRocket(ctx, rocket) {
    ctx.save();
    ctx.translate(rocket.x, rocket.y);
    ctx.scale(VISUAL_SCALE, VISUAL_SCALE); // Scale down visual representation
    ctx.rotate(rocket.angle);

    // -- Draw Legs --
    // Simple deployed legs visualization
    ctx.save();
    const legW = 5;
    const legH = 25;
    const legOffset = rocket.width / 2;

    // Leg 1 (Left)
    ctx.translate(-legOffset, rocket.height / 2 - 5);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = rocket.leg1Contact ? COLORS.legContact : COLORS.leg;
    ctx.fillRect(-legW / 2, 0, legW, legH);
    ctx.restore();

    // Leg 2 (Right)
    ctx.save();
    ctx.translate(legOffset, rocket.height / 2 - 5);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = rocket.leg2Contact ? COLORS.legContact : COLORS.leg;
    ctx.fillRect(-legW / 2, 0, legW, legH);
    ctx.restore();

    // -- Draw Main Body (Falcon Style) --
    // Body is White
    ctx.fillStyle = COLORS.rocket;
    ctx.fillRect(-rocket.width / 2, -rocket.height / 2, rocket.width, rocket.height);

    // Interstage (Black Band at Top)
    ctx.fillStyle = '#000000';
    const interstageHeight = 15;
    ctx.fillRect(-rocket.width / 2, -rocket.height / 2, rocket.width, interstageHeight);

    // Engine Bell (Dark Grey at Bottom)
    ctx.fillStyle = '#333333';
    const bellHeight = 8;
    ctx.fillRect(-rocket.width / 2, rocket.height / 2 - bellHeight, rocket.width, bellHeight);

    // -- Draw Gimballed Engine --
    drawEngine(ctx, rocket);

    // Orientation Marker (Top)
    ctx.fillStyle = 'red';
    ctx.fillRect(-2, -rocket.height / 2, 4, 10);

    // -- Draw RCS Puffs --
    if (rocket.rcsLeft) {
        // Firing Right Thruster to push Left
        // Draw puff on Top Right of rocket
        drawRCSPuff(ctx, rocket.width / 2, -rocket.height / 2 + 10, 'right');
    }
    if (rocket.rcsRight) {
        // Firing Left Thruster to push Right
        drawRCSPuff(ctx, -rocket.width / 2, -rocket.height / 2 + 10, 'left');
    }

    ctx.restore();
}

function drawRCSPuff(ctx, x, y, direction) {
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = 'white';

    // Simple Triangle Puff
    ctx.beginPath();
    if (direction === 'right') {
        ctx.moveTo(0, 0);
        ctx.lineTo(10, -5);
        ctx.lineTo(10, 5);
    } else {
        ctx.moveTo(0, 0);
        ctx.lineTo(-10, -5);
        ctx.lineTo(-10, 5);
    }
    ctx.fill();
    ctx.restore();
}

function drawEngine(ctx, rocket) {
    if (rocket.throttle <= 0.01) return;

    ctx.save();
    // Move to bottom center of rocket
    ctx.translate(0, rocket.height / 2);
    // Rotate by Gimbal angle
    ctx.rotate(rocket.engineGimbal);

    // Flame length scales with throttle
    const flameLen = 40 + (rocket.throttle * 80) + (Math.random() * 10);
    const flameW = 15;

    // Draw Flame
    ctx.beginPath();
    ctx.moveTo(-flameW / 2, 0);
    ctx.lineTo(0, flameLen);
    ctx.lineTo(flameW / 2, 0);
    ctx.closePath();

    // Gradient
    const grad = ctx.createLinearGradient(0, 0, 0, flameLen);
    grad.addColorStop(0, COLORS.flameOuter);
    grad.addColorStop(0.3, COLORS.flameOuter);
    grad.addColorStop(1, 'rgba(255, 100, 0, 0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Core
    ctx.beginPath();
    ctx.moveTo(-flameW / 4, 0);
    ctx.lineTo(0, flameLen * 0.6);
    ctx.lineTo(flameW / 4, 0);
    ctx.fillStyle = COLORS.flameCore;
    ctx.fill();

    ctx.restore();
}

function drawParticles(ctx, particles) {
    particles.forEach(p => {
        ctx.globalAlpha = p.life;
        ctx.fillStyle = '#aaaaaa';
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.globalAlpha = 1.0;
}
