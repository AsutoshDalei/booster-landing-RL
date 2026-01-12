# RL Training Script Parameters Explanation

This document explains all parameters in `train_falcon_rl.py` used for training the rocket landing agent.

## 1. Hyperparameters (Training Configuration)

### `LR = 0.0003`
- **Learning Rate**: Controls how fast the neural network learns
- Lower = slower but more stable learning
- Higher = faster but may overshoot optimal values
- **Typical range**: 0.0001 to 0.001

### `GAMMA = 0.99`
- **Discount Factor**: How much the agent values future rewards vs immediate rewards
- 0.99 = agent cares about rewards 100 steps in the future (99% of value)
- Closer to 1.0 = more long-term planning
- **Typical range**: 0.9 to 0.999

### `K_EPOCHS = 4`
- **PPO Update Epochs**: How many times to update the policy on the same batch of data
- More epochs = better use of collected data, but risk of overfitting
- **Typical range**: 3 to 10

### `BATCH_SIZE = 128`
- **Batch Size**: Not directly used in PPO, but affects memory usage
- Larger = more stable gradients but more memory
- **Note**: PPO uses all collected data in each update (rollout_len)

### `HIDDEN_DIM = 64`
- **Hidden Layer Size**: Number of neurons in each hidden layer of the neural network
- Larger = more capacity to learn complex policies, but slower training
- **Typical range**: 32 to 256

### `ACTION_STD = 0.5`
- **Action Standard Deviation**: Initial exploration noise (learned during training)
- Higher = more exploration initially
- The network learns the optimal std via `log_std` parameter

## 2. Physics Constants (Matches JS Implementation)

### `GRAVITY = 100.0`
- **Gravitational Acceleration**: Downward acceleration in px/s²
- Higher = rocket falls faster
- Matches JS: `GRAVITY = 100.0`

### `THRUST_POWER = 300.0`
- **Maximum Thrust Acceleration**: Maximum upward acceleration in px/s²
- Higher = more powerful engine
- Thrust-to-Weight Ratio (TWR) ≈ 3.0 (300/100)
- Matches JS: `THRUST_POWER = 300.0`

### `MASS = 1.0`
- **Rocket Mass**: Normalized mass (affects acceleration calculations)
- Used in F = ma calculations
- Matches JS: `MASS = 1.0`

### `MOMENT_OF_INERTIA = 1000.0`
- **Rotational Inertia**: Resistance to rotation
- Higher = harder to rotate (more stable)
- Lower = easier to rotate (more agile)
- Matches JS: `MOMENT_OF_INERTIA = 1000.0`

### `DRAG_COEFFICIENT = 0.05`
- **Air Resistance**: Linear drag coefficient
- Higher = more air resistance (slower movement)
- Matches JS: `DRAG_COEFFICIENT = 0.05`

### `ANGULAR_DRAG = 2.0`
- **Rotational Damping**: Reduces angular velocity over time
- Higher = rotation slows down faster
- Matches JS: `ANGULAR_DRAG = 2.0`

### `RCS_THRUST = 100.0`
- **RCS Force**: Force applied by Reaction Control System thrusters
- Applied at top of rocket (86% of height)
- Creates both rotation and horizontal translation
- **Updated**: Changed from 50.0 to 100.0 for force-based application
- Matches JS: `RCS_THRUST = 100.0`

### `MIN_THROTTLE = 0.4`
- **Minimum Throttle**: When engine is on, minimum power is 40%
- Realistic rocket behavior - engines can't throttle below minimum
- Actual power = MIN_THROTTLE + throttle * (1 - MIN_THROTTLE)
- **New**: Added to match JS implementation
- Matches JS: `MIN_THROTTLE = 0.4`

### `FRICTION = 0.5`
- **Ground Friction**: Reduces horizontal velocity on landing
- 0.5 = 50% of horizontal velocity lost on impact
- Matches JS: `FRICTION = 0.5`

### `RESTITUTION = 0.0`
- **Bounciness**: Energy retained after collision
- 0.0 = no bounce (rocket stops on impact)
- Matches JS: `RESTITUTION = 0.0`

### `STOP_THRESHOLD = 0.5`
- **Landing Velocity Threshold**: Below this speed, rocket stops (no bounce)
- In px/s
- Matches JS: `STOP_THRESHOLD = 0.5`

## 3. Simulation Constants

### `DT = 1 / 60.0`
- **Time Step**: Simulation timestep in seconds
- 60 FPS = 0.0167 seconds per step
- Matches JS: Fixed timestep physics

### `CANVAS_WIDTH = 800`
- **Canvas Width**: World width in pixels
- Matches JS canvas dimensions

### `CANVAS_HEIGHT = 600`
- **Canvas Height**: World height in pixels
- Matches JS canvas dimensions

### `PAD_X = 400`
- **Landing Pad X Position**: Center of landing pad (half of canvas width)
- Target landing location

### `PAD_Y = CANVAS_HEIGHT - 50`
- **Landing Pad Y Position**: Near bottom of canvas
- Target landing location

### `PAD_WIDTH = 120`
- **Landing Pad Width**: Width of landing zone in pixels
- Rocket must land within this width for success

### `ROCKET_WIDTH = 24`
- **Rocket Width**: Visual width of rocket
- Matches JS: `ROCKET_WIDTH = 24`

### `ROCKET_HEIGHT = 100`
- **Rocket Height**: Visual height of rocket
- **Updated**: Changed from 80 to 100 to match JS
- Matches JS: `ROCKET_HEIGHT = 100`

## 4. PPO Training Parameters

### `max_episodes = 500`
- **Total Training Episodes**: How many episodes to train
- Each episode = one landing attempt (success or failure)
- More episodes = better policy (but longer training)

### `rollout_len = 2048`
- **Rollout Length**: Number of steps to collect before updating policy
- Larger = more data per update, more stable learning
- **Typical range**: 512 to 4096

### `clip_eps = 0.2`
- **PPO Clip Epsilon**: Maximum allowed policy change per update
- Prevents large policy updates that could destabilize training
- **Typical range**: 0.1 to 0.3

### `value_coef = 0.5`
- **Value Function Coefficient**: Weight for value function loss
- Balances policy improvement vs value estimation accuracy
- **Typical range**: 0.1 to 1.0

### `entropy_coef = 0.01`
- **Entropy Coefficient**: Encourages exploration
- Higher = more exploration, prevents premature convergence
- **Typical range**: 0.001 to 0.1

## 5. Action Space

The agent outputs 3 continuous actions:

### `throttle` [0, 1]
- **Main Engine Throttle**: 0 = off, 1 = full power
- With MIN_THROTTLE: actual power = 0.4 + throttle * 0.6
- Controls vertical velocity

### `gimbal` [-1, 1]
- **Engine Gimbal**: -1 = max left, +1 = max right
- Scaled to [-0.6, 0.6] radians (updated from 0.25)
- Controls rotation via thrust vectoring

### `rcs` [-1, 1]
- **RCS Thrusters**: -1 = left thruster, +1 = right thruster
- Deadband: only active if |rcs| > 0.3
- Applied as force at top of rocket (86% height)
- Controls both rotation and horizontal translation

## 6. Observation Space (State)

The agent receives 6 normalized observations:

### `(x - PAD_X) / 400.0`
- **Horizontal Position**: Distance from pad center, normalized
- Negative = left of pad, Positive = right of pad
- Range: approximately [-1, 1]

### `(PAD_Y - y) / 600.0`
- **Altitude**: Height above pad, normalized
- Higher = further from pad
- Range: approximately [0, 1]

### `vx / 100.0`
- **Horizontal Velocity**: Normalized horizontal speed
- Positive = moving right, Negative = moving left
- Range: approximately [-2, 2] (clipped)

### `vy / 100.0`
- **Vertical Velocity**: Normalized vertical speed
- Positive = descending, Negative = ascending
- Range: approximately [-1, 2] (clipped)

### `angle / 3.14`
- **Rocket Angle**: Normalized angle in radians
- 0 = upright, ±1 = ~180 degrees
- Range: approximately [-1, 1]

### `angularVelocity / 10.0`
- **Angular Velocity**: Normalized rotation speed
- Positive = clockwise, Negative = counterclockwise
- Range: approximately [-2, 2] (clipped)

## 7. Reward Function Parameters

### Success/Failure Rewards
- **Success**: +100.0 (landed safely on pad)
- **Failure**: -100.0 (crashed, missed pad, or out of bounds)

### Distance Penalty: `-dist_to_pad * 0.005`
- Penalizes being far from landing pad
- Encourages horizontal alignment

### Angle Penalty: `-angle_error * angle_weight`
- Penalizes deviation from target angle
- `angle_weight = 2.0` when altitude < 15m (stronger when close)
- `angle_weight = 0.5` when altitude >= 15m
- Encourages proper orientation

### Velocity Penalty: `-abs(vy) * 0.01 * multiplier`
- Penalizes high vertical velocity
- Multiplier = 2.0 when altitude < 30m (stronger when close)
- Encourages slow, controlled descent

### Step Penalty: `-0.05`
- Small penalty per step
- Encourages efficient (fast) landings

## 8. Guidance-Based Target Angle (Reward Shaping)

The reward function uses guidance logic to compute target angles:

### High Altitude (>30m)
- `K_STEER = 0.004` (updated from 0.003)
- `MAX_TILT = 0.3` (updated from 0.25)
- Allows aggressive steering to reach pad

### Medium Altitude (15-30m)
- `K_STEER = 0.002`
- `MAX_TILT = 0.15`
- Reduces steering, starts straightening

### Low Altitude (<15m)
- `K_STEER = 0.001`
- `MAX_TILT = 0.08`
- Prioritizes being upright over horizontal position

## Summary of Recent Updates

1. **MIN_THROTTLE = 0.4**: Added minimum throttle for realistic engine behavior
2. **RCS_THRUST = 100.0**: Updated from 50.0 for force-based application
3. **Gimbal Range = 0.6 rad**: Updated from 0.25 rad for better angle control
4. **Torque Multiplier = 1.5x**: Increased gimbal torque for better responsiveness
5. **RCS Force Application**: Changed from direct torque to force at thruster position
6. **Updated Steering Gains**: K_STEER = 0.004, MAX_TILT = 0.3 for high altitude
7. **ROCKET_HEIGHT = 100**: Updated from 80 to match JS

All parameters now match the JavaScript implementation for consistent physics simulation.
