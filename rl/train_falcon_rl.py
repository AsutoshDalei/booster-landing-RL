import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import math

# --- Hyperparameters ---
LR = 0.0003
GAMMA = 0.99
K_EPOCHS = 4
BATCH_SIZE = 128
HIDDEN_DIM = 64
ACTION_STD = 0.5

# --- Physics Constants (Ported from JS) ---
GRAVITY = 100.0
THRUST_POWER = 300.0
MASS = 1.0
MOMENT_OF_INERTIA = 1000.0
DRAG_COEFFICIENT = 0.05
ANGULAR_DRAG = 2.0
RCS_THRUST = 50.0  # Not directly used in simple step logic in JS, but handy
FRICTION = 0.5
RESTITUTION = 0.0
STOP_THRESHOLD = 0.5

# --- Sim Constants ---
DT = 1 / 60.0
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
PAD_X = 400
PAD_Y = CANVAS_HEIGHT - 50
PAD_WIDTH = 120
ROCKET_WIDTH = 24
ROCKET_HEIGHT = 80

class FalconEnv:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 3
        self.reset()

    def reset(self):
        # Result: [x, y, vx, vy, angle, angularVelocity, throttle, gimbal, fuel, groundContact, landingResult, done, stepCount]
        # Randomize "Anywhere"
        self.state = {
            'x': np.random.uniform(100.0, 700.0),    # Anywhere (avoid extreme edges)
            'y': np.random.uniform(50.0, 300.0),     # Start high-ish
            'vx': np.random.uniform(-40.0, 40.0),    # More drift
            'vy': np.random.uniform(0.0, 20.0),      # Initial downward velocity
            'angle': np.random.uniform(-0.5, 0.5),   # Random orientation
            'angularVelocity': np.random.uniform(-0.2, 0.2),
            'throttle': 0.0,
            'engineGimbal': 0.0,
            'fuel': 100.0,
            'groundContact': False,
            'landingResult': None,
            'done': False,
            'stepCount': 0
        }
        return self._get_obs()

    def step(self, action):
        # Action: [throttle (0-1), gimbal (-1 to 1), rcs (-1 to 1)]
        throttle, gimbal, rcs = action
        
        # Clamp Inputs
        throttle = np.clip(throttle, 0.0, 1.0)
        gimbal = np.clip(gimbal, -1.0, 1.0)
        rcs = np.clip(rcs, -1.0, 1.0)
        
        s = self.state
        s['throttle'] = throttle
        s['engineGimbal'] = gimbal * 0.25 # Max 0.25 rad

        # RCS Logic
        torque_input = 0.0
        if rcs < -0.3: torque_input = -10000.0
        elif rcs > 0.3: torque_input = 10000.0
        
        # --- Physics Update ---
        fx = 0.0
        fy = GRAVITY * MASS
        torque = torque_input
        
        # Thrust
        if s['throttle'] > 0:
            thrust_mag = s['throttle'] * THRUST_POWER * MASS
            thrust_angle = s['angle'] + s['engineGimbal']
            
            # Thrust pushes opposite to global angle
            # Standard visual: 0 is Up. thrust_angle is direction of thrust vector.
            # Wait, JS port: 
            # fx += Math.sin(thrustAngle) * thrustMag;
            # fy -= Math.cos(thrustAngle) * thrustMag;
            fx += math.sin(thrust_angle) * thrust_mag
            fy -= math.cos(thrust_angle) * thrust_mag
            
            # Torque
            lever_arm = ROCKET_HEIGHT / 2.0
            torque += -lever_arm * thrust_mag * math.sin(s['engineGimbal'])
            
        # Drag
        fx -= s['vx'] * DRAG_COEFFICIENT
        fy -= s['vy'] * DRAG_COEFFICIENT
        torque -= s['angularVelocity'] * ANGULAR_DRAG
        
        # Integration
        s['vx'] += (fx / MASS) * DT
        s['vy'] += (fy / MASS) * DT
        s['angularVelocity'] += (torque / MOMENT_OF_INERTIA) * DT
        
        s['x'] += s['vx'] * DT
        s['y'] += s['vy'] * DT
        s['angle'] += s['angularVelocity'] * DT
        
        # Collision
        ground_y = PAD_Y - 10 # Approx pad top
        rocket_bottom = s['y'] + ROCKET_HEIGHT / 2.0
        
        if rocket_bottom >= ground_y:
            s['groundContact'] = True
            s['y'] = ground_y - ROCKET_HEIGHT / 2.0
            
            if s['vy'] > 0:
                s['vx'] *= FRICTION
                s['angularVelocity'] *= 0.8
                if s['vy'] < STOP_THRESHOLD:
                    s['vy'] = 0.0
                else:
                    s['vy'] = -s['vy'] * RESTITUTION
                    
        # --- Outcome Detection ---
        # Out of bounds
        if s['x'] < 0 or s['x'] > CANVAS_WIDTH or s['y'] > CANVAS_HEIGHT + 100 or s['y'] < -500:
            s['landingResult'] = 'FAILURE'
            s['done'] = True
            
        # Landing Logic
        if s['groundContact'] and not s['done']:
            if abs(s['vy']) > 10.0:
                s['landingResult'] = 'FAILURE'
                s['done'] = True
            else:
                angle_deg = abs(s['angle'] * 180.0 / math.pi)
                is_upright = angle_deg < 5.0
                on_pad = abs(s['x'] - PAD_X) < (PAD_WIDTH / 2.0 + ROCKET_WIDTH)
                
                if not is_upright:
                    s['landingResult'] = 'FAILURE'
                    s['done'] = True
                elif not on_pad:
                    s['landingResult'] = 'FAILURE'
                    s['done'] = True
                else:
                    # Success condition
                    if abs(s['vy']) < 1.0 and abs(s['vx']) < 1.0:
                        s['landingResult'] = 'SUCCESS'
                        s['done'] = True
        
        # Timeout
        s['stepCount'] += 1
        if s['stepCount'] > 1000:
            s['landingResult'] = 'FAILURE'
            s['done'] = True
            
        reward = self._compute_reward()
        
        return self._get_obs(), reward, s['done'], {'result': s['landingResult']}

    def _get_obs(self):
        s = self.state
        return np.array([
            (s['x'] - PAD_X) / 400.0,
            (PAD_Y - s['y']) / 600.0,
            s['vx'] / 100.0,
            s['vy'] / 100.0,
            s['angle'] / 3.14,
            s['angularVelocity'] / 10.0
        ], dtype=np.float32)

    def _compute_reward(self):
        s = self.state
        if s['landingResult'] == 'FAILURE': return -100.0
        if s['landingResult'] == 'SUCCESS': return 100.0
        
        reward = 0.0
        
        # Distance Penalty (Normalized)
        dist_to_pad = abs(s['x'] - PAD_X)
        reward -= dist_to_pad * 0.005 # Reduced slightly
        
        # Guidance Reward (Encourage tilting towards pad)
        # Detailed logic: Calculate desired tilt to cancel X error
        # P-Controller for Angle: Target Angle ~ (Target X - Current X)
        target_angle = (PAD_X - s['x']) * 0.005
        target_angle = np.clip(target_angle, -0.5, 0.5) # limit to ~30 deg
        
        # If we overlap the pad, we want to be upright (target 0)
        # Shaping: Penalize deviation from target angle
        angle_error = abs(s['angle'] - target_angle)
        reward -= angle_error * 0.5
        
        # Velocity Penalty (Encourage slow landing)
        # But allow free fall at height
        # Only penalize if close to ground?
        # For now, simple damping
        reward -= abs(s['vy']) * 0.01
        
        # Step Penalty
        reward -= 0.05
        
        return reward

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )

        # Learnable log std (PPO standard)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, state):
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(-1)
        value = self.critic(state).squeeze(-1)

        return action, log_prob, value

    def evaluate(self, states, actions):
        mean = self.actor(states)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        # Inverse tanh for logprob
        eps = 1e-6
        raw_actions = torch.atanh(torch.clamp(actions, -1 + eps, 1 - eps))

        log_probs = dist.log_prob(raw_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.critic(states).squeeze(-1)

        return log_probs, values, entropy


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return torch.tensor(advantages), torch.tensor(returns)


# --- PPO Training Loop ---
def train():
    env = FalconEnv()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Training on device: {device}")

    policy = ActorCritic(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    max_episodes = 500
    rollout_len = 2048
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    state = env.reset()

    for episode in range(1, max_episodes + 1):

        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []

        ep_reward = 0

        # -------- Rollout --------
        for _ in range(rollout_len):
            state_t = torch.tensor(state, dtype=torch.float32).to(device)

            with torch.no_grad():
                action, logprob, value = policy.act(state_t)

            # Rescale actions
            a = action.cpu().numpy()
            env_action = np.array([
                (a[0] + 1) / 2,   # throttle [0,1]
                a[1],             # gimbal [-1,1]
                a[2]              # rcs [-1,1]
            ])

            next_state, reward, done, _ = env.step(env_action)

            states.append(state_t)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            ep_reward += reward
            state = next_state

            if done:
                state = env.reset()

        # -------- PPO Update --------
        state_t = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, _, next_value = policy.act(state_t)
            next_value = next_value.item()

        advantages, returns = compute_gae(rewards, values, dones, next_value, GAMMA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        old_logprobs = torch.stack(logprobs).to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)

        for _ in range(K_EPOCHS):
            logprobs_new, values_new, entropy = policy.evaluate(states, actions)

            ratios = torch.exp(logprobs_new - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values_new).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode} \t Reward: {ep_reward:.2f}")

    torch.save(policy.state_dict(), "falcon_ppo_fixed.pth")
    print("Model saved to falcon_ppo_fixed.pth")

if __name__ == '__main__':
    train()

