import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import requests
import numpy as np
import time

# --- PPO Hyperparameters ---
LR = 0.0003
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 128  # Small batch for faster loops in this demo
HIDDEN_DIM = 64
ACTION_STD = 0.5  # Fixed std dev for exploration

# --- Environment Interface ---
class FalconRemoteEnv:
    def __init__(self, url='http://localhost:3000'):
        self.url = url
        self.state_dim = 6
        self.action_dim = 3 # Throttle, Gimbal, RCS

    def reset(self):
        resp = requests.post(f"{self.url}/reset", json={})
        return np.array(resp.json()['state'], dtype=np.float32)

    def step(self, action):
        # Action is np array
        resp = requests.post(f"{self.url}/step", json={'action': action.tolist()})
        data = resp.json()
        return (
            np.array(data['state'], dtype=np.float32), 
            data['reward'], 
            data['done'], 
            data.get('info', {})
        )

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, action_dim)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.action_var = torch.full((action_dim,), ACTION_STD * ACTION_STD)

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(state.device)
        dist = Normal(action_mean, torch.sqrt(cov_mat))
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean).to(state.device)
        cov_mat = torch.diag_embed(action_var)
        dist = Normal(action_mean, torch.sqrt(cov_mat))
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# --- PPO Training Loop ---
def train():
    env = FalconRemoteEnv()
    device = torch.device('cpu') # or 'cuda'
    
    policy = ActorCritic(env.state_dim, env.action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    policy_old = ActorCritic(env.state_dim, env.action_dim).to(device)
    policy_old.load_state_dict(policy.state_dict())
    
    buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'terminals': []}
    
    print("Starting Training... (Ensure node server.js is running!)")
    
    time_step = 0
    max_episodes = 500  # For demo
    update_timestep = 2000 
    
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        ep_reward = 0
        
        for t in range(1000): # Max steps per episode
            time_step += 1
            
            state_tensor = torch.FloatTensor(state).to(device)
            action, logprob = policy_old.act(state_tensor)
            
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
            # Record
            buffer['states'].append(state_tensor)
            buffer['actions'].append(action)
            buffer['logprobs'].append(logprob)
            buffer['rewards'].append(reward)
            buffer['terminals'].append(done)
            
            state = next_state
            ep_reward += reward
            
            # Update Policy
            if time_step % update_timestep == 0:
                print(f"Update at Step {time_step}")
                # Convert buffer
                # (Simplified PPO update implementation for brevity)
                # In real setup, would compute advantages, iterate K_EPOCHS, backwards pass
                # Reset buffer
                buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'terminals': []}
                policy_old.load_state_dict(policy.state_dict())
            
            if done:
                break
        
        print(f"Episode {i_episode} \t Reward: {ep_reward:.2f}")
        
    # Save Model
    torch.save(policy.state_dict(), "falcon_ppo.pth")
    print("Model saved to falcon_ppo.pth")

if __name__ == '__main__':
    train()
