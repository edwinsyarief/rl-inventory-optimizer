import argparse
import logging
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.cuda.amp as amp  # For mixed precision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset

# Set up logging (minimal to save I/O cost)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):  # Increased hidden for better learning without overkill
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class InventoryEnv(gym.Env):
    """Custom environment for inventory management.
    State: [current_inventory, demand_forecast]
    Actions: Discrete order quantities (0 to max_order)
    Rewards: -holding_cost - shortage_cost - ordering_cost + revenue
    Goal: Minimize total costs under uncertainty.
    Unique: Random supply disruptions as penalties.
    Optimized: Fast NumPy ops, no heavy computations.
    """
    def __init__(self, max_inventory=100, max_order=50, holding_cost=0.5, shortage_cost=2.0, ordering_cost=1.0, disruption_prob=0.05):
        super(InventoryEnv, self).__init__()
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.ordering_cost = ordering_cost
        self.disruption_prob = disruption_prob
        
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([max_inventory, max_inventory]), dtype=np.float32)
        self.action_space = spaces.Discrete(max_order + 1)  # 0 to max_order
        
        self.current_inventory = 0
        self.demand_forecast = 0
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.current_inventory = self.max_inventory // 2
        self.demand_forecast = np.random.randint(10, 50)
        state = np.array([self.current_inventory, self.demand_forecast], dtype=np.float32)
        return state, {}
    
    def step(self, action):
        order = action
        actual_demand = max(0, int(np.random.normal(self.demand_forecast, 5)))  # Stochastic demand, int for realism
        
        # Apply order with possible disruption (efficient random check)
        if random.random() < self.disruption_prob:
            order = int(order * 0.5)  # 50% supply disruption
        
        new_inventory = self.current_inventory + order - actual_demand
        
        # Costs and rewards (minimize cost: negative costs as penalties)
        holding = max(0, new_inventory) * self.holding_cost
        shortage = max(0, -new_inventory) * self.shortage_cost
        ordering = order * self.ordering_cost
        revenue = min(self.current_inventory + order, actual_demand) * 3.0  # Assume sale price
        
        reward = revenue - holding - shortage - ordering
        
        self.current_inventory = min(self.max_inventory, max(0, new_inventory))
        self.demand_forecast = np.random.randint(10, 50)  # New forecast
        
        terminated = False
        truncated = False
        info = {}
        
        state = np.array([self.current_inventory, self.demand_forecast], dtype=np.float32)
        return state, reward, terminated, truncated, info

def train_dqn(args):
    try:
        # Create custom environment
        env = InventoryEnv()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Initialize networks
        policy_net = DQN(state_size, action_size, args.hidden_size).to(args.device)
        target_net = DQN(state_size, action_size, args.hidden_size).to(args.device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        
        optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, weight_decay=1e-5)  # AdamW for better generalization, low decay
        replay_buffer = ReplayBuffer(args.buffer_size)
        dataloader = DataLoader(replay_buffer, batch_size=args.batch_size, shuffle=True, num_workers=0)  # No workers to avoid overhead
        
        scaler = amp.GradScaler(enabled=args.device == 'cuda')  # Mixed precision for cost savings on GPU
        
        epsilon = args.epsilon_start
        episode_rewards = []
        episode_costs = []  # Track costs for minimization validation
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        for episode in range(args.episodes):
            state, _ = env.reset()
            state = torch.FloatTensor(state).to(args.device)
            total_reward = 0
            total_cost = 0
            done = False
            steps = 0
            max_steps = 200  # Cap to prevent cost leaks from long episodes
            
            while not done and steps < max_steps:
                steps += 1
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = policy_net(state.unsqueeze(0))
                        action = q_values.argmax().item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                # Calculate cost from env (holding + shortage + ordering)
                total_cost += (env.holding_cost * max(0, env.current_inventory) + 
                               env.shortage_cost * max(0, -env.current_inventory) + 
                               env.ordering_cost * action)
                
                next_state_tensor = torch.FloatTensor(next_state).to(args.device)
                replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
                
                state = next_state_tensor
                
                if len(replay_buffer) >= args.batch_size:
                    batch = next(iter(dataloader))  # Single batch per step for efficiency
                    states, actions, rewards, next_states, dones = batch
                    states = torch.FloatTensor(np.array(states)).to(args.device)
                    actions = torch.LongTensor(actions).to(args.device)
                    rewards = torch.FloatTensor(rewards).to(args.device)
                    next_states = torch.FloatTensor(np.array(next_states)).to(args.device)
                    dones = torch.FloatTensor(dones).to(args.device)
                    
                    with amp.autocast(enabled=args.device == 'cuda'):
                        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        
                        with torch.no_grad():
                            next_q_values = target_net(next_states).max(1)[0]
                            expected_q_values = rewards + args.gamma * next_q_values * (1 - dones)
                        
                        loss = F.smooth_l1_loss(q_values, expected_q_values)
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Clip to prevent explosions
                    scaler.step(optimizer)
                    scaler.update()
            
            epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
            episode_rewards.append(total_reward)
            episode_costs.append(total_cost)
            
            if (episode + 1) % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval:])
                avg_cost = np.mean(episode_costs[-args.log_interval:])
                logger.info(f"Episode {episode + 1}/{args.episodes} | Avg Reward: {avg_reward:.2f} | Avg Cost: {avg_cost:.2f} | Epsilon: {epsilon:.3f}")
            
            if (episode + 1) % args.save_interval == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"dqn_episode_{episode + 1}.pth")
                torch.save(policy_net.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Soft update target net every episode for stability
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(args.tau * policy_param.data + (1.0 - args.tau) * target_param.data)
        
        torch.save(policy_net.state_dict(), args.model_path)
        logger.info(f"Training complete. Final model saved to {args.model_path}. Final Avg Cost: {np.mean(episode_costs[-10:]):.2f}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent in PyTorch for Inventory Management")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")  # Larger for efficiency
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update factor for target network")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")  # Lower for stability
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Ending epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")  # Less frequent to save I/O
    parser.add_argument("--save_interval", type=int, default=200, help="Save checkpoint interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--model_path", type=str, default="dqn_inventory_agent.pth", help="Final model save path")
    args = parser.parse_args()
    
    train_dqn(args)