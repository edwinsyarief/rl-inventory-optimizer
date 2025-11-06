import random
import time
from collections import deque

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class InventoryEnv:
    def __init__(self, max_inventory=100, max_order=50, holding_cost=0.5, shortage_cost=2.0, ordering_cost=1.0, disruption_prob=0.05):
        self.max_inventory = max_inventory
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.ordering_cost = ordering_cost
        self.disruption_prob = disruption_prob
        
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([max_inventory, max_inventory]), dtype=np.float32)
        self.action_space = spaces.Discrete(max_order + 1)
        
        self.current_inventory = 0
        self.demand_forecast = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.current_inventory = self.max_inventory // 2
        self.demand_forecast = np.random.randint(10, 50)
        state = np.array([self.current_inventory, self.demand_forecast], dtype=np.float32)
        return state, {}
    
    def step(self, action):
        order = action
        actual_demand = max(0, int(np.random.normal(self.demand_forecast, 5)))
        
        if random.random() < self.disruption_prob:
            order = int(order * 0.5)
        
        new_inventory = self.current_inventory + order - actual_demand
        
        holding = max(0, new_inventory) * self.holding_cost
        shortage = max(0, -new_inventory) * self.shortage_cost
        ordering = order * self.ordering_cost
        revenue = min(self.current_inventory + order, actual_demand) * 3.0
        
        reward = revenue - holding - shortage - ordering
        
        self.current_inventory = min(self.max_inventory, max(0, new_inventory))
        self.demand_forecast = np.random.randint(10, 50)
        
        terminated = False
        truncated = False
        info = {}
        
        state = np.array([self.current_inventory, self.demand_forecast], dtype=np.float32)
        return state, reward, terminated, truncated, info

def benchmark_training(episodes=50, batch_size=128, device='cpu'):
    env = InventoryEnv()
    state_size = 2
    action_size = 51
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=0.0005)
    replay_buffer = ReplayBuffer(10000)
    scaler = amp.GradScaler(enabled=device == 'cuda')
    epsilon = 1.0
    gamma = 0.99
    tau = 0.005
    start_time = time.time()
    episode_costs = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        total_cost = 0
        steps = 0
        while steps < 50:
            steps += 1
            if random.random() < epsilon:
                action = random.randint(0, 50)
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    action = q_values.argmax().item()
            next_state, reward, _, _, _ = env.step(action)
            total_cost += (env.holding_cost * max(0, env.current_inventory) + 
                           env.shortage_cost * max(0, -env.current_inventory) + 
                           env.ordering_cost * action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state, False)
            state = next_state_tensor
            
            if len(replay_buffer) >= batch_size:
                batch = list(random.sample(replay_buffer.buffer, batch_size))
                states = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
                actions = torch.LongTensor([b[1] for b in batch]).to(device)
                rewards = torch.FloatTensor([b[2] for b in batch]).to(device)
                next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
                dones = torch.FloatTensor([b[4] for b in batch]).to(device)
                
                with amp.autocast(enabled=device == 'cuda'):
                    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0]
                    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
                    loss = F.smooth_l1_loss(q_values, expected_q_values)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        epsilon = max(0.01, epsilon * 0.995)
        episode_costs.append(total_cost)
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    duration = time.time() - start_time
    avg_cost_first = np.mean(episode_costs[:5])
    avg_cost_last = np.mean(episode_costs[-5:])
    print(f"Training Benchmark: {episodes} episodes on {device}")
    print(f"Duration: {duration:.2f}s")
    print(f"Time per episode: {duration / episodes:.2f}s")
    print(f"Avg Cost First 5: {avg_cost_first:.2f}")
    print(f"Avg Cost Last 5: {avg_cost_last:.2f}")
    print(f"Cost Reduction: {(avg_cost_first - avg_cost_last) / avg_cost_first * 100:.2f}%")

def benchmark_inference(device='cpu', num_inferences=1000):
    state_size = 2
    action_size = 51
    model = DQN(state_size, action_size).to(device)
    model.eval()
    start_time = time.time()
    for _ in range(num_inferences):
        state = torch.randn(1, state_size).to(device)
        with torch.no_grad():
            output = model(state)
            action = output.argmax().item()
    duration = time.time() - start_time
    print(f"Inference Benchmark on {device}: {num_inferences} inferences")
    print(f"Duration: {duration:.2f}s")
    print(f"Avg time per inference: {duration / num_inferences * 1000:.2f}ms")

if __name__ == '__main__':
    benchmark_training()
    benchmark_inference()