import random
import unittest
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


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

class TestInventoryEnv(unittest.TestCase):
    def setUp(self):
        self.env = InventoryEnv()
        self.env.reset(seed=42)

    def test_reset(self):
        state, _ = self.env.reset(seed=42)
        self.assertEqual(state.shape, (2,))
        self.assertTrue(0 <= state[0] <= self.env.max_inventory)
        self.assertTrue(10 <= state[1] <= 50)

    def test_step_no_disruption(self):
        original_prob = self.env.disruption_prob
        self.env.disruption_prob = 0.0
        self.env.current_inventory = 50
        self.env.demand_forecast = 30
        np.random.seed(42)
        state, reward, _, _, _ = self.env.step(20)
        # Calc expected
        np.random.seed(42)
        actual_demand = max(0, int(np.random.normal(30, 5)))
        order = 20
        new_inv = 50 + order - actual_demand
        holding = max(0, new_inv) * 0.5
        shortage = max(0, -new_inv) * 2.0
        ordering = order * 1.0
        revenue = min(50 + order, actual_demand) * 3.0
        expected_reward = revenue - holding - shortage - ordering
        self.assertAlmostEqual(reward, expected_reward, places=4)
        self.env.disruption_prob = original_prob

    def test_step_disruption(self):
        original_prob = self.env.disruption_prob
        self.env.disruption_prob = 1.0
        self.env.current_inventory = 50
        self.env.demand_forecast = 30
        np.random.seed(42)
        random.seed(42)
        state, reward, _, _, _ = self.env.step(20)
        # Calc expected
        np.random.seed(42)
        random.seed(42)
        actual_demand = max(0, int(np.random.normal(30, 5)))
        order = 20
        random.random()  # Consume
        order = int(20 * 0.5)
        new_inv = 50 + order - actual_demand
        holding = max(0, new_inv) * 0.5
        shortage = max(0, -new_inv) * 2.0
        ordering = order * 1.0
        revenue = min(50 + order, actual_demand) * 3.0
        expected_reward = revenue - holding - shortage - ordering
        self.assertAlmostEqual(reward, expected_reward, places=4)
        self.env.disruption_prob = original_prob

class TestDQN(unittest.TestCase):
    def setUp(self):
        self.state_size = 2
        self.action_size = 51
        self.model = DQN(self.state_size, self.action_size)

    def test_forward(self):
        input_tensor = torch.randn(1, self.state_size)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, self.action_size))

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(100)

    def test_push_and_len(self):
        self.buffer.push([1,2], 0, 1.0, [3,4], False)
        self.assertEqual(len(self.buffer), 1)

    def test_maxlen(self):
        for i in range(150):
            self.buffer.push([i,i], i%51, float(i), [i+1,i+1], i%2==0)
        self.assertEqual(len(self.buffer), 100)

if __name__ == '__main__':
    unittest.main(verbosity=2)