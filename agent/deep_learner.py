import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network: Approximates Q(s, a)
    "Deep Procedural Memory"
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
     Episodic Memory / Experience Replay
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class DeepQLearner:
    """
    DQN Agent Component ("Procedural Module")
    """
    def __init__(self, input_dim, action_space_n, lr=1e-3, gamma=0.99, buffer_size=50000):
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 # Balanced decay
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(input_dim, action_space_n).to(self.device)
        self.target_net = DQN(input_dim, action_space_n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(buffer_size)
        self.learn_step_counter = 0
        self.target_update_freq = 100

    def get_q_values(self, state):
        """
        Returns Q-values for a given state (tensor or numpy array).
        """
        # Ensure state is float tensor
        if isinstance(state, tuple):
             # Handle the tuple state from HybridAgent (features, wm)
             # We need to flatten this into a vector for the NN
             state = self._flatten_state(state)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.cpu().numpy()[0]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_n)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        # Flatten states
        state_flat = self._flatten_state(state)
        next_state_flat = self._flatten_state(next_state)
        
        self.memory.push(state_flat, action, reward, next_state_flat, done)
        
        if len(self.memory) < self.batch_size:
            return
            
        # Sample
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # Compute V(s')
        # Double DQN logic: use policy net to choose action, target net to evaluate
        # Standard DQN for simplicity here: max Q from target
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))
            
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update Target Net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _flatten_state(self, state_tuple):
        """
        Flattens the (features, wm) tuple into a single numeric vector.
        """
        features, wm_content = state_tuple
        
        # Features: (('goal_east', 0), ('goal_north', 0), ...)
        # We need a stable ordering. agent.py already sorts them.
        # Just extract values.
        # But wait, 'goal_visible' is boolean/int. 'agent_pos' is not in features list usually?
        # Let's check perception.py.
        # Perception returns dict. Agent sorts it into tuple of pairs.
        
        # Better strategy: We need a fixed-size input vector for the NN.
        # The tuple format is variable length or at least generic.
        # For this gridworld, let's assume we map specific keys to indices.
        
        # Input Vector Design:
        # [AgentRow, AgentCol, GoalRow, GoalCol, Wall_Up, Wall_Right, Wall_Down, Wall_Left, 
        #  WM_Goal_N, WM_Goal_S, WM_Goal_E, WM_Goal_W]
        
        # This requires unwrapping the 'features' list more intelligently.
        # However, `state_tuple` passed here is: (tuple(sorted_features), wm_tuple)
        
        # Let's decode it back to a dict for easier processing
        feat_dict = dict(features)
        
        # Extract normalized coords (assuming max 10x10 grid for now)
        # Note: Perception output needs to be sufficient.
        # Current Perception: {'agent_pos': (r, c), 'goal_visible': T/F, ...}
        
        # We need to standardize this vector encoding.
        # Let's build a vector of size 12.
        vec = []
        
        # Flatten valid logic
        # Perception returns: {'wall_north': T/F, ..., 'goal_north': T/F}
        # It does NOT return 'agent_pos' or 'walls' list.
        # We need to map the dict keys to the vector.
        
        vec = []
        
        # 1. We don't have absolute position in features. (That's okay, RL should capture relative policy)
        # But if we want it, we'd need to change Perception. 
        # For now, let's zero it out or rely on the relative goal features (which encode position implicitly).
        # UPDATE: State Aliasing detected. We need absolute pos.
        vec.append(feat_dict.get('norm_r', 0.0))
        vec.append(feat_dict.get('norm_c', 0.0))
        
        # 2. Local Perception (Walls)
        vec.append(1.0 if feat_dict.get('wall_north') else 0.0)
        vec.append(1.0 if feat_dict.get('wall_east') else 0.0)
        vec.append(1.0 if feat_dict.get('wall_south') else 0.0)
        vec.append(1.0 if feat_dict.get('wall_west') else 0.0)
        
        # 3. Goal Perception
        vec.append(1.0 if feat_dict.get('goal_north') else 0.0)
        vec.append(1.0 if feat_dict.get('goal_south') else 0.0)
        vec.append(1.0 if feat_dict.get('goal_east') else 0.0)
        vec.append(1.0 if feat_dict.get('goal_west') else 0.0)
        
        # 4. Working Memory (Context)
        vec.append(1.0 if 'goal_north' in wm_content else 0.0)
        vec.append(1.0 if 'goal_south' in wm_content else 0.0)
        vec.append(1.0 if 'goal_east' in wm_content else 0.0)
        vec.append(1.0 if 'goal_west' in wm_content else 0.0)
        
        return np.array(vec, dtype=np.float32)
