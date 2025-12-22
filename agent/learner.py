import numpy as np
import random

class QLearner:
    """
    Tabular Q-Learning Agent.
    """
    def __init__(self, action_space_n, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space_n = action_space_n
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.q_table = {} # Map state (tuple) -> np.array([q_values])

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_n)
        return self.q_table[state]

    def choose_action(self, state, suggested_action=None):
        """
        Choose action based on Epsilon-Greedy policy.
        If suggested_action (from rules) is provided, we might prioritize it 
        (handled in agent.py usually, but here we treat it as a bias if we wanted).
        For pure RL, we ignore suggested_action or treat it as a separate concern.
        Here we implement standard epsilon-greedy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_n - 1)
        
        q_vals = self.get_q_values(state)
        # Random tie-breaking
        max_q = np.max(q_vals)
        actions = np.where(q_vals == max_q)[0]
        return np.random.choice(actions)

    def learn(self, state, action, reward, next_state, done=False):
        """
        Updates Q-value using Q-learning rule
        """
        old_q = self.get_q_values(state)[action]
        next_max = np.max(self.get_q_values(next_state))
        
        # Q-Learning target
        # Note: Tabular usually doesn't strictly need done loop break here if managed by agent loop,
        # but technically target should be reward if done.
        target = reward + (self.gamma * next_max * (not done)) 
        
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[state][action] = new_q
