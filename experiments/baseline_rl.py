import numpy as np
import random
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.learner import QLearner
from agent.logger import TraceLogger

def run_baseline(episodes=100):
    print(f"Starting Baseline RL for {episodes} episodes...")
    env = GridWorldEnv(layout_name="default")
    learner = QLearner(action_space_n=env.action_space.n, epsilon=0.1)
    logger = TraceLogger()

    for ep in range(episodes):
        state_arr, _ = env.reset()
        state = tuple(state_arr)
        done = False
        step = 0
        
        while not done and step < 50:
            action = learner.choose_action(state)
            
            next_state_arr, reward, done, _, _ = env.step(action)
            next_state = tuple(next_state_arr)
            
            learner.learn(state, action, reward, next_state)
            
            logger.log_step(ep, step, {}, "None", action, reward, "rl_pure")
            
            state = next_state
            step += 1
            
    output_path = "experiments/results/baseline_rl.csv"
    logger.save(output_path)
    print("Baseline done.")

if __name__ == "__main__":
    run_baseline()
