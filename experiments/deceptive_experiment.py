import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent
from agent.logger import TraceLogger

def run_deceptive_experiment(episodes=500):
    print(f"Starting Deceptive Maze Experiment for {episodes} episodes...")
    env = GridWorldEnv(layout_name="trap")
    
    # We use a standard hybrid agent
    agent = HybridAgent(env, wm_capacity=4)
    # Important: We want to see RL take over. 
    # Make sure learner has high enough alpha/epsilon to find the path eventually.
    agent.learner.epsilon = 0.2 
    agent.learner.alpha = 0.2
    
    logger = TraceLogger()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        reward = None
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 100: # Give it more time to escape trap
            action, source, features = agent.act(obs, reward, done)
            next_obs, reward, done, _, _ = env.step(action)
            
            # Log
            wm_content = agent.wm.get_all()
            logger.log_step(ep, step, features, str(wm_content), action, reward, source)
            
            obs = next_obs
            total_reward += reward
            step += 1
            
        # Final update
        agent.act(obs, reward, done=True)
        
        if ep % 10 == 0:
            print(f"Episode {ep}: Total Reward {total_reward:.2f}")

    output_path = "experiments/results/deceptive_agent.csv"
    logger.save(output_path)
    print("Deceptive Experiment done.")

if __name__ == "__main__":
    run_deceptive_experiment()
