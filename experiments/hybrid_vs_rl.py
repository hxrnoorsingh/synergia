import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent
from agent.learner import QLearner
from agent.logger import TraceLogger

def run_hybrid_experiment(episodes=100):
    print(f"Starting Hybrid vs RL Experiment for {episodes} episodes...")
    env = GridWorldEnv(layout_name="default")
    
    # Initialize Agents
    hybrid_agent = HybridAgent(env)
    
    # We can also track a pure RL agent here or reuse baseline data.
    # For this script we run the Hybrid Agent.
    
    logger = TraceLogger()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        reward = None
        done = False
        step = 0
        
        while not done and step < 50:
            # Agent acts (and learns from previous step if reward is not None)
            action, source, _ = hybrid_agent.act(obs, reward, done)
            
            # Environment Step
            next_obs, reward, done, _, _ = env.step(action)
            
            # Log
            # We need internal state of agent for logging if possible, 
            # but agent.act returns features which is good.
            # We'll need to capture features from act return.
            # Modifying agent.act to return features as well.
            
            # Correction: In my agent.py, act returns (action, source, features)
            # So I need to unpack 3 values.
            
            logger.log_step(ep, step, {}, str(hybrid_agent.wm.get_all()), action, reward, source)
            
            obs = next_obs
            step += 1
            
        # Final learning update for terminal state
        hybrid_agent.act(obs, reward, done=True)
            
    output_path = "experiments/results/hybrid_agent.csv"
    logger.save(output_path)
    print("Hybrid Experiment done.")

if __name__ == "__main__":
    run_hybrid_experiment()
