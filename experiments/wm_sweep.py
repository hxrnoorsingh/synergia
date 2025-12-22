import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent
from agent.logger import TraceLogger

def run_wm_sweep(capacities=[1, 2, 4, 8], episodes_per_config=50):
    print(f"Starting WM Capacity Sweep: {capacities}")
    env = GridWorldEnv(layout_name="obstacle_course") # Use complex layout where memory matters more
    
    all_results = []
    
    for cap in capacities:
        print(f"Testing Capacity: {cap}")
        agent = HybridAgent(env, wm_capacity=cap)
        # Reset agent logic between configs? Agent is re-instantiated, so yes.
        # But Q-table is fresh. 
        # Ideally, we want to see how quickly they learn OR how well they perform with fixed knowledge.
        # Let's do standard learning curve.
        
        for ep in range(episodes_per_config):
            obs, _ = env.reset()
            reward = None
            done = False
            step = 0
            
            total_reward = 0
            
            while not done and step < 50:
                action, source, _ = agent.act(obs, reward, done)
                next_obs, reward, done, _, _ = env.step(action)
                
                obs = next_obs
                total_reward += reward
                step += 1
                
            # Final update
            agent.act(obs, reward, done=True)
            
            all_results.append({
                'capacity': cap,
                'episode': ep,
                'total_reward': total_reward,
                'steps': step
            })
            
    df = pd.DataFrame(all_results)
    output_path = "experiments/results/wm_sweep.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("WM Sweep done.")

if __name__ == "__main__":
    run_wm_sweep()
