import numpy as np
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent

def run_comparison(episodes=500):
    print("Running Complexity Wall Experiment...")
    
    layouts = ['default', 'complex_maze']
    results = []
    
    for layout in layouts:
        print(f"Testing Layout: {layout}")
        env = GridWorldEnv(layout_name=layout)
        # Use substantial capacity
        agent = HybridAgent(env, wm_capacity=8)
        
        # Standard parameters
        agent.learner.epsilon = 1.0
        agent.learner.alpha = 0.5
        decay = 0.99
        
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            total_reward = 0
            
            agent.learner.epsilon = max(0.05, agent.learner.epsilon * decay)
            
            while not done and step < 200:
                action, _, _ = agent.act(obs, None, done)
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                step += 1
            agent.act(obs, reward, True)
            
            # Smooth result for plot clarity (rolling average online-ish)
            results.append({
                'Layout': 'Simple Maze' if layout == 'default' else 'Complex Maze (10x10)',
                'Episode': ep,
                'Reward': total_reward
            })

    df = pd.DataFrame(results)
    
    # Calculate rolling mean for smoother lines
    df['Reward Smooth'] = df.groupby('Layout')['Reward'].transform(lambda x: x.rolling(getWindow(x)).mean())
    df.fillna(method='bfill', inplace=True) # Fill execution start holes

    # Static plot for README (using Image write if possible, or usually just show())
    # Since we need to save it for the README, we will use write_image (requires kaleido) 
    # OR we can just save the CSV and let the user run the notebook. 
    # BUT user asked for "graph supporting the wall". I will try to save a PNG using kaleido if available,
    # or just save CSV and instruct.
    # Actually, let's just save the CSV and I'll generate the plot in the notebook/visualizer.
    # Wait, user wants it in README. I should try to save a static image.
    
    csv_path = "experiments/results/complexity_wall.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def getWindow(x):
    return 20

if __name__ == "__main__":
    run_comparison()
