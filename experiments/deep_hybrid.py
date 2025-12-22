import pandas as pd
import numpy as np
import os
import sys
import torch # Added missing import

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent

def run_deep_experiment():
    print("Running Deep Hybrid Experiment (Complex Maze)...")
    
    env = GridWorldEnv(layout_name="complex_maze")
    agent = HybridAgent(env, wm_capacity=8, use_deep_rl=True)
    
    episodes = 6000 # Quick run to generate model
    results = []
    
    # DQN Params are internal to DeepQLearner class, but we can set epsilon decaying here
    # DeepQLearner has its own decay logic in learn(), but we can override manually if needed.
    # Actually DeepQLearner logic I wrote: "epsilon = max(min, eps * decay)" inside learn().
    # That means it decays PER STEP if learn is called.
    # Wait, usually decay is per episode or per step.
    # In my code: `self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)` inside learn().
    # Only learns if batch ready.
    # That is very fast decay if per step! 0.995 * 100 steps -> 0.6.
    # Let's check deep_learner.py logic.
    pass

    for e in range(episodes):
        # Curriculum: Always keep 20% random starts to prevent catastrophic forgetting
        if e < episodes * 0.5:
            use_random_start = True # Force random start early
        else:
            use_random_start = (np.random.rand() < 0.2) # 20% chance later 
        
        obs, _ = env.reset(options={'random_start': use_random_start})
        state = agent.get_state_representation(agent.perception.perceive())
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 1000:
            action, _, _ = agent.act(obs, None, done)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
        agent.act(obs, reward, True) # Final update
        
        if e % 100 == 0:
            print(f"Ep {e}: Epsilon {agent.learner.epsilon:.2f}, Reward {total_reward:.2f}")
            results.append({'Episode': e, 'Reward': total_reward})
            
    # Save results
    df = pd.DataFrame(results)
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/deep_hybrid.csv", index=False)
    
    # Save Model
    torch.save(agent.learner.policy_net.state_dict(), "experiments/results/deep_agent_model.pth")
    print("Experiment Complete. Data and Model saved.")

if __name__ == "__main__":
    run_deep_experiment()
