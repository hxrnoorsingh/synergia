import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent

def add_overlay(img_array, text_lines):
    """
    Adds a text overlay to the bottom of the image.
    """
    # Convert to PIL
    img = Image.fromarray(img_array)
    width, height = img.size
    
    # Create new image with extra space at bottom
    footer_height = 80
    new_img = Image.new('RGB', (width, height + footer_height), (20, 20, 20))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # Text config (default font)
    # Try to load a known font if possible, else default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        
    y_pos = height + 5
    for line in text_lines:
        draw.text((10, y_pos), line, font=font, fill=(200, 200, 200))
        y_pos += 15
        
    return np.array(new_img)

def visualize_deceptive_solution():
    # 1. Train first (More thorough)
    print("Training Agent for Visualization (Complex Maze)...")
    env = GridWorldEnv(layout_name="complex_maze", render_mode="rgb_array")
    
    agent = HybridAgent(env, wm_capacity=8) # Higher capacity for complex maze
    agent.learner.epsilon = 1.0 # Start with full exploration
    agent.learner.alpha = 0.5 # High learning rate
    
    episodes = 10000 # Increased for larger state space
    decay_rate = 0.9995 
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        # Decay epsilon
        agent.learner.epsilon = max(0.05, agent.learner.epsilon * decay_rate) 
        
        while not done and step < 200: # Increased step limit for larger maze
            action, _, _ = agent.act(obs, None, done)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
        agent.act(obs, reward, True) # Final update
        
        if ep % 500 == 0:
            print(f"Ep {ep}: Epsilon {agent.learner.epsilon:.2f}, Reward {total_reward}")
        
    print(f"Training Complete. Final Epsilon: {agent.learner.epsilon:.4f}")
    
    # 2. Record Solution
    frames = []
    obs, _ = env.reset()
    done = False
    step = 0
    reward = 0
    
    # Make sure agent doesn't explore randomly during recording
    agent.learner.epsilon = 0.0
    
    while not done and step < 200: # Increased step limit
        # Capture "Thought Process" BEFORE stepping
        # We need to peek into the agent's act() to get details, 
        # but act() returns (action, source, features).
        
        # We'll use the agent's components directly to get extra info for display
        q_values = agent.learner.get_q_values(agent.get_state_representation(agent.perception.perceive()))
        wm_content = agent.wm.get_all()
        
        # Act
        action, source, features = agent.act(obs, reward, done)
        
        # Render
        frame = env.render()
        
        # Info Text
        text = [
            f"Step: {step}",
            f"Mode: {source.upper()}",
            f"WM: {wm_content}",
            f"Rule Suggestion: {agent.rules.suggest_action(features, agent.wm)}",
            f"Q-Vals: {['%.2f' % q for q in q_values]}"
        ]
        
        final_frame = add_overlay(frame, text)
        frames.append(final_frame)
        
        # Step Env
        obs, reward, done, _, _ = env.step(action)
        step += 1
        
    # Last frame check
    if done:
        frames.append(add_overlay(env.render(), ["SOLVED!"]))
    else:
        frames.append(add_overlay(env.render(), ["FAILED TO SOLVE (Timeout)"]))
    
    # Save GIF
    output_path = "analysis/agent_solution_complex.gif"
    imageio.mimsave(output_path, frames, duration=0.5, loop=0) # Faster fps for long maze
    print(f"Saved GIF to {output_path}")

if __name__ == "__main__":
    visualize_deceptive_solution()
