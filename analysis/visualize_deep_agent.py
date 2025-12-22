import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.gridworld import GridWorldEnv
from agent.agent import HybridAgent

def add_overlay(img_array, text_lines):
    img = Image.fromarray(img_array)
    width, height = img.size
    footer_height = 100
    new_img = Image.new('RGB', (width, height + footer_height), (20, 20, 20))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    y_pos = height + 5
    for line in text_lines:
        draw.text((10, y_pos), line, font=font, fill=(200, 200, 200))
        y_pos += 15
    return np.array(new_img)

def visualize_deep_solution():
    print("Loading Deep Hybrid Agent...")
    env = GridWorldEnv(layout_name="complex_maze", render_mode="rgb_array")
    
    agent = HybridAgent(env, wm_capacity=8, use_deep_rl=True)
    
    model_path = "experiments/results/deep_agent_model.pth"
    if os.path.exists(model_path):
        agent.learner.policy_net.load_state_dict(torch.load(model_path))
        print("Model loaded.")
    else:
        print("Model not found! Please run experiments/deep_hybrid.py first.")
        return

    print("Recording Solution...")
    frames = []
    obs, _ = env.reset()
    done = False
    step = 0
    reward = 0
    
    agent.learner.epsilon = 0.0 # Greedy
    
    while not done and step < 1000:
        q_values = agent.learner.get_q_values(agent.get_state_representation(agent.perception.perceive()))
        wm_content = agent.wm.get_all()
        
        action, source, features = agent.act(obs, reward, done)
        
        frame = env.render()
        text = [
            f"Step: {step}",
            f"Mode: {source.upper()} (Deep Procedural)",
            f"WM: {wm_content}",
            f"Rule Suggestion: {agent.rules.suggest_action(features, agent.wm)}",
            f"DQN Q-Vals: {['%.2f' % q for q in q_values]}"
        ]
        
        frames.append(add_overlay(frame, text))
        
        obs, reward, done, _, _ = env.step(action)
        step += 1
        
    if done:
        frames.append(add_overlay(env.render(), ["SOLVED! (Deep RL)"]))
    else:
        frames.append(add_overlay(env.render(), ["FAILED"]))
    
    output_path = "analysis/agent_solution_deep.gif"
    imageio.mimsave(output_path, frames, duration=0.5, loop=0)
    print(f"Saved GIF to {output_path}")

if __name__ == "__main__":
    visualize_deep_solution()
