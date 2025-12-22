import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A custom GridWorld environment for cognitive agent research.
    Supports multiple layouts to test transfer learning.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, layout_name="default", render_mode=None):
        self.render_mode = render_mode
        self.layout_name = layout_name
        
        # Define layouts (0: empty, 1: wall, 2: start, 3: goal)
        self.layouts = {
            "default": np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 3]
            ]),
            "obstacle_course": np.array([
                [0, 1, 0, 0, 3],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0]
            ]),
            "open_field": np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]),
            "trap": np.array([
                [0, 0, 3, 0, 0],  # Goal at top center
                [0, 1, 1, 1, 0],  # Wall blocking direct path
                [0, 1, 0, 1, 0],  # U-shape sides
                [0, 0, 2, 0, 0],  # Start inside the trap (or below it)
                [0, 0, 0, 0, 0]
            ]),
            "complex_maze": np.array([
                [0, 0, 0, 1, 0, 0, 0, 1, 3, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            ])
        }
        
        self.grid = self.layouts.get(layout_name, self.layouts["default"]).copy()
        self.rows, self.cols = self.grid.shape
        self.start_pos = (0, 0) # Default start, can be overridden by '2' in grid if we wanted
        
        # Action space: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Agent row, Agent col
        self.observation_space = spaces.Box(
            low=0, high=max(self.rows, self.cols), shape=(2,), dtype=np.int32
        )

        self.agent_pos = None
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset grid to original layout
        self.grid = self.layouts.get(self.layout_name, self.layouts["default"]).copy()
        
        # Handle Random Start (Curriculum Learning)
        if options and options.get('random_start'):
            # Find all empty spots (0)
            empty_spots = np.argwhere(self.grid == 0)
            if len(empty_spots) > 0:
                idx = np.random.choice(len(empty_spots))
                self.agent_pos = tuple(empty_spots[idx])
            else:
                 self.agent_pos = (0, 0)
        else:
            # Find fixed start position
            starts = np.argwhere(self.grid == 2)
            if len(starts) > 0:
                self.agent_pos = tuple(starts[0])
            else:
                self.agent_pos = (0, 0)
            
        # Init distance for shaping
        goal_pos = np.argwhere(self.grid == 3)[0]
        self.last_dist = abs(self.agent_pos[0] - goal_pos[0]) + abs(self.agent_pos[1] - goal_pos[1])
            
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        # Direction mapping: 0: Up, 1: Right, 2: Down, 3: Left
        # (row, col) changes
        moves = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        
        dr, dc = moves[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        # Check bounds
        if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
            # Check collisions
            if self.grid[new_row, new_col] != 1:
                self.agent_pos = (new_row, new_col)
        
        # Check reward and terminal state
        terminated = False
        reward = -0.1 # Step cost
        
        current_cell = self.grid[self.agent_pos]
        
        if current_cell == 3: # Goal
            reward = 10.0
            terminated = True
        
        # Shaped Reward Helper
        # Calculate Manhattan distance to goal
        goal_pos = np.argwhere(self.grid == 3)[0]
        dist = abs(self.agent_pos[0] - goal_pos[0]) + abs(self.agent_pos[1] - goal_pos[1])
        
        # Shaping: Reward for getting closer, penalty for getting further
        shaping = (self.last_dist - dist) * 0.5 # Stronger factor (0.5 vs 0.1)
        reward += shaping
        
        self.last_dist = dist
        
        truncated = False
        
        return np.array(self.agent_pos, dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
             # Simple console print
             print(f"Agent Pos: {self.agent_pos}")
             for r in range(self.rows):
                 line = ""
                 for c in range(self.cols):
                     if (r, c) == self.agent_pos:
                         line += "A "
                     elif self.grid[r, c] == 1:
                         line += "# "
                     elif self.grid[r, c] == 3:
                         line += "G "
                     else:
                         line += ". "
                 print(line)
             print("-" * 20)
        
        elif self.render_mode == "rgb_array":
            # Create a simple RGB array representation
            cell_size = 40
            img = np.zeros((self.rows * cell_size, self.cols * cell_size, 3), dtype=np.uint8)
            
            # Colors
            colors = {
                0: [255, 255, 255], # Empty: White
                1: [50, 50, 50],    # Wall: Dark Grey
                2: [200, 255, 200], # Start: Light Green (background)
                3: [255, 0, 0]      # Goal: Red
            }
            
            for r in range(self.rows):
                for c in range(self.cols):
                    cell_value = self.grid[r, c]
                    color = colors.get(cell_value, [255, 255, 255])
                    
                    # Fill cell
                    img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
                    
                    # Draw grid lines (optional, simple check)
                    img[r*cell_size, c*cell_size:(c+1)*cell_size] = [200, 200, 200]
                    img[r*cell_size:(r+1)*cell_size, c*cell_size] = [200, 200, 200]

            # Draw Agent
            ar, ac = self.agent_pos
            # Simple blue circle (approximate with square for now to avoid cv2/PIL dep inside env)
            # Or just a smaller blue square
            agent_color = [0, 0, 255]
            margin = 5
            img[ar*cell_size+margin:(ar+1)*cell_size-margin, ac*cell_size+margin:(ac+1)*cell_size-margin] = agent_color
            
            return img
