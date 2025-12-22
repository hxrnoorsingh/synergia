import numpy as np

class Perception:
    """
    Perception module that extracts relevant features from the GridWorld state.
    It translates raw grid data into symbolic predicates.
    """
    def __init__(self, env):
        self.env = env

    def perceive(self):
        """
        Scans the environment and returns a dictionary of perceived features.
        """
        features = {}
        
        agent_pos = self.env.agent_pos
        grid = self.env.grid
        rows, cols = self.env.rows, self.env.cols
        
        # Helper for relative checks
        directions = {
            'north': (-1, 0),
            'east': (0, 1),
            'south': (1, 0),
            'west': (0, -1)
        }
        
        # 1. Local Obstacle Detection
        for direct, (dr, dc) in directions.items():
            r, c = agent_pos[0] + dr, agent_pos[1] + dc
            if not (0 <= r < rows and 0 <= c < cols):
                features[f'wall_{direct}'] = True # Boundary is a wall for us
            elif grid[r, c] == 1:
                features[f'wall_{direct}'] = True
            else:
                features[f'wall_{direct}'] = False
                
        # 2. Global Goal Detection (Simulating visual search)
        # In a real cognitive model, this might be limited by line-of-sight
        goal_pos = np.argwhere(grid == 3)
        if len(goal_pos) > 0:
            gr, gc = goal_pos[0]
            ar, ac = agent_pos
            
            features['goal_visible'] = True
            
            # Determine relative direction of goal
            if gr < ar: features['goal_north'] = True
            if gr > ar: features['goal_south'] = True
            if gc > ac: features['goal_east'] = True
            if gc < ac: features['goal_west'] = True
            
            # Exact alignment check
            if gr == ar and gc > ac: features['goal_directly_east'] = True
            if gr == ar and gc < ac: features['goal_directly_west'] = True
            if gc == ac and gr < ar: features['goal_directly_north'] = True
            if gc == ac and gr > ar: features['goal_directly_south'] = True
            
        else:
            features['goal_visible'] = False
            
        # Add Normalized Position (Critical for disambiguating identical local states)
        features['norm_r'] = agent_pos[0] / rows
        features['norm_c'] = agent_pos[1] / cols
            
        return features
