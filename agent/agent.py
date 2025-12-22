import numpy as np
from .perception import Perception
from .working_memory import WorkingMemory
from .rules import RuleEngine
from .learner import QLearner
try:
    from .deep_learner import DeepQLearner
except ImportError:
    DeepQLearner = None

class HybridAgent:
    """
    Cognitive Agent integrating:
    - Perception
    - Working Memory
    - Symbolic Rules
    - Reinforcement Learning
    """
    def __init__(self, env, wm_capacity=4, use_deep_rl=False):
        self.env = env
        self.perception = Perception(env)
        self.wm = WorkingMemory(capacity=wm_capacity)
        self.rules = RuleEngine()
        
        self.use_deep_rl = use_deep_rl
        
        if use_deep_rl:
            if DeepQLearner is None:
                raise ImportError("Torch not installed or DeepQLearner not found.")
            # Input dim = 14 (2 pos + 4 walls + 4 goal_dir + 4 wm)
            self.learner = DeepQLearner(input_dim=14, action_space_n=env.action_space.n)
        else:
            self.learner = QLearner(action_space_n=env.action_space.n)
        
        self.last_state = None
        self.last_action = None
        
    def get_state_representation(self, features):
        """
        Converts perception features + WM state into a hashable state tuple for RL.
        """
        # Sort features to ensure consistent ordering
        feature_list = sorted([(k, v) for k, v in features.items()])
        
        # Add WM content (sorted or just as tuple)
        wm_content = tuple(sorted(self.wm.get_all()))
        
        return (tuple(feature_list), wm_content)

    def act(self, observation, reward=None, done=False):
        """
        Main decision cycle.
        1. Perceive
        2. Update WM
        3. Check Rules
        4. Fallback to RL
        5. Learn (if reward provided from previous step)
        """
        # 1. Perceive
        features = self.perception.perceive()
        
        # 2. Update WM
        if features.get('goal_visible'):
            # Store directional cues
            if features.get('goal_north'): self.wm.add('goal_north')
            if features.get('goal_south'): self.wm.add('goal_south')
            if features.get('goal_east'): self.wm.add('goal_east')
            if features.get('goal_west'): self.wm.add('goal_west')
        else:
            # If goal not visible, maybe we forget? Or rely on persistence.
            # For now, let's allow naturally decaying memory (via capacity) to handle it.
            pass
        
        # Construct State
        current_state = self.get_state_representation(features)
        
        # Learning Step (from previous action)
        if self.last_state is not None and self.last_action is not None and reward is not None:
             self.learner.learn(self.last_state, self.last_action, reward, current_state, done)
            
        # 3. Rule Arbitration
        suggested_action = self.rules.suggest_action(features, self.wm)
        
        # 4. Action Selection with Cognitive Control
        # If the rule suggests an action that RL has learned is BAD, we inhibit the rule.
        decision_source = "rl"
        action = None
        
        if suggested_action is not None:
             # Check what RL thinks of this "System 1" impulse
             q_values = self.learner.get_q_values(current_state)
             rule_q = q_values[suggested_action]
             max_q = np.max(q_values)
             
             # PHASE 3 LOGIC: Taboo List / Oscillation Prevention
             # If Rule suggests going back where we just came from, INHIBIT it.
             is_backtracking = False
             if self.last_action is not None:
                 # Check if opposite (0<->2, 1<->3)
                 if abs(suggested_action - self.last_action) == 2:
                     is_backtracking = True
             
             # Cognitive Control: 
             # 1. If Rule is strictly worse than RL best -> Inhibit
             # 2. If Rule causes immediate backtracking (Oscillation) -> Inhibit (Meta-Cognition)
             if is_backtracking:
                 decision_source = "inhibited_loop"
                 # Do NOT set action = suggested_action
             elif rule_q >= max_q - 0.1: # Tolerance of 0.1
                 action = suggested_action
                 decision_source = "rule"
             else:
                 decision_source = "inhibited_q"
        
        if action is None:
             action = self.learner.choose_action(current_state)
             if decision_source == "inhibited_rule":
                 decision_source = "rl_override"
             else:
                 decision_source = "rl"

             
        # 5. Global "Taboo" Check (Applies to RL too)
        # If we selected an action that is PURE backtracking, try to pick second best?
        # Or just force random.
        
        is_backtracking = False
        if self.last_action is not None and action is not None:
             if abs(action - self.last_action) == 2:
                 is_backtracking = True
                 
        if is_backtracking:
             # Override with random valid move that isn't the taboo one
             # Simple approach: Random choice
             decision_source = "taboo_override"
             action = self.env.action_space.sample()
             
        # 6. Stuck Check (Bonking)
        # If we are in the exact same state as before, we hit a wall.
        # Don't do the same thing again.
        if self.last_state == current_state and self.last_action == action:
            decision_source = "stuck_override"
            action = self.env.action_space.sample()

              
        # Store for next learn step
        self.last_state = current_state
        self.last_action = action
        
        return action, decision_source, features
