class RuleEngine:
    """
    Symbolic Rule Engine.
    Maps perception and working memory contents to suggested actions.
    """
    def __init__(self):
        pass

    def suggest_action(self, perception, wm):
        """
        Returns an action (int) if a rule fires, else None.
        Action mapping: 0: Up, 1: Right, 2: Down, 3: Left
        """
        # Rule 1: Goal is directly visible and path is clear -> Move to Goal
        if perception.get('goal_visible'):
            if perception.get('goal_directly_north') and not perception.get('wall_north'): return 0
            if perception.get('goal_directly_east') and not perception.get('wall_east'): return 1
            if perception.get('goal_directly_south') and not perception.get('wall_south'): return 2
            if perception.get('goal_directly_west') and not perception.get('wall_west'): return 3
            
            # Rule 2: Goal is generally visible -> Move in that general direction
            if perception.get('goal_north') and not perception.get('wall_north'): return 0
            if perception.get('goal_east') and not perception.get('wall_east'): return 1
            if perception.get('goal_south') and not perception.get('wall_south'): return 2
            if perception.get('goal_west') and not perception.get('wall_west'): return 3

        # Rule 3: Memory-based Persistence (System 2)
        # If I can't see the goal, but I remember where it was...
        if wm.contains('goal_north') and not perception.get('wall_north'): return 0
        if wm.contains('goal_east') and not perception.get('wall_east'): return 1
        if wm.contains('goal_south') and not perception.get('wall_south'): return 2
        if wm.contains('goal_west') and not perception.get('wall_west'): return 3

        # Rule 4: Avoid Walls (Reflexive)
        # This is tricky without a goal. 
        # For now, we leave this to RL or default random if no goal.
        
        return None
