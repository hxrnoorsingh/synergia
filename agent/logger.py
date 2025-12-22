import pandas as pd
import os

class TraceLogger:
    """
    Logs agent decisions, memory state, and performance metrics.
    """
    def __init__(self):
        self.trace = []

    def log_step(self, episode, step, features, wm_content, action, reward, decision_source):
        """
        Records a single simulation step.
        """
        entry = {
            'episode': episode,
            'step': step,
            'action': action,
            'reward': reward,
            'decision_source': decision_source,
            'goal_visible': features.get('goal_visible', False),
            'wall_north': features.get('wall_north', False),
            'wm_content': str(wm_content)
        }
        self.trace.append(entry)

    def save(self, filepath="experiment_log.csv"):
        """
        Saves the log to a CSV file.
        """
        df = pd.DataFrame(self.trace)
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Log saved to {filepath}")
        return df
