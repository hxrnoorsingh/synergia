import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_wall():
    csv_path = "experiments/results/complexity_wall.csv"
    if not os.path.exists(csv_path):
        print("Data not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Simple Maze
    simple_data = df[df['Layout'].str.contains('Simple')]
    plt.plot(simple_data['Episode'], simple_data['Reward Smooth'], label='Simple Maze (Tabular)', color='green', linewidth=2)
    
    # Plot Complex Maze
    complex_data = df[df['Layout'].str.contains('Complex')]
    plt.plot(complex_data['Episode'], complex_data['Reward Smooth'], label='Complex Maze (Tabular Failure)', color='red', linewidth=2, linestyle='--')
    
    plt.title('The "Complexity Wall": Tabular RL Failure Scale', fontsize=16)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Average Reward (Moving Avg)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Annotate the gap
    plt.text(100, 5, "Fast Convergence", color='green', fontweight='bold')
    plt.text(200, -18, "Failure to Learn", color='red', fontweight='bold')
    
    output_path = "analysis/complexity_wall.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_wall()
