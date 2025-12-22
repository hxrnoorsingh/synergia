import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_training_curve():
    # Load data
    data_path = os.path.join("experiments", "results", "deep_hybrid.csv")
    if not os.path.exists(data_path):
        print("No data found at", data_path)
        return

    df = pd.read_csv(data_path)
    
    # Setup Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # Plot Reward
    sns.lineplot(data=df, x="Episode", y="Reward", marker="o", color="b", label="Reward (Every 100th Ep)")
    
    # Add annotations for Phases
    plt.axvline(x=3000, color='r', linestyle='--', label="Curriculum Switch (50%)")
    plt.text(1500, -20, "Random Exploration Phase", ha='center', color='r')
    plt.text(4500, -20, "Test Phase (Fixed Start)", ha='center', color='r')
    
    plt.title("Deep Hybrid Agent Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    
    # Save
    output_path = os.path.join("analysis", "training_curve.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_training_curve()
