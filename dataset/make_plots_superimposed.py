import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading data...")
    df = pd.read_csv("dataset/train.csv")
    
    # Process timestamps to tmin
    print("Pre-processing timestamps...")
    df['tmin'] = df['timestamp'].map(lambda s: int(s.split(':')[0]) * 60 + int(s.split(':')[1]))
    
    # Set up modern high-quality styling
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 18,
        'figure.facecolor': '#0F172A',  # Dark slate background
        'axes.facecolor': '#1E293B',    # Dark slate axes container
        'text.color': '#F8FAFC',        # Off-white text
        'axes.labelcolor': '#F8FAFC',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'grid.color': '#334155'         # Subtle grid lines
    })
    
    # Create a figure with 2 subplots (vertical layout, Plot 1 above Plot 2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14), facecolor='#0F172A')
    
    # ============================================================
    # PLOT 1: Morning Comparison (Day 48 vs Day 49, Hours 0, 1, 2)
    # ============================================================
    print("Processing Plot 1 (Morning Comparison)...")
    # Filters for morning hours 0, 1, 2 (tmin < 180)
    df_m48 = df[(df['day'] == 48) & (df['tmin'] < 180)].copy()
    df_m49 = df[(df['day'] == 49) & (df['tmin'] < 180)].copy()
    
    # Aggregate and pivot for fast plotting of all individual trajectories
    pivot_m48 = df_m48.groupby(['tmin', 'geohash'])['demand'].mean().unstack()
    pivot_m49 = df_m49.groupby(['tmin', 'geohash'])['demand'].mean().unstack()
    
    # Plot Day 48 individual morning trajectories in Cyan
    lines_48 = ax1.plot(pivot_m48.index, pivot_m48.values, color='#0EA5E9', alpha=0.03, linewidth=0.5)
    # Plot Day 49 individual morning trajectories in Rose
    lines_49 = ax1.plot(pivot_m49.index, pivot_m49.values, color='#F43F5E', alpha=0.03, linewidth=0.5)
    
    # Calculate and plot the average/mean trajectories (thicker lines)
    avg_m48 = df_m48.groupby('tmin')['demand'].mean()
    avg_m49 = df_m49.groupby('tmin')['demand'].mean()
    
    line_avg48, = ax1.plot(avg_m48.index, avg_m48.values, color='#38BDF8', linewidth=3.0, label='Day 48 Morning Mean')
    line_avg49, = ax1.plot(avg_m49.index, avg_m49.values, color='#FB7185', linewidth=3.0, label='Day 49 Morning Mean')
    
    ax1.set_title("Superimposed Morning Demand Trajectories (Hours 0, 1, 2: 00:00 - 02:45)", pad=15)
    ax1.set_xlabel("Time of Day")
    ax1.set_ylabel("Demand")
    
    # Set tick labels to time formats
    morning_ticks = [0, 30, 60, 90, 120, 150, 180]
    morning_labels = ["00:00", "00:30", "01:00", "01:30", "02:00", "02:30", "03:00"]
    ax1.set_xticks(morning_ticks)
    ax1.set_xticklabels(morning_labels)
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, 1.0)
    
    # Legend displaying only the mean lines to keep it clean
    ax1.legend(handles=[line_avg48, line_avg49], facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', loc='upper right')
    
    # ============================================================
    # PLOT 2: Day 48 Full Trajectory (Zoomed out / Smooth)
    # ============================================================
    print("Processing Plot 2 (Day 48 Full Trajectory)...")
    df_48 = df[df['day'] == 48].copy()
    
    # Aggregate and pivot for fast plotting of all individual trajectories
    pivot_48 = df_48.groupby(['tmin', 'geohash'])['demand'].mean().unstack()
    
    # Plot all geohashes for Day 48 in Emerald/Mint
    lines_all48 = ax2.plot(pivot_48.index, pivot_48.values, color='#10B981', alpha=0.015, linewidth=0.3)
    
    # Calculate and plot the average/mean trajectory
    avg_48 = df_48.groupby('tmin')['demand'].mean()
    line_avg_all48, = ax2.plot(avg_48.index, avg_48.values, color='#34D399', linewidth=3.0, label='Day 48 Overall Mean')
    
    ax2.set_title("Superimposed Day 48 Full Trajectories (All 1,241 Geohashes)", pad=15)
    ax2.set_xlabel("Time of Day")
    ax2.set_ylabel("Demand")
    
    # Set tick labels to time formats
    full_ticks = np.arange(0, 1441, 120)
    full_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]
    ax2.set_xticks(full_ticks)
    ax2.set_xticklabels(full_labels)
    ax2.set_xlim(0, 1440)
    ax2.set_ylim(0, 1.0)
    
    # Legend displaying only the mean line to keep it clean
    ax2.legend(handles=[line_avg_all48], facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', loc='upper right')
    
    plt.suptitle("Traffic Demand Trajectories by Geohash", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the output image
    artifact_dir = r"C:\Users\bagri\.gemini\antigravity\brain\4f3a64cf-d452-4f38-80a8-8486623b7c84"
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
        
    output_path = os.path.join(artifact_dir, "demand_trajectories_superimposed.png")
    plt.savefig(output_path, dpi=300, facecolor='#0F172A', edgecolor='none')
    print(f"Plot successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
