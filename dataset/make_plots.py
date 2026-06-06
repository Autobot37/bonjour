import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Geohash decoding helper
_B32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_B32_IDX = {c: i for i, c in enumerate(_B32)}

def decode_geohash(gh):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for c in gh:
        if c not in _B32_IDX:
            continue
        cd = _B32_IDX[c]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

def main():
    print("Loading data...")
    df = pd.read_csv("dataset/train.csv")
    
    print("Filtering for Day 48...")
    df_48 = df[df['day'] == 48].copy()
    
    # Calculate demand metrics per geohash
    print("Aggregating demand by geohash...")
    geo_stats = df_48.groupby('geohash').agg(
        mean_demand=('demand', 'mean'),
        sum_demand=('demand', 'sum'),
        records_count=('demand', 'count')
    ).reset_index()
    
    # Decode geohashes
    print("Decoding geohashes to coordinates...")
    lats = []
    lons = []
    for gh in geo_stats['geohash']:
        lat, lon = decode_geohash(gh)
        lats.append(lat)
        lons.append(lon)
    geo_stats['latitude'] = lats
    geo_stats['longitude'] = lons
    
    # Set up styling
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 20,
        'figure.facecolor': '#121212',
        'axes.facecolor': '#1e1e1e',
        'text.color': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#a0a0a0',
        'ytick.color': '#a0a0a0',
        'grid.color': '#333333'
    })
    
    # Create figure
    fig = plt.figure(figsize=(20, 15), facecolor='#121212')
    
    # 1. Geographic Distribution
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    scatter = ax1.scatter(
        geo_stats['longitude'], 
        geo_stats['latitude'], 
        c=geo_stats['mean_demand'], 
        cmap='magma', 
        s=geo_stats['mean_demand']*400 + 10, 
        alpha=0.8,
        edgecolors='none'
    )
    ax1.set_title("Geographic Demand Heatmap (Lat vs Lon)", pad=15)
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=ax1)
    cbar.set_label('Mean Demand', color='#e0e0e0', size=12)
    cbar.ax.yaxis.set_tick_params(color='#a0a0a0', labelcolor='#a0a0a0')
    
    # 2. Demand Profile (Sorted Geohashes)
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    sorted_stats = geo_stats.sort_values(by='mean_demand').reset_index(drop=True)
    ax2.plot(sorted_stats.index, sorted_stats['mean_demand'], color='#39FF14', linewidth=2.5)
    ax2.fill_between(sorted_stats.index, sorted_stats['mean_demand'], color='#39FF14', alpha=0.15)
    ax2.set_title("Demand Distribution Profile Across All 1,241 Geohashes", pad=15)
    ax2.set_xlabel("Sorted Geohash Index")
    ax2.set_ylabel("Mean Demand")
    ax2.set_xlim(0, len(sorted_stats))
    
    # 3. Top 15 Geohashes Bar Chart
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    top_15 = geo_stats.sort_values(by='mean_demand', ascending=False).head(15)
    sns.barplot(
        x='mean_demand', 
        y='geohash', 
        data=top_15, 
        ax=ax3, 
        hue='geohash',
        palette='viridis',
        legend=False
    )
    ax3.set_title("Top 15 Highest-Demand Geohashes", pad=15)
    ax3.set_xlabel("Mean Demand")
    ax3.set_ylabel("Geohash")
    
    # 4. Temporal Demand Variation (Aggregated)
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    df_48['tmin'] = df_48['timestamp'].map(lambda s: int(s.split(':')[0]) * 60 + int(s.split(':')[1]))
    temporal = df_48.groupby('tmin')['demand'].agg(['mean', 'median', 'std']).reset_index()
    # Smooth a bit or plot directly
    ax4.plot(temporal['tmin'] / 60.0, temporal['mean'], color='#00E5FF', label='Mean Demand', linewidth=2)
    ax4.fill_between(
        temporal['tmin'] / 60.0, 
        temporal['mean'] - 0.5 * temporal['std'], 
        temporal['mean'] + 0.5 * temporal['std'], 
        color='#00E5FF', 
        alpha=0.1,
        label='Mean ± 0.5 Std Dev'
    )
    ax4.plot(temporal['tmin'] / 60.0, temporal['median'], color='#FF007F', linestyle='--', label='Median Demand', linewidth=1.5)
    ax4.set_title("Temporal Demand Profile Across Day 48", pad=15)
    ax4.set_xlabel("Hour of Day")
    ax4.set_ylabel("Demand")
    ax4.set_xticks(np.arange(0, 25, 4))
    ax4.set_xlim(0, 24)
    ax4.legend(facecolor='#1e1e1e', edgecolor='#333333', labelcolor='#e0e0e0')
    
    plt.suptitle("Day 48 Demand Variation Analysis by Geohash", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ensure save directory exists
    output_dir = r"C:\Users\bagri\AppData\Local\Temp" # fallback if not set
    # Let's write to the requested artifact directory
    artifact_dir = r"C:\Users\bagri\.gemini\antigravity\brain\4f3a64cf-d452-4f38-80a8-8486623b7c84"
    if os.path.exists(artifact_dir):
        output_dir = artifact_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        
    output_path = os.path.join(output_dir, "demand_variation_day48.png")
    plt.savefig(output_path, dpi=300, facecolor='#121212', edgecolor='none')
    print(f"Plot successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
