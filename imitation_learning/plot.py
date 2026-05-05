import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import glob
import numpy as np
from tbparse import SummaryReader

# --- CONFIGURATION ---
LOG_DIR = "./tensorboard/Imitation Learning-20260428-135240"       # Folder containing your tfevents subfolders
JSON_DIR = "./results"       # Folder containing your episode JSON files
METRICS = [
    'train_loss', 'valid_loss', 
    'train_accuracy', 'validation_accuracy',
    'train_accuracy_f', 'validation_accuracy_f'
]

def load_episode_results(json_folder):
    all_results = []
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Extract the raw data
            rewards = data.get('episode_rewards', [])
            
            # We calculate/verify stats here to ensure consistency
            res = {
                "Run": os.path.basename(file_path).replace('.json', ''),
                "Episodes": len(rewards),
                "Mean": np.mean(rewards),
                "Std": np.std(rewards),
                "Min": np.min(rewards),
                "Max": np.max(rewards)
            }
            all_results.append(res)
            
    return pd.DataFrame(all_results)

def build_report():
    # 1. Load the HPO/Episode Table
    hpo_df = load_episode_results(JSON_DIR)
    
    # 2. Load TensorBoard Data (using the binary reader to skip pkg_resources errors)
    reader = SummaryReader(LOG_DIR)
    df_events = reader.scalars
    
    # 3. Setup Plotting Grid
    num_metrics = len(METRICS)
    # +1 row for the table
    fig = plt.figure(figsize=(14, 5 * (num_metrics + 1)))
    gs = fig.add_gridspec(num_metrics + 1, 1, height_ratios=[1]*num_metrics + [0.6])

    # 4. Plot Training Curves
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[i])
        subset = df_events[df_events['tag'] == metric]
        
        if not subset.empty:
            # Seaborn automatically handles colors for different 'dir_name' runs
            sns.lineplot(data=subset, x='step', y='value', hue='dir_name', ax=ax)
            ax.set_title(f'Temporal Analysis: {metric}', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title='Run ID', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.text(0.5, 0.5, f"Metric '{metric}' not found in logs", ha='center')

    # 5. Add the Results Table at the bottom
    ax_table = fig.add_subplot(gs[-1])
    ax_table.axis('off')
    
    # Format the Mean ± Std column for the publication-style table
    display_df = hpo_df.copy()
    display_df['Performance (Mean ± Std)'] = display_df.apply(
        lambda x: f"{x['Mean']:.2f} ± {x['Std']:.2f}", axis=1
    )
    
    # Select columns to show in the final table
    table_data = display_df[['Run', 'Episodes', 'Performance (Mean ± Std)', 'Min', 'Max']]

    # Create the table
    tbl = ax_table.table(
        cellText=table_data.values, 
        colLabels=table_data.columns, 
        loc='center', 
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 2) # Stretch for readability

    plt.tight_layout()
    plt.savefig("hpo_final_report.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    build_report()