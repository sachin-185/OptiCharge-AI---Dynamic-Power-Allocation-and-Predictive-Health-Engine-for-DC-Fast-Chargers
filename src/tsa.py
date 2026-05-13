import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import numpy as np

def run_tsa_visuals(input_file="preprocessed_minute_data.csv"):
    print(f"\n=== Detailed Session Comparison Visuals (Module: tsa.py) ===")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure preprocessed minute data is generated.")
        return
        
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Identify unique chargers
    chargers = df['charger_id'].unique()
    print(f"  Generating detailed comparison plots for {len(chargers)} chargers...")
    
    rectifiers = [f'R{i}' for i in range(1, 9)]
    
    for charger in chargers:
        charger_data = df[df['charger_id'] == charger]
        
        # Get unique transactions for this charger, sorted by time (latest first)
        all_sessions = charger_data.groupby('transaction_id')['timestamp'].min().sort_values(ascending=False).index.tolist()
        
        # We will show the latest 5 sessions to represent "each session" analysis
        sessions_to_plot = all_sessions[:5]
        
        # Create a 4x2 grid for the 8 rectifiers
        fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True)
        axes = axes.flatten()
        
        # Define a color map for sessions
        colors = plt.cm.viridis(np.linspace(0, 1, len(sessions_to_plot)))
        
        for i, rect in enumerate(rectifiers):
            ax = axes[i]
            
            for s_idx, session_id in enumerate(sessions_to_plot):
                session_data = charger_data[(charger_data['transaction_id'] == session_id) & 
                                            (charger_data['module_id'] == rect)].sort_values('minutes_into_session')
                
                if not session_data.empty:
                    alpha = 1.0 if s_idx == 0 else 0.6 # Highlight the latest session
                    linewidth = 2 if s_idx == 0 else 1
                    label = f"Session {session_id}" if i == 0 else "" # Only label first subplot to save space
                    ax.plot(session_data['minutes_into_session'], session_data['temp'], 
                            label=label, color=colors[s_idx], linewidth=linewidth, alpha=alpha)
            
            ax.set_title(f'Rectifier: {rect}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 100) # Fixed Y-axis as requested
            ax.set_ylabel('Temp (°C)')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if i >= 6: # Only bottom plots get X label
                ax.set_xlabel('Minutes into Session')

        # Add global legend for the sessions
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize='small', title="Latest Sessions")
        
        plt.suptitle(f'Charger {charger}: Detailed Thermal Trends (Per Rectifier)', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plot_name = f'detailed_session_comparison_{charger}.png'
        plt.savefig(plot_name)
        plt.close()
        print(f"  - Generated grid plot for {charger}: {plot_name}")

    print("  Detailed Session Visuals complete.")

if __name__ == "__main__":
    run_tsa_visuals()