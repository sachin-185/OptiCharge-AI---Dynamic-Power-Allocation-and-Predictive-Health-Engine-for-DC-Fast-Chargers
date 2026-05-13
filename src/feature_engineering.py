import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

def engineer_features(input_file, output_file):
    print("EV Charging Station - Per-Rectifier Feature Engineering")
    print("=" * 70)
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records from {input_file}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    df = df.sort_values(['charger_id', 'transaction_id', 'module_id', 'timestamp'])  
    rectifier_features = []
    # Include user_id in the grouping metadata
    transactions = df.groupby(['charger_id', 'transaction_id', 'user_id']).size().reset_index()[['charger_id', 'transaction_id', 'user_id']]
    total_txns = len(transactions)
    print(f"Processing {total_txns} unique charging sessions...")
    
    all_modules = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
    
    for idx, (_, txn_row) in enumerate(transactions.iterrows()):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{total_txns} sessions...")        
        charger_id = txn_row['charger_id']
        transaction_id = txn_row['transaction_id']
        user_id = txn_row['user_id']
        session_data = df[(df['charger_id'] == charger_id) & (df['transaction_id'] == transaction_id)]
        session_data = session_data.sort_values('timestamp')
        
        session_start = session_data['timestamp'].min()
        session_end = session_data['timestamp'].max()
        session_duration = (session_end - session_start).total_seconds() / 60
        
        for mod in all_modules:
            mod_data = session_data[session_data['module_id'] == mod].sort_values('timestamp')
            
            if len(mod_data) == 0:
                continue
            
            features = {}
            features['charger_id'] = charger_id
            features['transaction_id'] = transaction_id
            features['user_id'] = user_id
            features['rectifier_id'] = mod
            features['session_start_time'] = session_start
            features['session_end_time'] = session_end
            features['session_duration_min'] = session_duration
            
            power_on_data = mod_data[mod_data['power_on_flag'] == 1]
            power_on_count = len(power_on_data)
            
            features['power_on_flag'] = mod_data['power_on_flag'].iloc[-1] if len(mod_data) > 0 else 0
            features['was_active'] = 1 if power_on_count > 0 else 0
            features['active_minutes'] = power_on_count
            
            if power_on_count > 0:
                features['live_capacity_kw'] = power_on_data['live_capacity_kw'].mean()
                features['live_capacity_kw_max'] = power_on_data['live_capacity_kw'].max()
                features['live_capacity_kw_min'] = power_on_data['live_capacity_kw'].min()
                features['live_capacity_kw_std'] = power_on_data['live_capacity_kw'].std() if len(power_on_data) > 1 else 0
                
                features['utilization_pct'] = power_on_data['utilization_pct'].mean()
                features['utilization_pct_max'] = power_on_data['utilization_pct'].max()
                features['utilization_pct_min'] = power_on_data['utilization_pct'].min()
                
                temps = power_on_data['temp']
                features['temp'] = temps.mean()
                features['temp_max'] = temps.max()
                features['temp_min'] = temps.min()
                features['temp_std'] = temps.std() if len(temps) > 1 else 0
                features['temp_start'] = temps.iloc[0]
                features['temp_end'] = temps.iloc[-1]
                features['delta_temp'] = features['temp_end'] - features['temp_start']
                
                temp_diffs = temps.diff().abs()
                features['max_temp_jump_1min'] = temp_diffs.max() if len(temp_diffs) > 0 else 0
                
                if len(temps) > 1:
                    x = np.arange(len(temps))
                    slope, _ = np.polyfit(x, temps.values, 1)
                    features['temp_slope'] = slope
                else:
                    features['temp_slope'] = 0
                
                features['temp_above_threshold'] = (temps > 50).sum()
                features['high_temp_ratio'] = features['temp_above_threshold'] / len(temps) * 100
            else:
                features['live_capacity_kw'] = 0
                features['live_capacity_kw_max'] = 0
                features['live_capacity_kw_min'] = 0
                features['live_capacity_kw_std'] = 0
                features['utilization_pct'] = 0
                features['utilization_pct_max'] = 0
                features['utilization_pct_min'] = 0
                features['temp'] = mod_data['temp'].mean()
                features['temp_max'] = mod_data['temp'].max()
                features['temp_min'] = mod_data['temp'].min()
                features['temp_std'] = 0
                features['temp_start'] = features['temp']
                features['temp_end'] = features['temp']
                features['delta_temp'] = 0
                features['max_temp_jump_1min'] = 0
                features['temp_slope'] = 0
                features['temp_above_threshold'] = 0
                features['high_temp_ratio'] = 0
            
            features['rank_score'] = features['live_capacity_kw'] * (100 - features['utilization_pct']) / 100           
            rectifier_features.append(features)
    
    result_df = pd.DataFrame(rectifier_features)
    result_df = add_history_features_rectifier(result_df)
    result_df = create_ranking_targets(result_df)
    result_df.to_csv(output_file, index=False)    
    print(f"\n{'=' * 70}")
    print("Feature Engineering Complete!")
    print(f"{'=' * 70}")
    print(f"Output file: {output_file}")
    print(f"Total features: {len(result_df.columns)}")
    print(f"Total rectifier records: {len(result_df)}")
    print(f"Unique rectifiers: {result_df['rectifier_id'].nunique()}")
    return result_df

def generate_preprocessed_minute_data(input_file, output_file):
    """
    Creates a minute-by-minute preprocessed dataset by cleaning raw data 
    and adding time-relative features for visualization.
    """
    print(f"\n--- Generating Preprocessed Minute-Level Data ---")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort and clean
    df = df.sort_values(['charger_id', 'transaction_id', 'module_id', 'timestamp'])   
    preprocessed_chunks = []
    
    # Group by session to calculate relative time
    for (charger_id, transaction_id), session_group in df.groupby(['charger_id', 'transaction_id']):
        session_start = session_group['timestamp'].min()
        session_group = session_group.copy()        
        # Calculate minutes into session
        session_group['minutes_into_session'] = (session_group['timestamp'] - session_start).dt.total_seconds() / 60
        preprocessed_chunks.append(session_group)    
    result_df = pd.concat(preprocessed_chunks, ignore_index=True)
    
    # Optional: Basic normalization or cleaning for visualization
    # (e.g., ensure we only plot if some power was actually drawn, or mark anomalies)
    
    result_df.to_csv(output_file, index=False)
    print(f"Preprocessed minute data saved to: {output_file}")
    return result_df

def add_history_features_rectifier(df):
    print("\n  Computing history/wear features per rectifier...")   
    df = df.sort_values(['charger_id', 'rectifier_id', 'session_start_time'])    
    history_features = []    
    for charger_id in df['charger_id'].unique():
        for rect_id in df['rectifier_id'].unique():
            rect_sessions = df[(df['charger_id'] == charger_id) & (df['rectifier_id'] == rect_id)].sort_values('session_start_time').copy()           
            if len(rect_sessions) == 0:
                continue
            
            for idx, (_, row) in enumerate(rect_sessions.iterrows()):
                current_time = row['session_start_time']
                
                last_24h = rect_sessions[(rect_sessions['session_start_time'] >= current_time - timedelta(hours=24)) & 
                                        (rect_sessions['session_start_time'] < current_time)]
                row['cumulative_runtime_last_24h'] = last_24h['session_duration_min'].sum()
                
                last_7d = rect_sessions[(rect_sessions['session_start_time'] >= current_time - timedelta(days=7)) & 
                                        (rect_sessions['session_start_time'] < current_time)]
                row['cumulative_runtime_last_7d'] = last_7d['session_duration_min'].sum()
                row['cumulative_energy_last_7d'] = (last_7d['live_capacity_kw'] * last_7d['session_duration_min']).sum() / 60
                row['avg_temp_last_7d'] = last_7d['temp'].mean() if len(last_7d) > 0 else 0
                row['max_temp_last_7d'] = last_7d['temp_max'].max() if len(last_7d) > 0 else 0
                row['count_high_temp_events_last_7d'] = (last_7d['temp_above_threshold'] > 0).sum()
                
                prev_sessions = rect_sessions[rect_sessions['session_start_time'] < current_time]
                if len(prev_sessions) > 0:
                    last_session = prev_sessions.iloc[-1]
                    time_since_last = (current_time - last_session['session_end_time']).total_seconds() / 60
                    row['cooldown_minutes_since_last_use'] = max(0, time_since_last)
                    row['last_session_temp_end'] = last_session['temp_end']
                    row['last_session_peak_temp'] = last_session['temp_max']
                else:
                    row['cooldown_minutes_since_last_use'] = 0
                    row['last_session_temp_end'] = 0
                    row['last_session_peak_temp'] = 0                
                history_features.append(row)    
    result_df = pd.DataFrame(history_features)
    return result_df

def create_ranking_targets(df):
    print("\n  Creating ranking targets...")    
    ranking_data = []    
    for (_, group) in df.groupby(['charger_id', 'transaction_id']):
        group = group.copy()
        group = group.sort_values('rank_score', ascending=False)
        group['target_rank'] = range(1, len(group) + 1)
        group['selected'] = (group['target_rank'] == 1).astype(int)
        ranking_data.append(group)
    result_df = pd.concat(ranking_data, ignore_index=True)
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Charging Station - Feature Engineering")
    parser.add_argument('--input', type=str, default='charging_data_1weeks.csv', help='Input raw data file')
    parser.add_argument('--output', type=str, default='engineered_features.csv', help='Output engineered features file')
    parser.add_argument('--minute_output', type=str, default='preprocessed_minute_data.csv', help='Output minute-level preprocessed data')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    
    if not os.path.exists(input_file):
        # Fallback check for common naming patterns if default is used
        if input_file == 'charging_data_1weeks.csv' and os.path.exists('charging_data_7weeks.csv'):
            input_file = 'charging_data_7weeks.csv'
            print(f"Default '{args.input}' not found. Falling back to '{input_file}'")
        else:
            print(f"Error: Input file '{input_file}' not found.")
            exit(1)

    # 1. Generate engineered features (per-rectifier session summaries)
    result = engineer_features(input_file, output_file)
    
    # 2. Generate minute-level data (for visualization)
    generate_preprocessed_minute_data(input_file, args.minute_output)
    print(f"\nSample output (first 5 rows):")
    print(result.head().to_string())