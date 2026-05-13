import pandas as pd
import numpy as np
import random
import string
import datetime
from datetime import timedelta
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import feature_engineering
import tsa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DC_CHARGER_POWER = 240 # TOTAL DC CHARGER POWER
RECTIFIER_POWER = 30 # POWER OF EACH RECTIFIER
RECTIFIERS = [f"R{i}" for i in range(1, 9)]
CHARGER_IDS = ["DC-CHG-240-001"]

def generate_charging_data(weeks):
    print("=== EV Charging Station Dataset Generator ===")
    print(f"Generating {weeks} weeks ({weeks * 7} days) of charging sessions...")
    print("-" * 70)
    print(f"{'Session ID':<25} | {'Charger':<16} | {'Power (kW)':<10} | {'Duration (min)':<12}")
    print("-" * 70)
    output = []
    output.append("timestamp,charger_id,transaction_id,module_id,temp,live_capacity_kw,utilization_pct,power_on_flag")
    start_date = datetime.datetime(2026, 1, 1, 0, 0, 0)
    sessions_per_day = 5
    total_days = weeks * 7
    
    for day in range(total_days):
        current_date = start_date + datetime.timedelta(days=day)    
        num_sessions = random.randint(1, sessions_per_day)   
        for session in range(num_sessions):
            session_hour = random.randint(6, 22)
            session_minute = random.randint(0, 59)
            timestamp = current_date + datetime.timedelta(hours=session_hour, minutes=session_minute)
            charger_id = random.choice(CHARGER_IDS)
            transaction_id = f"TXN-{timestamp.strftime('%Y%m%d')}-{session+1:03d}"
            
            if "DC-CHG-240" in charger_id:
                max_power = 240
                active_modules = random.randint(3, 8)
            elif "DC-CHG-180" in charger_id:
                max_power = 180
                active_modules = random.randint(3, 6)
            elif "DC-CHG-120" in charger_id:
                max_power = 120
                active_modules = random.randint(2, 4)
            else:
                max_power = 22
                active_modules = 1
            modules_per_rectifier = 30 if max_power > 22 else 22
            requested_power = random.choice([60, 90, 120, 150, 180, 240])
            requested_power = min(requested_power, max_power)
            actual_power = min(requested_power, active_modules * modules_per_rectifier)
            duration_minutes = random.randint(60, 120)            
            print(f"{transaction_id:<25} | {charger_id:<16} | {actual_power:<10} | {duration_minutes:<12}")           
            base_temp = random.uniform(18, 25)
            base_capacity = modules_per_rectifier * random.uniform(0.95, 1.0)
            
            for minute in range(duration_minutes):
                session_timestamp = timestamp + datetime.timedelta(minutes=minute)
                temp_increase = (minute / duration_minutes) * 40                
                for module_idx in range(8):
                    module_id = RECTIFIERS[module_idx]                   
                    if module_idx < active_modules:
                        power_on_flag = 1
                        live_capacity_kw = base_capacity + (minute * 0.05) + random.uniform(-0.3, 0.3)
                        live_capacity_kw = min(30.0, live_capacity_kw)
                        
                        max_available = active_modules * 30
                        utilization = (actual_power / max_available) * 100
                        utilization_pct = utilization + random.uniform(-3, 3)
                        utilization_pct = max(60, min(102, utilization_pct))
                        
                        module_temp_offset = (module_idx / active_modules) * 8
                        temp = base_temp + temp_increase + module_temp_offset + random.uniform(-1, 1)
                        temp = min(75, temp)
                    else:
                        power_on_flag = 0
                        live_capacity_kw = 0.0
                        utilization_pct = 0.0
                        temp = random.uniform(18, 28)
                    output.append(f"{session_timestamp.strftime('%Y-%m-%d %H:%M:%S')},{charger_id},{transaction_id},{module_id},{temp},{live_capacity_kw},{utilization_pct},{power_on_flag}")
    
    filename = f"charging_data_{weeks}weeks.csv"
    with open(filename, "w") as f:
        f.write("\n".join(output))
    
    print("-" * 70)
    print(f"\nDataset generated successfully: {filename}")
    return filename

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
    return pd.DataFrame(history_features)

def engineer_features(input_file, output_file):
    print("\n=== Feature Engineering Pipeline ===")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values(['charger_id', 'transaction_id', 'module_id', 'timestamp'])  
    rectifier_features = []
    transactions = df.groupby(['charger_id', 'transaction_id']).size().reset_index()[['charger_id', 'transaction_id']]
    total_txns = len(transactions)
    
    for idx, (_, txn_row) in enumerate(transactions.iterrows()):
        if idx % 500 == 0: print(f"  Progress: {idx}/{total_txns} sessions...")        
        charger_id, transaction_id = txn_row['charger_id'], txn_row['transaction_id']        
        session_data = df[(df['charger_id'] == charger_id) & (df['transaction_id'] == transaction_id)].sort_values('timestamp')
        
        session_start, session_end = session_data['timestamp'].min(), session_data['timestamp'].max()
        session_duration = (session_end - session_start).total_seconds() / 60
        
        for mod in RECTIFIERS:
            mod_data = session_data[session_data['module_id'] == mod].sort_values('timestamp')
            if len(mod_data) == 0: continue
            
            features = {'charger_id': charger_id, 'transaction_id': transaction_id, 'rectifier_id': mod,
                        'session_start_time': session_start, 'session_end_time': session_end, 'session_duration_min': session_duration}
            
            power_on_data = mod_data[mod_data['power_on_flag'] == 1]
            power_on_count = len(power_on_data)
            features['power_on_flag'] = mod_data['power_on_flag'].iloc[-1]
            features['was_active'] = 1 if power_on_count > 0 else 0
            features['active_minutes'] = power_on_count
            
            if power_on_count > 0:
                features['live_capacity_kw'] = power_on_data['live_capacity_kw'].mean()
                features['live_capacity_kw_max'] = power_on_data['live_capacity_kw'].max()
                features['live_capacity_kw_min'] = power_on_data['live_capacity_kw'].min()
                features['utilization_pct'] = power_on_data['utilization_pct'].mean()
                features['utilization_pct_max'] = power_on_data['utilization_pct'].max()
                
                temps = power_on_data['temp']
                features['temp'] = temps.mean()
                features['temp_max'] = temps.max()
                features['temp_min'] = temps.min()
                features['temp_end'] = temps.iloc[-1]
                features['delta_temp'] = features['temp_end'] - temps.iloc[0]
                
                if len(temps) > 1:
                    features['temp_slope'] = np.polyfit(np.arange(len(temps)), temps.values, 1)[0]
                else: features['temp_slope'] = 0
                
                features['temp_above_threshold'] = (temps > 50).sum()
                features['high_temp_ratio'] = features['temp_above_threshold'] / len(temps) * 100
            else:
                for col in ['live_capacity_kw', 'live_capacity_kw_max', 'live_capacity_kw_min', 'utilization_pct', 'utilization_pct_max', 'temp_slope', 'temp_above_threshold', 'high_temp_ratio', 'delta_temp']:
                    features[col] = 0
                features['temp'] = mod_data['temp'].mean()
                features['temp_max'] = mod_data['temp'].max()
                features['temp_min'] = mod_data['temp'].min()
                features['temp_end'] = features['temp']
            
            features['rank_score'] = features['live_capacity_kw'] * (100 - features['utilization_pct']) / 100
            rectifier_features.append(features)
    
    result_df = pd.DataFrame(rectifier_features)
    result_df = add_history_features_rectifier(result_df)
    
    ranking_data = []
    for (_, group) in result_df.groupby(['charger_id', 'transaction_id']):
        group = group.copy().sort_values('rank_score', ascending=False)
        group['target_rank'] = range(1, len(group) + 1)
        ranking_data.append(group)
    
    result_df = pd.concat(ranking_data, ignore_index=True)
    result_df.to_csv(output_file, index=False)
    print(f"Feature engineering complete: {output_file}")
    return output_file

def scalar_normalization(input_file, output_file="normalized_data.csv"):
    print("\n=== Scalar Normalization ===")
    df = pd.read_csv(input_file)
    df['Temp_Score'] = 1 - (df['temp'] - df['temp'].min()) / (df['temp'].max() - df['temp'].min() + 1)
    df['Util_Score'] = 1 - (df['utilization_pct'] - df['utilization_pct'].min()) / (df['utilization_pct'].max() - df['utilization_pct'].min() + 1)
    df['Efficiency'] = (99 - ((df['temp'] - 43) / (72 - 43 + 1) * 45)).clip(lower=0)
    df.to_csv(output_file, index=False)
    print(f"Normalization complete: {output_file}")
    return output_file

def time_series_analysis(input_file):
    print("\n=== Time Series Analysis (Visuals) ===")
    df = pd.read_csv(input_file)
    for session in df['transaction_id'].unique()[:2]:
        session_data = df[df['transaction_id'] == session]
        plt.figure(figsize=(10, 4))
        plt.plot(session_data['rectifier_id'], session_data['temp'], marker='o', color='red', label='Temp Profile')
        plt.xlabel('Rectifier ID')
        plt.ylabel('Temperature (C)')
        plt.title(f'Thermal Profile - Session {session}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'temp_profile_{session}.png')
        plt.close()
    print("Thermal profile plots saved.")

def main(weeks=1):
    print("\n" + "="*60)
    print("PHASE 1: DATA PREPARATION PIPELINE")
    print("="*60)
    
    raw_file = generate_charging_data(weeks)
    features_file = "engineered_features.csv"
    engineer_features(raw_file, features_file)
    scalar_normalization(features_file)
    preprocessed_ts_file = "preprocessed_minute_data.csv"
    feature_engineering.generate_preprocessed_minute_data(raw_file, preprocessed_ts_file)
    tsa.run_tsa_visuals(preprocessed_ts_file)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("Proceed to run code2.py for Training & Simulation")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weeks', type=int, default=1, help='Data generation duration (weeks)')
    args = parser.parse_args()
    main(args.weeks)
