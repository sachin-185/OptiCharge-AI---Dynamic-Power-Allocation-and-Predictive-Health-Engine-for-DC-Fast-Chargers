import random
import datetime

DC_CHARGER_POWER = 240  # Total power per DC charger (kW)
RECTIFIER_MODULE_POWER = 30  # Power per rectifier module (kW)
# 240kW DC charger → 240/30 = 8 rectifiers per charger
def generate_dataset(weeks):
    print("=== EV Charging Station Dataset Generator ===")
    print(f"Generating {weeks} weeks ({weeks * 7} days) of charging sessions...")
    print("-" * 70)
    print(f"{'Session ID':<25} | {'User':<10} | {'Charger':<16} | {'Power (kW)':<10}")
    print("-" * 70)
    charger_ids = [
        "DC-CHG-240-001"
    ]
    
    # 10 unique users for the comparison plots
    user_ids = [f"USR-{i:03d}" for i in range(1, 11)]
    
    modules = [f"R{i}" for i in range(1, 9)] 
    output = []
    output.append("timestamp,user_id,charger_id,transaction_id,module_id,temp,live_capacity_kw,utilization_pct,power_on_flag")
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
            
            charger_id = random.choice(charger_ids)
            user_id = random.choice(user_ids)
            transaction_id = f"TXN-{timestamp.strftime('%Y%m%d')}-{session+1:03d}"
            
            if "DC-CHG-240" in charger_id:
                max_power = 240
                active_modules = random.randint(3, 8)
                modules_per_rectifier = 30
            elif "DC-CHG-180" in charger_id:
                max_power = 180
                active_modules = random.randint(3, 6)
                modules_per_rectifier = 30
            elif "DC-CHG-120" in charger_id:
                max_power = 120
                active_modules = random.randint(2, 4)
                modules_per_rectifier = 30
            else:
                max_power = 22
                active_modules = 1
                modules_per_rectifier = 22
            
            requested_power = random.choice([60, 90, 120, 150, 180, 240])
            requested_power = min(requested_power, max_power)
            actual_power = min(requested_power, active_modules * modules_per_rectifier)
            duration_minutes = random.randint(60, 120)
            
            print(f"{transaction_id:<25} | {user_id:<10} | {charger_id:<16} | {actual_power:<10}")
            
            base_temp = random.uniform(18, 25)
            base_capacity = modules_per_rectifier * random.uniform(0.95, 1.0)
            
            for minute in range(duration_minutes):
                session_timestamp = timestamp + datetime.timedelta(minutes=minute)
                temp_increase = (minute / duration_minutes) * 40
                
                for module_idx in range(8):
                    module_id = modules[module_idx]
                    
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
                    output.append(f"{session_timestamp.strftime('%Y-%m-%d %H:%M:%S')},{user_id},{charger_id},{transaction_id},{module_id},{temp},{live_capacity_kw},{utilization_pct},{power_on_flag}")
    
    filename = f"charging_data_{weeks}weeks.csv"
    with open(filename, "w") as f:
        f.write("\n".join(output))
    
    print("-" * 70)
    print(f"\nDataset generated successfully!")
    print(f"Total records: {len(output) - 1}")
    print(f"Duration: {weeks} weeks ({total_days} days)")
    print(f"File saved as: {filename}")
    return filename

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weeks', type=int, default=None)
    args = parser.parse_args()
    
    weeks = args.weeks
    if weeks is None:
        try:
            weeks = int(input("Enter duration in weeks (1-52): "))
        except ValueError:
            weeks = 1
    
    if weeks < 1 or weeks > 52:
        print("Please enter a value between 1 and 52:" )
    else:
        generate_dataset(weeks)
