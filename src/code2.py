import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import ndcg_score
from sklearn.ensemble import IsolationForest
from typing import Tuple, List, Dict
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RectifierRankingConfig:
    """Configuration for the ranking model"""
    FEATURES = [
        'live_capacity_kw', 'live_capacity_kw_max', 'live_capacity_kw_min', 'utilization_pct', 'utilization_pct_max',
        'temp', 'temp_max', 'temp_min', 'temp_end', 'delta_temp', 'temp_slope', 'temp_above_threshold', 'high_temp_ratio',
        'session_duration_min', 'active_minutes', 'power_on_flag', 'was_active',
        'cumulative_runtime_last_24h', 'cumulative_runtime_last_7d', 'cumulative_energy_last_7d', 'avg_temp_last_7d', 'max_temp_last_7d', 'count_high_temp_events_last_7d',
        'cooldown_minutes_since_last_use', 'last_session_temp_end', 'last_session_peak_temp',
        'anomaly_score'
    ]
    
    XGBOOST_PARAMS = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg',
        'learning_rate': 0.05,
        'max_depth': 3,
        'seed': 42,
        'verbosity': 0,
        'early_stopping_rounds': 10
    }
    NUM_BOOST_ROUNDS = 100
    TEST_SIZE = 0.2

class DataProcessor:
    """Handles data preparation for XGBoost"""
    def __init__(self, config: RectifierRankingConfig):
        self.config = config
    
    def prepare(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading engineered features from {filepath}")
        df = pd.read_csv(filepath)
        
        df['query_id'] = df.groupby(['charger_id', 'transaction_id']).ngroup()
        
        df_active = df[df['was_active'] == 1].copy()
        logger.info(f"Prepared {len(df_active)} active records across {df['query_id'].nunique()} sessions")
        return df_active

class AnomalyDetector:
    """Unsupervised Health Guard using Isolation Forest"""
    def __init__(self, features: List[str]):
        self.features = [f for f in features if f != 'anomaly_score']
        self.model = IsolationForest(contamination=0.05, random_state=42)
        
    def train(self, df: pd.DataFrame):
        logger.info("Training Isolation Forest Anomaly Detector...")
        X = df[self.features].fillna(0)
        self.model.fit(X)
        return self
        
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].fillna(0)
        df['anomaly_score'] = self.model.decision_function(X)     
        predictions = self.model.predict(X)
        df['is_healthy'] = (predictions == 1).astype(int)
        
        if 'transaction_id' in df.columns and 'temp' in df.columns:
            session_medians = df.groupby('transaction_id')['temp'].transform('median')
            override_mask = (predictions == -1) & (df['temp'] < session_medians)
            df.loc[override_mask, 'is_healthy'] = 1
            
            if 'temp_max' in df.columns:
                thermal_mask = df['temp_max'] > 90.0
                df.loc[thermal_mask, 'is_healthy'] = 0

            if 'was_active' in df.columns and 'live_capacity_kw' in df.columns:
                active_mask = df['was_active'] == 1
                session_mean_cap = df[active_mask].groupby('transaction_id')['live_capacity_kw'].transform('mean')
                df.loc[active_mask, 'session_mean_cap'] = session_mean_cap
                
                variance_mask = active_mask & (
                    (abs(df['live_capacity_kw'] - df['session_mean_cap']) / (df['session_mean_cap'] + 1e-9) > 0.10) |
                    (df['live_capacity_kw'] == 0)
                )
                df.loc[variance_mask, 'is_healthy'] = 0
                df = df.drop(columns=['session_mean_cap'])              
        return df

class RankingModelTrainer:
    """Handles XGBoost LambdaMART training"""
    def __init__(self, config: RectifierRankingConfig):
        self.config = config
        self.model = None

    def train(self, df: pd.DataFrame):
        unique_queries = df['query_id'].unique()
        np.random.seed(42)
        train_queries = np.random.choice(unique_queries, size=int(len(unique_queries) * (1-self.config.TEST_SIZE)), replace=False)
        
        df_train = df[df['query_id'].isin(train_queries)]
        df_test = df[~df['query_id'].isin(train_queries)]
        
        def to_dmatrix(data):
            X = data[self.config.FEATURES].fillna(0).values
            y = data['target_rank'].values
            groups = data.groupby('query_id').size().values
            return xgb.DMatrix(X, label=y, group=groups)
        
        dtrain = to_dmatrix(df_train)
        dtest = to_dmatrix(df_test)
        
        logger.info("Training XGBoost Ranker........")
        self.model = xgb.train(self.config.XGBOOST_PARAMS, dtrain, num_boost_round=self.config.NUM_BOOST_ROUNDS, evals=[(dtest, 'eval')], verbose_eval=False)
        return self.model

class RectifierRecommender:
    """Main interface for recommendations"""
    def __init__(self, model, detector, config):
        self.model = model
        self.detector = detector
        self.config = config

    def predict(self, session_data, top_n):
        session_data = self.detector.score(session_data.copy())
        X = session_data[self.config.FEATURES].fillna(0).values
        dmat = xgb.DMatrix(X)
        scores = self.model.predict(dmat)     
        result = session_data.copy()
        result['ai_score'] = scores
        return result.sort_values('ai_score', ascending=False).head(top_n)

def calculate_health_and_efficiency(row):
    temp_score = max(0, 100 - (row['temp'] - 20) * 2)
    health = (0.7 * temp_score) + (0.3 * 80)
    efficiency = max(0, 98 - (row['temp'] - 40) * 0.5)
    return health, efficiency

def run_interactive_terminal(recommender, full_data):
    print("="*70)
    print("WELCOME")
    print("="*70)
    while True:
        try:
            print("\n" + "="*50)
            print("Select Scenario:")
            print("1. Two EVs plug in at the same time")
            print("2. One EV plugs in, later a second EV arrives")
            print("3. Exit")
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == '3' or choice == 'exit':
                break
                
            if choice not in ['1', '2']:
                print("Invalid choice.")
                continue

            sample_session = full_data[full_data['charger_id'] == 'DC-CHG-240-001'].sample(1)['transaction_id'].iloc[0]
            current_state = full_data[full_data['transaction_id'] == sample_session]
            all_recommendations = recommender.predict(current_state, top_n=len(current_state))
            
            def calculate_and_allocate(req1, req2):
                if req1 < 0 or req2 < 0 or (req1 == 0 and req2 == 0):
                    print("Invalid power request.")
                    return

                total_modules = 8
                if req1 + req2 <= 240:
                    quota1 = int(np.ceil(req1 / 30))
                    quota2 = int(np.ceil(req2 / 30))
                else:
                    if req1 <= 120 and req2 > 120:
                        quota1 = int(np.ceil(req1 / 30))
                        quota2 = total_modules - quota1
                    elif req2 <= 120 and req1 > 120:
                        quota2 = int(np.ceil(req2 / 30))
                        quota1 = total_modules - quota2
                    else:
                        quota1 = 4
                        quota2 = 4
                
                quota1 = min(total_modules, max(0, quota1))
                quota2 = min(total_modules - quota1, max(0, quota2))

                ev1_assigned = []
                ev2_assigned = []
                for idx, row in all_recommendations.iterrows():
                    if len(ev1_assigned) < quota1 and (len(ev1_assigned) <= len(ev2_assigned) or len(ev2_assigned) == quota2):
                        ev1_assigned.append((len(ev1_assigned)+len(ev2_assigned)+1, row))
                    elif len(ev2_assigned) < quota2:
                        ev2_assigned.append((len(ev1_assigned)+len(ev2_assigned)+1, row))
                    else:
                        break
                        
                def allocate_and_print(assigned_list, req_kw, plug_name):
                    if not assigned_list or req_kw == 0:
                        return
                    best_k = len(assigned_list)
                    power_per_rect = req_kw / best_k if best_k > 0 else 0
                    for k in range(1, len(assigned_list) + 1):
                        selected = assigned_list[:k]
                        ppr = req_kw / k
                        total_pos = sum(min(ppr, r['live_capacity_kw']) for _, r in selected)
                        if total_pos >= (req_kw - 0.1):
                            best_k = k
                            power_per_rect = ppr
                            break
                    
                    final_selection = assigned_list[:best_k]
                    print(f"\n--- {plug_name} OPTIMIZED ALLOCATION ---".center(75))
                    print(f"{'Rank':<5}{'Rectifier':<11}{'Capacity(kW)':<14}{'Output(kW)':<12}{'Temp(C)':<10}{'Health Guard':<15}{'Efficiency':<12}")
                    print("-" * 80)
                    total_alloc = 0
                    for orig_rank, row in final_selection:
                        _, e = calculate_health_and_efficiency(row)
                        guard_status = "HEALTHY" if row['is_healthy'] == 1 else "SUSPECT"
                        actual_out = min(power_per_rect, row['live_capacity_kw'])
                        total_alloc += actual_out
                        print(f"{orig_rank:<5}{row['rectifier_id']:<11}{row['live_capacity_kw']:<14.1f}{actual_out:<12.1f}{row['temp']:<10.1f}{guard_status:<15}{e:<12.1f}")
                    print("-" * 80)
                    print(f"{plug_name} Allocated: {total_alloc:.1f} kW / {req_kw:.1f} kW requested")

                allocate_and_print(ev1_assigned, req1, "PLUG A (EV 1)")
                allocate_and_print(ev2_assigned, req2, "PLUG B (EV 2)")

            if choice == '1':
                print(f"\nAvailable Charger: DC-CHG-240-001 (8 Modules, Max 240kW)")
                val1 = input("Plug A - EV 1 Power Request in kW: ").strip()
                val2 = input("Plug B - EV 2 Power Request in kW: ").strip()
                calculate_and_allocate(float(val1), float(val2))
                
            elif choice == '2':
                print(f"\n--- TIME: T=0 (Charger Idle) ---")
                val1 = input("EV 1 arrives! Power Request in kW: ").strip()
                req1 = float(val1)
                calculate_and_allocate(req1, 0)
                
                print("\n... time passes ...")
                ans = input("Does EV 2 arrive at Plug B? (y/n): ").strip().lower()
                if ans == 'y':
                    val2 = input("EV 2 arrives! Power Request in kW: ").strip()
                    req2 = float(val2)
                    print("\n>> MID-SESSION REALLOCATION TRIGGERED <<")
                    calculate_and_allocate(req1, req2)
                else:
                    print("EV 1 finishes charging alone.")

        except ValueError:
            print("Please enter a valid power input.")
        except Exception as e:
            print(f"Error: {e}")

def main(input_file, **kwargs):
    config = RectifierRankingConfig()
    processor = DataProcessor(config)
    try:
        data = processor.prepare(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run code1.py first.")
        return
    detector = AnomalyDetector(config.FEATURES).train(data)
    data = detector.score(data) 
    trainer = RankingModelTrainer(config)
    model = trainer.train(data)    
    recommender = RectifierRecommender(model, detector, config)
    if not kwargs.get('skip_simulation', False):
        run_interactive_terminal(recommender, data)
    else:
        print("Model training complete. Simulation skipped as per pipeline request.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='engineered_features.csv', help='Input feature file')
    args = parser.parse_args()
    main(args.input)
