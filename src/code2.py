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
    # Features produced by code1.py
    FEATURES = [
        # Efficiency & Capacity
        'live_capacity_kw', 'live_capacity_kw_max', 'live_capacity_kw_min', 'utilization_pct', 'utilization_pct_max',
        # Thermal Performance
        'temp', 'temp_max', 'temp_min', 'temp_end', 'delta_temp', 'temp_slope', 'temp_above_threshold', 'high_temp_ratio',
        # Current Session Metrics
        'session_duration_min', 'active_minutes', 'power_on_flag', 'was_active',
        # Historical Health
        'cumulative_runtime_last_24h', 'cumulative_runtime_last_7d', 'cumulative_energy_last_7d', 'avg_temp_last_7d', 'max_temp_last_7d', 'count_high_temp_events_last_7d',
        # Recovery Metrics
        'cooldown_minutes_since_last_use', 'last_session_temp_end', 'last_session_peak_temp',
        # AI Internal
        'anomaly_score'
    ]
    
    XGBOOST_PARAMS = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@3',
        'eta': 0.1,
        'max_depth': 5,
        'seed': 42,
        'verbosity': 0
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
        
        # Ensure query IDs for ranking groups
        df['query_id'] = df.groupby(['charger_id', 'transaction_id']).ngroup()
        
        # We only train on rectifiers that were actually active/available
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
        # -1 for anomaly, 1 for normal -> we use decision_function for ranking
        df['anomaly_score'] = self.model.decision_function(X)
        
        # Initial prediction: 1 for normal, -1 for anomaly
        predictions = self.model.predict(X)
        df['is_healthy'] = (predictions == 1).astype(int)
        
        # ONE-SIDED HEALTH GUARD OVERRIDE
        # If an anomaly is "too good" (e.g., unusually cool), override to healthy
        if 'transaction_id' in df.columns and 'temp' in df.columns:
            session_medians = df.groupby('transaction_id')['temp'].transform('median')
            # Condition: It is an anomaly (-1) BUT its temperature is below the session median
            override_mask = (predictions == -1) & (df['temp'] < session_medians)
            df.loc[override_mask, 'is_healthy'] = 1
            
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
        
        logger.info("Training XGBoost Ranker....")
        self.model = xgb.train(self.config.XGBOOST_PARAMS, dtrain, num_boost_round=self.config.NUM_BOOST_ROUNDS, evals=[(dtest, 'eval')], verbose_eval=False)
        return self.model

class RectifierRecommender:
    """Main interface for recommendations"""
    def __init__(self, model, detector, config):
        self.model = model
        self.detector = detector
        self.config = config

    def predict(self, session_data, top_n):
        # 1. Apply Health Guard (Anomaly Detection)
        session_data = self.detector.score(session_data.copy())
        
        # 2. Predict Ranking
        X = session_data[self.config.FEATURES].fillna(0).values
        dmat = xgb.DMatrix(X)
        scores = self.model.predict(dmat)
        
        result = session_data.copy()
        result['ai_score'] = scores
        return result.sort_values('ai_score', ascending=False).head(top_n)

def calculate_health_and_efficiency(row):
    # Health based on temp and active minutes
    temp_score = max(0, 100 - (row['temp'] - 20) * 2)
    health = (0.7 * temp_score) + (0.3 * 80) # Simplified health
    efficiency = max(0, 98 - (row['temp'] - 40) * 0.5)
    return health, efficiency

def run_interactive_terminal(recommender, full_data):
    print("="*70)
    print("PLUGZMART AI DC CHARGER SIMULATOR")
    print("="*70)
    
    while True:
        try:
            print(f"\nAvailable Charger: DC-CHG-240-001 (Max 240kW)")
            val = input("Enter requested power in kW (or 'exit' to quit): ").strip().lower()
            if val == 'exit': break
            
            requested_kw = float(val)
            if not (0 < requested_kw <= 240):
                print("Invalid range (0-240kW).")
                continue
                
            num_rects = int(np.ceil(requested_kw / 30))           
            # Simulate a "current state" by picking a random recent session's data
            sample_session = full_data[full_data['charger_id'] == 'DC-CHG-240-001'].sample(1)['transaction_id'].iloc[0]
            current_state = full_data[full_data['transaction_id'] == sample_session]
            
            recommendations = recommender.predict(current_state, top_n=num_rects)           
            print("\n" + "--- AI OPTIMIZED ALLOCATION ---".center(70))
            print(f"{'Rank':<5}{'Rect':<7}{'Cap(kW)':<10}{'Temp(C)':<10}{'Health Guard':<15}{'Efficiency':<12}")
            print("-" * 75)
            
            total_cap = 0
            for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                _, e = calculate_health_and_efficiency(row)
                guard_status = "HEALTHY" if row['is_healthy'] == 1 else "SUSPECT"
                print(f"{i:<5}{row['rectifier_id']:<7}{row['live_capacity_kw']:<10.1f}{row['temp']:<10.1f}{guard_status:<15}{e:<12.1f}")
                total_cap += row['live_capacity_kw']
                
            print("-" * 75)
            print(f"Total Allocated: {total_cap:.1f} kW / {requested_kw} kW requested")
            print(f"Status: {'SUCCESS' if total_cap >= requested_kw else 'WAITING/LIMITED'}")            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")

def main(input_file, **kwargs):
    config = RectifierRankingConfig()
    processor = DataProcessor(config)
    
    # 1. Prepare Data
    try:
        data = processor.prepare(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run code1.py first.")
        return

    # 2. Train Models
    detector = AnomalyDetector(config.FEATURES).train(data)
    data = detector.score(data) # Add anomaly_score for XGBoost training
    
    trainer = RankingModelTrainer(config)
    model = trainer.train(data)    
    recommender = RectifierRecommender(model, detector, config)
    
    # 3. Interactive Terminal (Skip if requested)
    if not kwargs.get('skip_simulation', False):
        run_interactive_terminal(recommender, data)
    else:
        print("Model training complete. Simulation skipped as per pipeline request.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='engineered_features.csv', help='Input feature file')
    args = parser.parse_args()
    main(args.input)