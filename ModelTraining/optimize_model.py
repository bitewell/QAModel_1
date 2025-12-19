import pandas as pd
import xgboost as xgb
import optuna
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from pathlib import Path


def objective(trial):
    # 1. Load Data
    data_path = Path(__file__).parent.parent / "Data" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    # Split: Train (80%) and Validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define the Hyperparameter Search Space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42,
        'eval_metric': 'aucpr',
        # Use the weight logic to prioritize the 1s (anomalies)
        'scale_pos_weight': (len(y) - sum(y)) / sum(y) 
    }

    # 3. Initialize and Train 
    # Removed pruning_callback to fix the TypeError
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 4. Evaluate using Average Precision (best for finding rare anomalies)
    preds_proba = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, preds_proba)
    
    return score


if __name__ == "__main__":
    print("=" * 70)
    print("HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 70)
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        print("\n" + "-" * 70)
        print("OPTIMIZATION COMPLETE")
        print("-" * 70)
        print(f"Best Avg Precision Score: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save results for train_model.py
        with open("best_hyperparams.json", "w") as f:
            json.dump(study.best_params, f)
        print("\nSaved best parameters to best_hyperparams.json")
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
