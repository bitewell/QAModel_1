import pandas as pd
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from pathlib import Path


def train_final_model():
    data_path = Path(__file__).parent.parent / "Data" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Load the Optuna Best Params
    with open("best_hyperparams.json", "r") as f:
        best_params = json.load(f)
    
    # 2. Add fixed params
    weight = (len(y) - sum(y)) / sum(y)
    model = xgb.XGBClassifier(**best_params, scale_pos_weight=weight, objective='binary:logistic', eval_metric='aucpr', random_state=42)

    print("Training final model with optimized parameters...")
    model.fit(X_train, y_train)

    # 3. GET PROBABILITIES (Instead of just 0 or 1)
    # This is the key to catching more anomalies
    probs = model.predict_proba(X_test)[:, 1]

    # 4. Threshold Analysis
    print("\n--- Sensitivity Analysis (How many errors can we catch?) ---")
    thresholds = [0.7, 0.5, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]
    for t in thresholds:
        t_preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, t_preds)
        print(f"Threshold {t:.2f}: Catch Rate (Recall) = {recall:.1%}, Precision = {precision:.1%}, F1 = {f1:.3f}, Total Flagged = {tp+fp}")

    # Save the final model
    model.get_booster().save_model("final_anomaly_detector.json")
    print("\nFinal model saved as final_anomaly_detector.json")


if __name__ == "__main__":
    train_final_model()
