import pandas as pd
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path


def train_final_model():

    # Trains final model with optimized hyperparameters and analyzes threshold sensitivity
    # Uses only: text embeddings + sodium column (sodium-specific anomaly detection)
    data_path = Path(__file__).parent.parent / "DataSodium" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    # Verify feature set
    print(f"Training on {len(X.columns)} features:")
    embedding_cols = [c for c in X.columns if '_emb_' in c]
    sodium_cols = [c for c in X.columns if c == 'sodium']
    print(f"  - Text embeddings: {len(embedding_cols)} features")
    print(f"  - Sodium column: {len(sodium_cols)} feature")
    print(f"  - Total features: {len(X.columns)}")
    print(f"  - Training samples: {len(X):,}")
    print(f"  - Anomalies: {y.sum():,} ({y.mean()*100:.2f}%)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    

    # Load optimized hyperparameters from Optuna
    with open("best_hyperparams_sodium.json", "r") as f:
        best_params = json.load(f)
    

    # Calculate class weight for imbalanced data
    weight = (len(y) - sum(y)) / sum(y)
    model = xgb.XGBClassifier(**best_params, scale_pos_weight=weight, objective='binary:logistic', eval_metric='aucpr', random_state=42)

    print("Training final model with optimized parameters...")
    model.fit(X_train, y_train)


    # Get probability predictions (not just binary)
    probs = model.predict_proba(X_test)[:, 1]


    # Analyze different thresholds to find optimal balance
    print("\nSensitivity Analysis (How many errors can we catch?):")
    thresholds = [0.7, 0.5, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]
    for t in thresholds:
        t_preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, t_preds).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_test, t_preds)
        print(f"Threshold {t:.2f}: Recall = {recall:.1%}, Precision = {precision:.1%}, F1 = {f1:.3f}, Total Flagged = {tp+fp}")


    # Save the final model
    model.get_booster().save_model("final_anomaly_detector_sodium.json")
    print("\nFinal model saved as final_anomaly_detector_sodium.json")


if __name__ == "__main__":
    train_final_model()
