import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from pathlib import Path


def plot_and_measure_recall_at_k(y_test, probs, k_values=[500, 1000, 2000]):
    # Plots Precision-Recall curve and calculates Recall@K metrics
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, label='XGBoost Anomaly Model')
    plt.xlabel('Recall (How many errors we caught)')
    plt.ylabel('Precision (How often we were right)')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate Recall@K - shows how many errors caught when checking top K flagged items
    results = pd.DataFrame({'true_label': y_test, 'prob': probs}).sort_values(by='prob', ascending=False)
    
    total_anomalies = sum(y_test)
    print(f"\nTotal anomalies in test set: {total_anomalies}")
    print("\nRecall@K Analysis:")
    
    for k in k_values:
        top_k = results.head(k)
        found = sum(top_k['true_label'])
        recall_at_k = (found / total_anomalies) * 100 if total_anomalies > 0 else 0
        print(f"Recall@{k}: Found {found} errors ({recall_at_k:.1f}%)")
    
    return results


def check_nutrient_segments(X_test, y_test, preds):
    # Checks precision for different sodium value ranges
    print("\nPrecision by Sodium Value Ranges:")
    results = X_test.copy()
    results['true_label'] = y_test
    results['prediction'] = preds
    
    # Check sodium column specifically
    if 'sodium' in results.columns:
        # Check high-value ranges (potential anomalies)
        high_threshold = results['sodium'].quantile(0.9)
        high_values = results[results['sodium'] > high_threshold]
        if len(high_values) > 0:
            precision = high_values['true_label'].mean()
            print(f"Precision for high sodium (> {high_threshold:.1f}mg): {precision:.1%} ({len(high_values)} items)")
        
        # Check medium-value ranges
        med_threshold = results['sodium'].quantile(0.5)
        med_values = results[(results['sodium'] > med_threshold) & (results['sodium'] <= high_threshold)]
        if len(med_values) > 0:
            precision = med_values['true_label'].mean()
            print(f"Precision for medium sodium ({med_threshold:.1f}-{high_threshold:.1f}mg): {precision:.1%} ({len(med_values)} items)")
        
        # Check low-value ranges
        low_values = results[results['sodium'] <= med_threshold]
        if len(low_values) > 0:
            precision = low_values['true_label'].mean()
            print(f"Precision for low sodium (≤ {med_threshold:.1f}mg): {precision:.1%} ({len(low_values)} items)")


def export_false_positives(X_test, y_test, probs, df_original):
    # Exports high-confidence false positives for manual review
    print("\nFalse Positive Analysis:")
    analysis_df = X_test.copy()
    analysis_df['true_label'] = y_test
    analysis_df['prob'] = probs
    
    # Find high-confidence false positives (model confident but not actually an anomaly)
    false_positives = analysis_df[(analysis_df['true_label'] == 0) & (analysis_df['prob'] > 0.7)]
    
    print(f"Found {len(false_positives)} high-confidence false positives (prob > 0.7)")
    
    if len(false_positives) > 0:
        fp_sample = false_positives.nlargest(20, 'prob')
        output_file = Path(__file__).parent / "audit_false_positives.csv"
        fp_sample.to_csv(output_file, index=False)
        print(f"Exported top 20 False Positives to {output_file}")
        print("Look for patterns: Are these actually typos the QA team missed?")


def evaluate_model():
    # Main evaluation function - Sodium-specific anomaly detection
    print("MODEL EVALUATION - SODIUM ANOMALY DETECTION")
    
    # Load data from DataSodium folder
    data_path = Path(__file__).parent.parent / "DataSodium" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load trained model
    model_path = Path(__file__).parent / "final_anomaly_detector_sodium.json"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate predictions
    print("Generating predictions...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Run all evaluation metrics
    print("\n1. PRECISION-RECALL CURVE & RECALL@K")
    results_df = plot_and_measure_recall_at_k(y_test, probs, k_values=[500, 1000, 2000, 5000])
    
    print("\n2. PRECISION BY SODIUM VALUE RANGES")
    check_nutrient_segments(X_test, y_test, preds)
    
    print("\n3. FALSE POSITIVE AUDIT")
    merged_path = Path(__file__).parent.parent / "DataSodium" / "PreOpDataCSV" / "merged.csv"
    df_original = None
    if merged_path.exists():
        df_original = pd.read_csv(merged_path)
    
    export_false_positives(X_test, y_test, probs, df_original)
    
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    evaluate_model()
