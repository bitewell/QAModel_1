import pandas as pd
import xgboost as xgb
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix
from pathlib import Path


def plot_and_measure_recall_at_k(y_test, probs, k_values=[500, 1000, 2000]):
    # Plot Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, label='XGBoost Anomaly Model')
    plt.xlabel('Recall (How many errors we caught)')
    plt.ylabel('Precision (How often we were right)')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Calculate Recall@K - shows how many errors we catch when checking top K flagged items
    results = pd.DataFrame({'true_label': y_test, 'prob': probs}).sort_values(by='prob', ascending=False)
    
    total_anomalies = sum(y_test)
    print(f"\nTotal anomalies in test set: {total_anomalies}")
    print("\n--- Recall@K Analysis ---")
    
    for k in k_values:
        top_k = results.head(k)
        found = sum(top_k['true_label'])
        recall_at_k = (found / total_anomalies) * 100 if total_anomalies > 0 else 0
        print(f"Recall@{k}: Found {found} errors ({recall_at_k:.1f}%)")
    
    return results


def check_nutrient_segments(X_test, y_test, preds):
    print("\n--- Precision by Nutrient Segments ---")
    results = X_test.copy()
    results['true_label'] = y_test
    results['prediction'] = preds
    
    # Check precision for different nutrient flag segments
    if 'flag_calorie_mismatch' in results.columns:
        calorie_errors = results[results['flag_calorie_mismatch'] == True]
        if len(calorie_errors) > 0:
            precision = calorie_errors['true_label'].mean()
            print(f"Precision for Calorie Mismatches: {precision:.1%} ({len(calorie_errors)} flagged)")
    
    if 'flag_fat_mismatch' in results.columns:
        fat_errors = results[results['flag_fat_mismatch'] == True]
        if len(fat_errors) > 0:
            precision = fat_errors['true_label'].mean()
            print(f"Precision for Fat Mismatches: {precision:.1%} ({len(fat_errors)} flagged)")
    
    if 'flag_carb_mismatch' in results.columns:
        carb_errors = results[results['flag_carb_mismatch'] == True]
        if len(carb_errors) > 0:
            precision = carb_errors['true_label'].mean()
            print(f"Precision for Carb Mismatches: {precision:.1%} ({len(carb_errors)} flagged)")
    
    if 'flag_sugar_mismatch' in results.columns:
        sugar_errors = results[results['flag_sugar_mismatch'] == True]
        if len(sugar_errors) > 0:
            precision = sugar_errors['true_label'].mean()
            print(f"Precision for Sugar Mismatches: {precision:.1%} ({len(sugar_errors)} flagged)")
    
    if 'flag_negative_values' in results.columns:
        neg_errors = results[results['flag_negative_values'] == True]
        if len(neg_errors) > 0:
            precision = neg_errors['true_label'].mean()
            print(f"Precision for Negative Values: {precision:.1%} ({len(neg_errors)} flagged)")


def export_false_positives(X_test, y_test, probs, df_original):
    print("\n--- False Positive Analysis ---")
    analysis_df = X_test.copy()
    analysis_df['true_label'] = y_test
    analysis_df['prob'] = probs
    
    # Find high-confidence false positives - model was confident but human didn't mark as anomaly
    false_positives = analysis_df[(analysis_df['true_label'] == 0) & (analysis_df['prob'] > 0.7)]
    
    print(f"Found {len(false_positives)} high-confidence false positives (prob > 0.7)")
    
    if len(false_positives) > 0:
        fp_sample = false_positives.nlargest(20, 'prob')
        output_file = Path(__file__).parent / "audit_false_positives.csv"
        fp_sample.to_csv(output_file, index=False)
        print(f"Exported top 20 False Positives to {output_file}")
        print("Look for patterns: Are these actually typos the QA team missed?")


def evaluate_model():
    print("=" * 70)
    print("MODEL EVALUATION - PRODUCTION METRICS")
    print("=" * 70)
    
    # Load embedded parquet data
    data_path = Path(__file__).parent.parent / "Data" / "PostOpData" / "merged_embedded.parquet"
    df = pd.read_parquet(data_path)
    
    X = df.drop(columns=["gtin", "label_is_anomaly"], errors="ignore")
    y = df["label_is_anomaly"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load trained model
    model_path = Path(__file__).parent / "final_anomaly_detector.json"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate predictions and probabilities
    print("Generating predictions...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # Precision-Recall curve and Recall@K analysis
    print("\n" + "=" * 70)
    print("1. PRECISION-RECALL CURVE & RECALL@K")
    print("=" * 70)
    results_df = plot_and_measure_recall_at_k(y_test, probs, k_values=[500, 1000, 2000, 5000])
    
    # Precision by nutrient segments
    print("\n" + "=" * 70)
    print("2. PRECISION BY NUTRIENT SEGMENTS")
    print("=" * 70)
    check_nutrient_segments(X_test, y_test, preds)
    
    # False positive audit
    print("\n" + "=" * 70)
    print("3. FALSE POSITIVE AUDIT")
    print("=" * 70)
    
    merged_path = Path(__file__).parent.parent / "Data" / "PreOpDataCSV" / "merged.csv"
    df_original = None
    if merged_path.exists():
        df_original = pd.read_csv(merged_path)
    
    export_false_positives(X_test, y_test, probs, df_original)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_model()
