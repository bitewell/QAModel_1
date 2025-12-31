import pandas as pd
import xgboost as xgb
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Column definitions matching DataOpSodium/embed_text.py
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

# Only sodium nutrient column for sodium-specific anomaly detection
NUTRIENT_COLS = ["sodium"]

# Columns to keep: only text + sodium + gtin (no tags/flags)
COLS_TO_KEEP = TEXT_COLS + NUTRIENT_COLS + ["gtin"]


def process_and_embed(df):
    # Cleans and embeds raw CSV data to match sodium model training feature set
    # Only uses: text columns (embedded) + sodium column
    
    # Keep only columns used in training
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    print(f"Selected {len(keep)} columns: text + sodium + gtin")
    
    # Convert ID and text columns to string
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Convert sodium to numeric (remove carets, fill missing with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Remove rows without GTIN or empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Embed text columns using SentenceTransformer
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for col in TEXT_COLS:
        if col in df.columns:
            print(f"Embedding {col}...")
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Drop original text columns (keep only embeddings)
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    # Verify feature set matches training
    final_cols = df.columns.tolist()
    embedding_count = len([c for c in final_cols if '_emb_' in c])
    sodium_count = len([c for c in final_cols if c in NUTRIENT_COLS])
    print(f"Final features: {embedding_count} embeddings + {sodium_count} sodium = {len(final_cols)} total")
    
    # Separate GTIN for reference, return feature matrix
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def test_model_on_dataset(csv_path, expected_result, dataset_name, threshold=0.3):
    # Tests sodium model on a dataset using custom probability threshold
    
    print(f"\nTesting on: {dataset_name}")
    print(f"Expected: {expected_result} | Using Threshold: {threshold}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return 0, None, None
    
    # Load and process data
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")
    
    X, gtin_column = process_and_embed(df)
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Load trained sodium model (check root directory first, then ModelTrainingSodium folder)
    model_path = Path(__file__).parent.parent / "final_anomaly_detector_sodium.json"
    if not model_path.exists():
        # Try ModelTrainingSodium folder
        model_path = Path(__file__).parent.parent / "ModelTrainingSodium" / "final_anomaly_detector_sodium.json"
    
    if not model_path.exists():
        print(f"Error: Sodium model not found")
        print("Please train the sodium model first using ModelTrainingSodium/train_model.py")
        return 0, None, None
    
    print(f"Loading sodium model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate probabilities
    print("Generating predictions...")
    probs = model.predict_proba(X)[:, 1]
    
    # Apply custom threshold (instead of default 0.5)
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    total_rows = len(preds)
    anomalies_detected = sum(preds)
    anomaly_percentage = (anomalies_detected / total_rows * 100) if total_rows > 0 else 0
    avg_probability = probs.mean()
    
    print(f"\nResults (at {threshold} threshold):")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Anomalies detected: {anomalies_detected:,}")
    print(f"  Catch Rate / FP Rate: {anomaly_percentage:.2f}%")
    print(f"  Average probability: {avg_probability:.3f}")
    
    # Confidence breakdown
    high = sum(probs >= 0.7)
    med = sum((probs >= 0.3) & (probs < 0.7))
    low = sum(probs < 0.3)
    
    print(f"\nConfidence Breakdown:")
    print(f"  High (>=0.7): {high} | Medium (0.3-0.7): {med} | Low (<0.3): {low}")
    
    return anomaly_percentage, probs, preds


def main():
    # Main function to test sodium model on both validation datasets
    data_folder = Path(__file__).parent / "Data"
    
    # Target threshold balances high recall with low false positives
    TARGET_THRESHOLD = 0.35 
    
    print("=" * 60)
    print("SODIUM MODEL TESTING ON AI TRAINING DATA")
    print("=" * 60)
    
    # Test on errors dataset (measures recall/catch rate)
    errors_file = data_folder / "AI Training Data - items data errors.csv"
    catch_rate, _, _ = test_model_on_dataset(errors_file, "100% anomalies", "Items Data Errors", threshold=TARGET_THRESHOLD)
    
    # Test on approved dataset (measures false positive rate)
    approved_file = data_folder / "AI Training Data - approved profiles.csv"
    fp_rate, _, _ = test_model_on_dataset(approved_file, "0% anomalies", "Approved Profiles", threshold=TARGET_THRESHOLD)
    
    print(f"\n" + "=" * 60)
    print(f"FINAL PERFORMANCE SUMMARY - SODIUM MODEL")
    print("=" * 60)
    print(f"Overall Catch Rate (Recall): {catch_rate:.2f}%")
    print(f"Overall False Positive Rate: {fp_rate:.2f}%")
    
    if fp_rate <= 10.0:
        print("SUCCESS: Threshold met the <10% False Positive requirement!")
    else:
        print(f"ALERT: False Positive rate exceeds 10%. Consider raising threshold above {TARGET_THRESHOLD}.")


if __name__ == "__main__":
    main()

