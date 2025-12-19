import pandas as pd
import xgboost as xgb
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import DataOps modules
sys.path.append(str(Path(__file__).parent.parent))
from DataOps.cleandata import (
    ID_COL, TEXT_COLS, NUTRIENT_COLS, TAG_COLS, FLAG_COLS, COLS_TO_KEEP
)

# Text columns to embed
TEXT_COLS_EMBED = ["category_name", "product_name", "ingredients_text"]


def process_and_embed(df):
    # Cleans and embeds the raw CSV data to match the training feature set
    
    # Keep only columns used in training
    keep = [c for c in COLS_TO_KEEP if c in df.columns and c != "label_is_anomaly"]
    df = df[keep].copy()
    
    # Convert ID + text to string
    for col in ID_COL + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Nutrient columns → clean float (remove carets, fill missing with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Tags + flags → bool
    bool_cols = TAG_COLS + FLAG_COLS
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
                .fillna(False)
                .astype(bool)
            )
    
    # Drop rows without GTIN and empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Embed text columns using SentenceTransformer
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for col in TEXT_COLS_EMBED:
        if col in df.columns:
            print(f"Embedding {col}...")
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Drop original text columns to leave only embeddings + numeric features
    df = df.drop(columns=TEXT_COLS_EMBED, errors="ignore")
    
    # Separate GTIN for reference, return Feature Matrix X
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def test_model_on_dataset(csv_path, expected_result, dataset_name, threshold=0.3):
    # Runs the model on a dataset using a CUSTOM probability threshold
    
    print(f"\n{'=' * 70}")
    print(f"Testing on: {dataset_name}")
    print(f"Expected: {expected_result} | Using Threshold: {threshold}")
    print(f"{'=' * 70}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return 0, None, None
    
    # Load and process
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")
    
    X, gtin_column = process_and_embed(df)
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Load trained model
    model_path = Path(__file__).parent.parent / "ModelTraining" / "final_anomaly_detector.json"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "final_anomaly_detector.json"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 0, None, None
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate probabilities
    print("Generating predictions...")
    probs = model.predict_proba(X)[:, 1]
    
    # Apply the custom threshold instead of using model.predict()
    # This increases the 'catch rate' (Recall)
    preds = (probs >= threshold).astype(int)
    
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
    data_folder = Path(__file__).parent / "Data"
    
    # Target Threshold: 0.3 balances high recall with low false positives
    TARGET_THRESHOLD = 0.35 
    
    # Test on errors (Measures RECALL/Catch Rate)
    errors_file = data_folder / "AI Training Data - items data errors.csv"
    catch_rate, _, _ = test_model_on_dataset(errors_file, "100% anomalies", "Items Data Errors", threshold=TARGET_THRESHOLD)
    
    # Test on approved (Measures FALSE POSITIVE Rate)
    approved_file = data_folder / "AI Training Data - approved profiles.csv"
    fp_rate, _, _ = test_model_on_dataset(approved_file, "0% anomalies", "Approved Profiles", threshold=TARGET_THRESHOLD)
    
    print(f"\n{'=' * 70}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Overall Catch Rate (Recall): {catch_rate:.2f}%")
    print(f"Overall False Positive Rate: {fp_rate:.2f}%")
    
    if fp_rate <= 10.0:
        print("SUCCESS: Your threshold of 0.3 met the <10% False Positive requirement!")
    else:
        print("ALERT: False Positive rate exceeds 10%. Consider raising threshold to 0.35 or 0.4.")


if __name__ == "__main__":
    main()
