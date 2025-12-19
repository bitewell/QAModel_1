import pandas as pd
import xgboost as xgb
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import DataOps modules
sys.path.append(str(Path(__file__).parent.parent))
from DataOps.cleandata import (
    ID_COL, TEXT_COLS, NUTRIENT_COLS, TAG_COLS, FLAG_COLS, COLS_TO_KEEP, clean
)

# Text columns to embed
TEXT_COLS_EMBED = ["category_name", "product_name", "ingredients_text"]


def process_and_embed(df):
    # Clean the data (but skip label_is_anomaly creation since we don't have labels)
    # We need to manually process since clean() expects label_is_anomaly
    
    # Keep only columns we need
    keep = [c for c in COLS_TO_KEEP if c in df.columns and c != "label_is_anomaly"]
    df = df[keep].copy()
    
    # Convert ID + text to string
    for col in ID_COL + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Nutrient columns → clean float (remove carets, keep pure numbers)
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
    
    # Drop rows without GTIN
    df = df.dropna(subset=["gtin"])
    
    # Remove rows with empty ingredients_text
    if "ingredients_text" in df.columns:
        df = df[
            df["ingredients_text"].notna() & 
            (df["ingredients_text"].astype(str).str.strip() != "")
        ]
    
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
    
    # Drop original text columns
    df = df.drop(columns=TEXT_COLS_EMBED, errors="ignore")
    
    # Ensure GTIN is kept for reference but drop it before prediction
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def test_model_on_dataset(csv_path, expected_result, dataset_name):
    print(f"\n{'=' * 70}")
    print(f"Testing on: {dataset_name}")
    print(f"Expected: {expected_result}")
    print(f"{'=' * 70}")
    
    # Load CSV file
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows")
    
    # Process and embed
    X, gtin_column = process_and_embed(df)
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Load trained model
    model_path = Path(__file__).parent.parent / "ModelTraining" / "final_anomaly_detector.json"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "final_anomaly_detector.json"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate predictions and probabilities
    print("Generating predictions...")
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Calculate results
    total_rows = len(preds)
    anomalies_detected = sum(preds)
    anomaly_percentage = (anomalies_detected / total_rows * 100) if total_rows > 0 else 0
    avg_probability = probs.mean()
    
    print(f"\nResults:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Anomalies detected: {anomalies_detected:,}")
    print(f"  Anomaly percentage: {anomaly_percentage:.2f}%")
    print(f"  Average probability: {avg_probability:.3f}")
    print(f"  Expected: {expected_result}")
    
    # Show distribution of probabilities
    high_confidence = sum(probs >= 0.7)
    medium_confidence = sum((probs >= 0.3) & (probs < 0.7))
    low_confidence = sum(probs < 0.3)
    
    print(f"\nProbability distribution:")
    print(f"  High confidence (≥0.7): {high_confidence:,} ({high_confidence/total_rows*100:.1f}%)")
    print(f"  Medium confidence (0.3-0.7): {medium_confidence:,} ({medium_confidence/total_rows*100:.1f}%)")
    print(f"  Low confidence (<0.3): {low_confidence:,} ({low_confidence/total_rows*100:.1f}%)")
    
    return anomaly_percentage, probs, preds


def main():
    data_folder = Path(__file__).parent / "Data"
    
    # Test on errors dataset (should be 100% anomalies)
    errors_file = data_folder / "AI Training Data - items data errors.csv"
    test_model_on_dataset(errors_file, "100% anomalies", "Items Data Errors")
    
    # Test on approved dataset (should be 0% anomalies)
    approved_file = data_folder / "AI Training Data - approved profiles.csv"
    test_model_on_dataset(approved_file, "0% anomalies", "Approved Profiles")
    
    print(f"\n{'=' * 70}")
    print("TESTING COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

