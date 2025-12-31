import pandas as pd
import xgboost as xgb
import sys
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Column definitions matching embed_text.py
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

# Columns to keep: only text + nutrients + gtin (no tags/flags)
COLS_TO_KEEP = TEXT_COLS + NUTRIENT_COLS + ["gtin"]


def process_and_embed(df):
    # Cleans and embeds raw CSV data to match training feature set
    # Only uses: text columns (embedded) + nutrient columns (no tag/flag columns)
    
    # Keep only columns used in training
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    print(f"Selected {len(keep)} columns: text + nutrients + gtin")
    
    # Convert ID and text columns to string
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Convert nutrient columns to numeric (remove carets, fill missing with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Load and apply sodium scaler (if it exists)
    if "sodium" in df.columns:
        scaler_path = Path(__file__).parent.parent / "DataOps" / "sodium_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            # Transform sodium using saved scaler
            df["sodium"] = scaler.transform(df[["sodium"]])
            print("Applied saved sodium scaler")
        else:
            print("Warning: Sodium scaler not found. Sodium will not be scaled.")
    
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
    nutrient_count = len([c for c in final_cols if c in NUTRIENT_COLS])
    print(f"Final features: {embedding_count} embeddings + {nutrient_count} nutrients = {len(final_cols)} total")
    
    # Separate GTIN for reference, return feature matrix
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def predict_test_sample(csv_path, threshold=0.35):
    # Predicts anomalies on a test CSV file and outputs results
    
    print(f"Loading test file: {csv_path}")
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    # Load and process data - keep original dataframe for output
    df_original = pd.read_csv(csv_path)
    print(f"Loaded {len(df_original):,} rows")
    
    # Process and embed (this creates a filtered version)
    X, gtin_column = process_and_embed(df_original.copy())
    print(f"After processing: {len(X):,} rows, {len(X.columns)} features")
    
    # Match predictions back to original dataframe by GTIN
    # Some rows may have been filtered out during processing
    df_original['gtin'] = df_original['gtin'].astype(str)
    gtin_column = gtin_column.astype(str)
    
    # Load trained model
    model_path = Path(__file__).parent.parent / "ModelTraining" / "final_anomaly_detector.json"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent / "final_anomaly_detector.json"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using ModelTraining/train_model.py")
        return None
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Generate probabilities
    print("Generating predictions...")
    probs = model.predict_proba(X)[:, 1]
    
    # Apply threshold to get binary predictions
    preds = (probs >= threshold).astype(bool)
    
    # Create predictions dataframe with GTIN
    predictions_df = pd.DataFrame({
        'gtin': gtin_column,
        'label_is_anomaly': preds,
        'probability': probs
    })
    
    # Merge predictions back to original dataframe
    # Keep all original columns and add our prediction columns
    results_df = df_original.merge(
        predictions_df,
        on='gtin',
        how='left'
    )
    
    # Fill NaN for rows that were filtered out during processing
    results_df['label_is_anomaly'] = results_df['label_is_anomaly'].fillna(False)
    results_df['probability'] = results_df['probability'].fillna(0.0)
    
    # Summary statistics (only on rows that were successfully processed)
    processed_mask = results_df['probability'] > 0
    processed_probs = results_df.loc[processed_mask, 'probability']
    processed_preds = results_df.loc[processed_mask, 'label_is_anomaly']
    
    print(f"\nPrediction Summary:")
    print(f"  Total rows in output: {len(results_df):,}")
    print(f"  Rows successfully processed: {processed_mask.sum():,}")
    print(f"  Rows filtered out (missing GTIN/ingredients): {(~processed_mask).sum():,}")
    print(f"  Anomalies detected (label_is_anomaly=True): {processed_preds.sum():,} ({processed_preds.mean()*100:.2f}%)")
    print(f"  Normal (label_is_anomaly=False): {(~processed_preds).sum():,}")
    if len(processed_probs) > 0:
        print(f"  Average probability: {processed_probs.mean():.3f}")
        print(f"  Min probability: {processed_probs.min():.3f}")
        print(f"  Max probability: {processed_probs.max():.3f}")
        
        # Confidence breakdown
        high = (processed_probs >= 0.7).sum()
        med = ((processed_probs >= 0.3) & (processed_probs < 0.7)).sum()
        low = (processed_probs < 0.3).sum()
        print(f"\nConfidence Breakdown:")
        print(f"  High (>=0.7): {high} | Medium (0.3-0.7): {med} | Low (<0.3): {low}")
    
    return results_df


if __name__ == "__main__":
    # Default threshold for predictions
    THRESHOLD = 0.2
    
    # Get input file from command line argument or use default
    if len(sys.argv) > 1:
        # Use provided file path
        input_file = Path(sys.argv[1])
        if not input_file.is_absolute():
            # If relative path, assume it's relative to Data folder
            input_file = Path(__file__).parent / "Data" / sys.argv[1]
    else:
        # Default to test sample file
        input_file = Path(__file__).parent / "Data" / "test sample not fully qa'd - all_scores.csv"
    
    # Get threshold from command line if provided
    if len(sys.argv) > 2:
        try:
            THRESHOLD = float(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid threshold '{sys.argv[2]}', using default {THRESHOLD}")
    
    # Generate output filename from input filename
    input_stem = input_file.stem
    output_file = Path(__file__).parent / "Data" / f"{input_stem}_predictions.csv"
    
    print("=" * 60)
    print("ANOMALY DETECTION ON TEST SAMPLE")
    print("=" * 60)
    print(f"Input file: {input_file.name}")
    print(f"Threshold: {THRESHOLD}")
    print()
    
    # Run predictions
    results = predict_test_sample(input_file, threshold=THRESHOLD)
    
    if results is not None:
        # Save results - includes all original columns plus label_is_anomaly and probability
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Output includes: All original columns + 'label_is_anomaly' (True/False) + 'probability' (0-1)")
        print(f"\nUsage: python3 predict_test_sample.py [filename.csv] [threshold]")
        print(f"Example: python3 predict_test_sample.py 'TEST 2 for Anomoly Test (from NIQ pt3 sheet1 - Sheet1.csv' 0.2")
