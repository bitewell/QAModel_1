import pandas as pd
import xgboost as xgb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Column definitions for general model
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

NUTRIENT_COLS_GENERAL = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

NUTRIENT_COLS_SODIUM = ["sodium"]

COLS_TO_KEEP_GENERAL = TEXT_COLS + NUTRIENT_COLS_GENERAL + ["gtin"]
COLS_TO_KEEP_SODIUM = TEXT_COLS + NUTRIENT_COLS_SODIUM + ["gtin"]


def process_and_embed_general(df):
    # Process data for general model: text embeddings + all nutrients
    keep = [c for c in COLS_TO_KEEP_GENERAL if c in df.columns]
    df = df[keep].copy()
    
    # Convert ID and text columns to string
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Convert nutrient columns to numeric
    for col in NUTRIENT_COLS_GENERAL:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Remove rows without GTIN or empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Embed text columns
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for col in TEXT_COLS:
        if col in df.columns:
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Drop original text columns
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def process_and_embed_sodium(df):
    # Process data for sodium model: text embeddings + sodium only
    keep = [c for c in COLS_TO_KEEP_SODIUM if c in df.columns]
    df = df[keep].copy()
    
    # Convert ID and text columns to string
    for col in ["gtin"] + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Convert sodium to numeric
    for col in NUTRIENT_COLS_SODIUM:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)
    
    # Remove rows without GTIN or empty ingredients
    df = df.dropna(subset=["gtin"])
    if "ingredients_text" in df.columns:
        df = df[df["ingredients_text"].notna() & (df["ingredients_text"].astype(str).str.strip() != "")]
    
    # Embed text columns
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for col in TEXT_COLS:
        if col in df.columns:
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
    
    # Drop original text columns
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    gtin_column = df["gtin"].copy() if "gtin" in df.columns else None
    X = df.drop(columns=["gtin"], errors="ignore")
    
    return X, gtin_column


def load_models():
    # Load both trained models
    root = Path(__file__).parent.parent
    
    # Load general model
    general_model_path = root / "final_anomaly_detector.json"
    if not general_model_path.exists():
        general_model_path = root / "ModelTraining" / "final_anomaly_detector.json"
    
    if not general_model_path.exists():
        raise FileNotFoundError(f"General model not found. Checked: {general_model_path}")
    
    print(f"Loading general model from {general_model_path}...")
    general_model = xgb.XGBClassifier()
    general_model.load_model(str(general_model_path))
    
    # Load sodium model
    sodium_model_path = root / "final_anomaly_detector_sodium.json"
    if not sodium_model_path.exists():
        sodium_model_path = root / "ModelTrainingSodium" / "final_anomaly_detector_sodium.json"
    
    if not sodium_model_path.exists():
        raise FileNotFoundError(f"Sodium model not found. Checked: {sodium_model_path}")
    
    print(f"Loading sodium model from {sodium_model_path}...")
    sodium_model = xgb.XGBClassifier()
    sodium_model.load_model(str(sodium_model_path))
    
    return general_model, sodium_model


def fuse_predictions(probs_general, probs_sodium, method='union', 
                     threshold_general=0.2, threshold_sodium=0.35, 
                     weight_general=0.6, weight_sodium=0.4):
    # Fuse predictions from both models using different strategies
    # Methods: 'union' (either flags), 'intersection' (both flag), 'weighted_avg', 'max', 'min'
    if method == 'union':
        # Flag if either model flags it
        preds_general = (probs_general >= threshold_general).astype(int)
        preds_sodium = (probs_sodium >= threshold_sodium).astype(int)
        fused_preds = np.maximum(preds_general, preds_sodium)
        fused_probs = np.maximum(probs_general, probs_sodium)
        
    elif method == 'intersection':
        # Flag only if both models flag it
        preds_general = (probs_general >= threshold_general).astype(int)
        preds_sodium = (probs_sodium >= threshold_sodium).astype(int)
        fused_preds = preds_general * preds_sodium
        fused_probs = np.minimum(probs_general, probs_sodium)
        
    elif method == 'weighted_avg':
        # Weighted average of probabilities
        fused_probs = weight_general * probs_general + weight_sodium * probs_sodium
        # Use average threshold
        avg_threshold = weight_general * threshold_general + weight_sodium * threshold_sodium
        fused_preds = (fused_probs >= avg_threshold).astype(int)
        
    elif method == 'max':
        # Take maximum probability
        fused_probs = np.maximum(probs_general, probs_sodium)
        fused_preds = (fused_probs >= min(threshold_general, threshold_sodium)).astype(int)
        
    elif method == 'min':
        # Take minimum probability (very conservative)
        fused_probs = np.minimum(probs_general, probs_sodium)
        fused_preds = (fused_probs >= max(threshold_general, threshold_sodium)).astype(int)
        
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused_preds, fused_probs


def test_ensemble_on_dataset(csv_path, expected_result, dataset_name, 
                             fusion_method='union',
                             threshold_general=0.2, threshold_sodium=0.35,
                             return_full_df=False):
    # Test ensemble of both models on a dataset
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TESTING: {dataset_name}")
    print(f"Expected: {expected_result} | Fusion Method: {fusion_method}")
    print(f"Thresholds - General: {threshold_general}, Sodium: {threshold_sodium}")
    print(f"{'='*70}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return None
    
    # Load data - keep original for merging later
    print(f"\nLoading {csv_path}...")
    df_original = pd.read_csv(csv_path)
    print(f"Loaded {len(df_original):,} rows")
    
    # Process data for both models
    print("\nProcessing data for general model...")
    X_general, gtin_general = process_and_embed_general(df_original.copy())
    print(f"General model features: {len(X_general.columns)}")
    
    print("\nProcessing data for sodium model...")
    X_sodium, gtin_sodium = process_and_embed_sodium(df_original.copy())
    print(f"Sodium model features: {len(X_sodium.columns)}")
    
    # Align GTINs (they should match, but handle edge cases)
    if gtin_general is not None and gtin_sodium is not None:
        common_gtins = set(gtin_general) & set(gtin_sodium)
        print(f"\nCommon GTINs between models: {len(common_gtins)}")
        
        # Filter to common GTINs
        mask_general = gtin_general.isin(common_gtins)
        mask_sodium = gtin_sodium.isin(common_gtins)
        X_general = X_general[mask_general]
        X_sodium = X_sodium[mask_sodium]
        gtin_general = gtin_general[mask_general]
        gtin_sodium = gtin_sodium[mask_sodium]
        
        # Sort by GTIN to ensure alignment
        general_df = pd.DataFrame({'gtin': gtin_general}).reset_index(drop=True)
        sodium_df = pd.DataFrame({'gtin': gtin_sodium}).reset_index(drop=True)
        general_df = general_df.sort_values('gtin').reset_index(drop=True)
        sodium_df = sodium_df.sort_values('gtin').reset_index(drop=True)
        
        X_general = X_general.loc[general_df.index].reset_index(drop=True)
        X_sodium = X_sodium.loc[sodium_df.index].reset_index(drop=True)
        gtin_general = general_df['gtin'].reset_index(drop=True)
        gtin_sodium = sodium_df['gtin'].reset_index(drop=True)
    
    print(f"\nAfter alignment: {len(X_general):,} rows")
    
    # Load models
    general_model, sodium_model = load_models()
    
    # Get predictions from both models
    print("\nGenerating predictions from general model...")
    probs_general = general_model.predict_proba(X_general)[:, 1]
    
    print("Generating predictions from sodium model...")
    probs_sodium = sodium_model.predict_proba(X_sodium)[:, 1]
    
    # Individual model results
    preds_general = (probs_general >= threshold_general).astype(int)
    preds_sodium = (probs_sodium >= threshold_sodium).astype(int)
    
    print(f"\n{'─'*70}")
    print("INDIVIDUAL MODEL RESULTS:")
    print(f"{'─'*70}")
    print(f"General Model:")
    print(f"  Anomalies detected: {preds_general.sum():,} ({preds_general.mean()*100:.2f}%)")
    print(f"  Avg probability: {probs_general.mean():.3f}")
    print(f"\nSodium Model:")
    print(f"  Anomalies detected: {preds_sodium.sum():,} ({preds_sodium.mean()*100:.2f}%)")
    print(f"  Avg probability: {probs_sodium.mean():.3f}")
    
    # Fuse predictions
    print(f"\n{'─'*70}")
    print(f"FUSED PREDICTIONS (Method: {fusion_method}):")
    print(f"{'─'*70}")
    
    fused_preds, fused_probs = fuse_predictions(
        probs_general, probs_sodium, 
        method=fusion_method,
        threshold_general=threshold_general,
        threshold_sodium=threshold_sodium
    )
    
    # Calculate metrics
    total_rows = len(fused_preds)
    anomalies_detected = fused_preds.sum()
    anomaly_percentage = (anomalies_detected / total_rows * 100) if total_rows > 0 else 0
    avg_probability = fused_probs.mean()
    
    print(f"\nResults:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Anomalies detected: {anomalies_detected:,}")
    print(f"  Catch Rate / FP Rate: {anomaly_percentage:.2f}%")
    print(f"  Average probability: {avg_probability:.3f}")
    
    # Confidence breakdown
    high = (fused_probs >= 0.7).sum()
    med = ((fused_probs >= 0.3) & (fused_probs < 0.7)).sum()
    low = (fused_probs < 0.3).sum()
    
    print(f"\nConfidence Breakdown:")
    print(f"  High (>=0.7): {high} | Medium (0.3-0.7): {med} | Low (<0.3): {low}")
    
    # Model agreement analysis
    agreement = (preds_general == preds_sodium).sum()
    agreement_pct = (agreement / len(preds_general) * 100) if len(preds_general) > 0 else 0
    both_flagged = ((preds_general == 1) & (preds_sodium == 1)).sum()
    only_general = ((preds_general == 1) & (preds_sodium == 0)).sum()
    only_sodium = ((preds_general == 0) & (preds_sodium == 1)).sum()
    
    print(f"\nModel Agreement Analysis:")
    print(f"  Agreement: {agreement:,} ({agreement_pct:.1f}%)")
    print(f"  Both flagged: {both_flagged:,}")
    print(f"  Only general: {only_general:,}")
    print(f"  Only sodium: {only_sodium:,}")
    
    result_dict = {
        'total_rows': total_rows,
        'anomalies_detected': anomalies_detected,
        'anomaly_percentage': anomaly_percentage,
        'avg_probability': avg_probability,
        'probs_general': probs_general,
        'probs_sodium': probs_sodium,
        'probs_fused': fused_probs,
        'preds_general': preds_general,
        'preds_sodium': preds_sodium,
        'preds_fused': fused_preds,
        'gtin': gtin_general if gtin_general is not None else gtin_sodium
    }
    
    # If return_full_df is True, merge predictions back with original dataframe
    if return_full_df:
        # Create predictions dataframe with GTIN as key
        gtin_key = gtin_general if gtin_general is not None else gtin_sodium
        # Convert GTIN to string for consistent merging
        gtin_key_str = gtin_key.astype(str)
        
        preds_df = pd.DataFrame({
            'gtin': gtin_key_str,
            'prob_general': probs_general,
            'prob_sodium': probs_sodium,
            'prob_fused': fused_probs,
            'is_anomaly_general': preds_general.astype(bool),
            'is_anomaly_sodium': preds_sodium.astype(bool),
        })
        
        # Union logic: is_anomaly = True if either model flags it
        preds_df['is_anomaly'] = (preds_df['is_anomaly_general'] | preds_df['is_anomaly_sodium'])
        
        # Merge with original dataframe
        # First, try to find GTIN column (case-insensitive)
        gtin_col = None
        for col in df_original.columns:
            if col.lower() == 'gtin':
                gtin_col = col
                break
        
        if gtin_col:
            # Convert original GTIN column to string for consistent merging
            df_original_merged = df_original.copy()
            df_original_merged[gtin_col] = df_original_merged[gtin_col].astype(str)
            
            # Merge on GTIN column
            df_output = df_original_merged.merge(preds_df, left_on=gtin_col, right_on='gtin', how='left')
            # Drop the duplicate 'gtin' column from preds_df if it's different from original
            if 'gtin_y' in df_output.columns:
                df_output = df_output.drop(columns=['gtin_y'])
                df_output = df_output.rename(columns={'gtin_x': gtin_col})
            elif 'gtin' in df_output.columns and gtin_col != 'gtin':
                df_output = df_output.drop(columns=['gtin'])
        else:
            # If no GTIN column found, try UPC or other ID columns
            id_cols = [col for col in df_original.columns if any(x in col.lower() for x in ['gtin', 'upc', 'id', 'product_id'])]
            if id_cols:
                # Try merging on first ID column found
                id_col = id_cols[0]
                df_original_merged = df_original.copy()
                df_original_merged[id_col] = df_original_merged[id_col].astype(str)
                preds_df_renamed = preds_df.rename(columns={'gtin': id_col})
                df_output = df_original_merged.merge(preds_df_renamed, on=id_col, how='left')
            else:
                # Last resort: add predictions by index (risky if rows were filtered)
                print("Warning: No GTIN/UPC/ID column found. Matching by index (may be inaccurate if rows were filtered).")
                df_output = df_original.copy()
                # Only add predictions for rows that exist in preds_df
                for idx in range(min(len(df_output), len(preds_df))):
                    df_output.loc[idx, 'prob_general'] = preds_df.loc[idx, 'prob_general']
                    df_output.loc[idx, 'prob_sodium'] = preds_df.loc[idx, 'prob_sodium']
                    df_output.loc[idx, 'prob_fused'] = preds_df.loc[idx, 'prob_fused']
                    df_output.loc[idx, 'is_anomaly_general'] = preds_df.loc[idx, 'is_anomaly_general']
                    df_output.loc[idx, 'is_anomaly_sodium'] = preds_df.loc[idx, 'is_anomaly_sodium']
                    df_output.loc[idx, 'is_anomaly'] = preds_df.loc[idx, 'is_anomaly']
                # Fill NaN for rows that don't have predictions
                df_output['prob_general'] = df_output['prob_general'].fillna(0.0)
                df_output['prob_sodium'] = df_output['prob_sodium'].fillna(0.0)
                df_output['prob_fused'] = df_output['prob_fused'].fillna(0.0)
                df_output['is_anomaly_general'] = df_output['is_anomaly_general'].fillna(False)
                df_output['is_anomaly_sodium'] = df_output['is_anomaly_sodium'].fillna(False)
                df_output['is_anomaly'] = df_output['is_anomaly'].fillna(False)
        
        # Fill NaN values for rows that didn't match (were filtered out during processing)
        df_output['prob_general'] = df_output['prob_general'].fillna(0.0)
        df_output['prob_sodium'] = df_output['prob_sodium'].fillna(0.0)
        df_output['prob_fused'] = df_output['prob_fused'].fillna(0.0)
        df_output['is_anomaly_general'] = df_output['is_anomaly_general'].fillna(False)
        df_output['is_anomaly_sodium'] = df_output['is_anomaly_sodium'].fillna(False)
        df_output['is_anomaly'] = df_output['is_anomaly'].fillna(False)
        
        # Add rule: if sodium > 1300, mark as anomaly
        # Find sodium column (case-insensitive)
        sodium_col = None
        for col in df_output.columns:
            if col.lower() == 'sodium':
                sodium_col = col
                break
        
        if sodium_col:
            # Convert sodium to numeric, handling any non-numeric values
            df_output[sodium_col] = pd.to_numeric(df_output[sodium_col], errors='coerce')
            # Apply rule: if sodium > 1300, mark as anomaly
            high_sodium_mask = df_output[sodium_col] > 1300
            df_output.loc[high_sodium_mask, 'is_anomaly'] = True
            
            # Count how many were flagged by this rule
            high_sodium_count = high_sodium_mask.sum()
            if high_sodium_count > 0:
                print(f"\nAdditional Rule Applied:")
                print(f"  Items with sodium > 1300: {high_sodium_count}")
                print(f"  These items are now marked as anomalies")
        else:
            print("\nWarning: Sodium column not found. Skipping sodium > 1300 rule.")
        
        result_dict['full_dataframe'] = df_output
    
    return result_dict


def main():
    # Main function to test ensemble on AI training data
    data_folder = Path(__file__).parent / "Data"
    
    # Test different fusion methods
    fusion_methods = ['union', 'intersection', 'weighted_avg', 'max']
    
    # Thresholds
    threshold_general = 0.2
    threshold_sodium = 0.35
    
    print("\n" + "="*70)
    print("ENSEMBLE MODEL TESTING ON AI TRAINING DATA")
    print("="*70)
    
    results_summary = {}
    
    for method in fusion_methods:
        print(f"\n\n{'#'*70}")
        print(f"TESTING FUSION METHOD: {method.upper()}")
        print(f"{'#'*70}")
        
        # Test on errors dataset
        errors_file = data_folder / "AI Training Data - items data errors.csv"
        errors_result = test_ensemble_on_dataset(
            errors_file, "100% anomalies", "Items Data Errors",
            fusion_method=method,
            threshold_general=threshold_general,
            threshold_sodium=threshold_sodium
        )
        
        # Test on approved dataset
        approved_file = data_folder / "AI Training Data - approved profiles.csv"
        approved_result = test_ensemble_on_dataset(
            approved_file, "0% anomalies", "Approved Profiles",
            fusion_method=method,
            threshold_general=threshold_general,
            threshold_sodium=threshold_sodium
        )
        
        if errors_result and approved_result:
            results_summary[method] = {
                'catch_rate': errors_result['anomaly_percentage'],
                'fp_rate': approved_result['anomaly_percentage'],
                'errors_avg_prob': errors_result['avg_probability'],
                'approved_avg_prob': approved_result['avg_probability']
            }
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL ENSEMBLE PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Method':<20} {'Catch Rate':<15} {'FP Rate':<15} {'Errors Avg Prob':<18} {'Approved Avg Prob':<18}")
    print(f"{'-'*70}")
    
    for method, metrics in results_summary.items():
        print(f"{method:<20} {metrics['catch_rate']:>6.2f}%       {metrics['fp_rate']:>6.2f}%       "
              f"{metrics['errors_avg_prob']:>6.3f}            {metrics['approved_avg_prob']:>6.3f}")
    
    # Recommendation
    print(f"\n{'─'*70}")
    print("RECOMMENDATIONS:")
    print(f"{'─'*70}")
    
    best_catch = max(results_summary.items(), key=lambda x: x[1]['catch_rate'])
    best_fp = min(results_summary.items(), key=lambda x: x[1]['fp_rate'])
    
    print(f"Best Catch Rate: {best_catch[0]} ({best_catch[1]['catch_rate']:.2f}%)")
    print(f"Lowest FP Rate: {best_fp[0]} ({best_fp[1]['fp_rate']:.2f}%)")
    
    # Find balanced option (high catch, reasonable FP)
    balanced = None
    for method, metrics in results_summary.items():
        if metrics['catch_rate'] >= 80 and metrics['fp_rate'] <= 25:
            balanced = (method, metrics)
            break
    
    if balanced:
        print(f"\nBalanced Option: {balanced[0]} (Catch: {balanced[1]['catch_rate']:.2f}%, FP: {balanced[1]['fp_rate']:.2f}%)")
    else:
        print("\nNo method found with both high catch rate (>=80%) and low FP rate (<=25%)")


def export_fused_predictions(csv_path, output_path, fusion_method='union',
                            threshold_general=0.2, threshold_sodium=0.35):
    # Export fused predictions to CSV for analysis
    # Includes individual model predictions and fused results
    print(f"\n{'='*70}")
    print(f"EXPORTING FUSED PREDICTIONS")
    print(f"{'='*70}")
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return
    
    # Load and process data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path.name}")
    
    # Process for both models
    X_general, gtin_general = process_and_embed_general(df.copy())
    X_sodium, gtin_sodium = process_and_embed_sodium(df.copy())
    
    # Align GTINs
    if gtin_general is not None and gtin_sodium is not None:
        common_gtins = set(gtin_general) & set(gtin_sodium)
        mask_general = gtin_general.isin(common_gtins)
        mask_sodium = gtin_sodium.isin(common_gtins)
        X_general = X_general[mask_general]
        X_sodium = X_sodium[mask_sodium]
        gtin_general = gtin_general[mask_general]
        gtin_sodium = gtin_sodium[mask_sodium]
        
        # Sort by GTIN
        general_df = pd.DataFrame({'gtin': gtin_general}).reset_index(drop=True)
        sodium_df = pd.DataFrame({'gtin': gtin_sodium}).reset_index(drop=True)
        general_df = general_df.sort_values('gtin').reset_index(drop=True)
        sodium_df = sodium_df.sort_values('gtin').reset_index(drop=True)
        
        X_general = X_general.loc[general_df.index].reset_index(drop=True)
        X_sodium = X_sodium.loc[sodium_df.index].reset_index(drop=True)
        gtin_general = general_df['gtin'].reset_index(drop=True)
        gtin_sodium = sodium_df['gtin'].reset_index(drop=True)
    
    # Load models
    general_model, sodium_model = load_models()
    
    # Get predictions
    probs_general = general_model.predict_proba(X_general)[:, 1]
    probs_sodium = sodium_model.predict_proba(X_sodium)[:, 1]
    
    preds_general = (probs_general >= threshold_general).astype(int)
    preds_sodium = (probs_sodium >= threshold_sodium).astype(int)
    
    # Fuse predictions
    fused_preds, fused_probs = fuse_predictions(
        probs_general, probs_sodium,
        method=fusion_method,
        threshold_general=threshold_general,
        threshold_sodium=threshold_sodium
    )
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'gtin': gtin_general if gtin_general is not None else gtin_sodium,
        'prob_general': probs_general,
        'prob_sodium': probs_sodium,
        'prob_fused': fused_probs,
        'pred_general': preds_general,
        'pred_sodium': preds_sodium,
        'pred_fused': fused_preds,
        'agreement': (preds_general == preds_sodium).astype(int),
        'both_flagged': ((preds_general == 1) & (preds_sodium == 1)).astype(int),
        'only_general': ((preds_general == 1) & (preds_sodium == 0)).astype(int),
        'only_sodium': ((preds_general == 0) & (preds_sodium == 1)).astype(int),
    })
    
    # Sort by fused probability (highest first)
    output_df = output_df.sort_values('prob_fused', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"\nExported {len(output_df):,} predictions to {output_path}")
    print(f"\nSummary:")
    print(f"  Fused anomalies: {fused_preds.sum():,} ({fused_preds.mean()*100:.2f}%)")
    print(f"  Model agreement: {(preds_general == preds_sodium).sum():,} ({(preds_general == preds_sodium).mean()*100:.2f}%)")
    print(f"  Both flagged: {((preds_general == 1) & (preds_sodium == 1)).sum():,}")
    print(f"  Only general: {((preds_general == 1) & (preds_sodium == 0)).sum():,}")
    print(f"  Only sodium: {((preds_general == 0) & (preds_sodium == 1)).sum():,}")
    
    return output_df


if __name__ == "__main__":
    import sys
    
    # Check if running on specific file
    if len(sys.argv) > 1 and sys.argv[1] == '--file':
        # Run on specific file
        data_folder = Path(__file__).parent / "Data"
        filename = sys.argv[2] if len(sys.argv) > 2 else None
        fusion_method = sys.argv[3] if len(sys.argv) > 3 else 'union'
        threshold_general = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
        threshold_sodium = float(sys.argv[5]) if len(sys.argv) > 5 else 0.35
        
        if not filename:
            print("Error: Please provide a filename")
            print(f"Usage: python ensemble_models.py --file <filename> [fusion_method] [threshold_general] [threshold_sodium]")
            sys.exit(1)
        
        test_file = data_folder / filename
        if not test_file.exists():
            print(f"Error: File not found: {test_file}")
            print(f"Looking in: {data_folder}")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"ENSEMBLE PREDICTION ON SPECIFIC FILE")
        print(f"{'='*70}")
        print(f"File: {test_file.name}")
        print(f"Fusion Method: {fusion_method}")
        print(f"Thresholds - General: {threshold_general}, Sodium: {threshold_sodium}")
        print(f"{'='*70}")
        
        result = test_ensemble_on_dataset(
            test_file, 
            "Testing", 
            test_file.stem,
            fusion_method=fusion_method,
            threshold_general=threshold_general,
            threshold_sodium=threshold_sodium,
            return_full_df=True
        )
        
        if result and 'full_dataframe' in result:
            # Export full dataframe with all original columns plus predictions
            output_file = data_folder / f"ensemble_predictions_{test_file.stem}_{fusion_method}.csv"
            df_output = result['full_dataframe']
            
            # Sort by fused probability (highest first) for easier review
            df_output = df_output.sort_values('prob_fused', ascending=False, na_position='last').reset_index(drop=True)
            
            df_output.to_csv(output_file, index=False)
            print(f"\n{'='*70}")
            print(f"Full predictions exported to: {output_file}")
            print(f"Total rows: {len(df_output):,}")
            print(f"Anomalies detected (is_anomaly=True): {df_output['is_anomaly'].sum():,} ({df_output['is_anomaly'].mean()*100:.2f}%)")
            print(f"{'='*70}")
        elif result:
            # Fallback if full dataframe not available
            output_file = data_folder / f"ensemble_predictions_{test_file.stem}_{fusion_method}.csv"
            output_df = pd.DataFrame({
                'gtin': result['gtin'] if result['gtin'] is not None else [''] * len(result['preds_fused']),
                'prob_general': result['probs_general'],
                'prob_sodium': result['probs_sodium'],
                'prob_fused': result['probs_fused'],
                'is_anomaly_general': result['preds_general'].astype(bool),
                'is_anomaly_sodium': result['preds_sodium'].astype(bool),
                'is_anomaly': result['preds_fused'].astype(bool),
            })
            output_df = output_df.sort_values('prob_fused', ascending=False).reset_index(drop=True)
            output_df.to_csv(output_file, index=False)
            print(f"\n{'='*70}")
            print(f"Predictions exported to: {output_file}")
            print(f"{'='*70}")
    
    # Check if export mode is requested
    elif len(sys.argv) > 1 and sys.argv[1] == '--export':
        # Export mode: export fused predictions to CSV
        data_folder = Path(__file__).parent / "Data"
        fusion_method = sys.argv[2] if len(sys.argv) > 2 else 'union'
        
        # Export errors dataset
        errors_file = data_folder / "AI Training Data - items data errors.csv"
        errors_output = data_folder / f"ensemble_predictions_errors_{fusion_method}.csv"
        print("\nExporting errors dataset predictions...")
        export_fused_predictions(errors_file, errors_output, fusion_method=fusion_method)
        
        # Export approved dataset
        approved_file = data_folder / "AI Training Data - approved profiles.csv"
        approved_output = data_folder / f"ensemble_predictions_approved_{fusion_method}.csv"
        print("\nExporting approved dataset predictions...")
        export_fused_predictions(approved_file, approved_output, fusion_method=fusion_method)
        
        print(f"\n{'='*70}")
        print("EXPORT COMPLETE")
        print(f"{'='*70}")
        print(f"Files saved:")
        print(f"  - {errors_output}")
        print(f"  - {approved_output}")
    else:
        # Normal testing mode
        main()

