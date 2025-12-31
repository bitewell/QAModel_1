import pandas as pd


# Column definitions - Only sodium and text columns for sodium-specific anomaly detection
ID_COL = ["gtin"]

TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

# Only sodium nutrient column
NUTRIENT_COLS = ["sodium"]

COLS_TO_KEEP = ID_COL + TEXT_COLS + NUTRIENT_COLS + ["label_is_anomaly"]


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    # Cleans and preprocesses dataframe for model training
    initial_count = len(df)
    

    # Keep only target columns
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()


    # Convert ID and text columns to string
    for col in ID_COL + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")


    # Create label_is_anomaly from carets in original data (before cleaning)
    # Only check sodium and text columns for carets
    df["label_is_anomaly"] = 0
    cols_with_carets = TEXT_COLS + NUTRIENT_COLS  # Only text columns + sodium
    for col in cols_with_carets:
        if col in df.columns:
            col_str = df[col].astype(str).str.strip()
            has_caret = col_str.fillna("").str.endswith("^")
            df["label_is_anomaly"] = df["label_is_anomaly"] | has_caret.astype(int)


    # Convert sodium to numeric (remove carets, fill NaN with -1)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            df[col] = df[col].fillna(-1)


    # Remove rows without GTIN
    before_gtin = len(df)
    df = df.dropna(subset=["gtin"])
    removed_no_gtin = before_gtin - len(df)


    # Remove rows with empty ingredients_text
    removed_empty_ingredients = 0
    if "ingredients_text" in df.columns:
        before_ingredients = len(df)
        df = df[
            df["ingredients_text"].notna() & 
            (df["ingredients_text"].astype(str).str.strip() != "")
        ]
        removed_empty_ingredients = before_ingredients - len(df)


    # Remove duplicate GTINs (keep first occurrence)
    before_duplicates = len(df)
    df = df.sort_values("gtin")
    df = df.drop_duplicates(subset=["gtin"], keep="first")
    removed_duplicates = before_duplicates - len(df)


    final_count = len(df)
    total_removed = initial_count - final_count


    if verbose:
        print(f"  Initial rows: {initial_count:,}")
        print(f"  Removed (no GTIN): {removed_no_gtin:,}")
        print(f"  Removed (empty ingredients_text): {removed_empty_ingredients:,}")
        print(f"  Removed (duplicate GTINs): {removed_duplicates:,}")
        print(f"  Final rows: {final_count:,}")
        print(f"  Total removed: {total_removed:,}")

    return df
