import pandas as pd


# ID column
ID_COL = ["gtin"]

# Text columns
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

# Nutrient columns (numeric)
NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]

# Tag columns (boolean)
TAG_COLS = [
    "is_whole_grain", "is_omega_three", "is_healthy_oils", "is_healthy_fats",
    "is_seed_oil", "is_refined_grains", "is_deep_fried", "is_sugars_added",
    "is_artificial_sweeteners", "is_artificial_flavors",
    "is_artificial_preservatives", "is_artificial_colors",
    "is_artificial_red_color", "is_ph_oil", "is_aspartame",
    "is_acesulfame_potassium", "is_saccharin", "is_corn_syrup",
    "is_brominated_vegetable_oil", "is_potassium_bromate",
    "is_titanium_dioxide", "is_phosphate_additives", "is_polysorbate60",
    "is_mercury_fish", "is_caregeenan", "is_natural_non_kcal_sweeteners",
    "is_natural_additives", "is_unspecific_ingredient", "is_propellant",
    "is_starch", "is_active_live_cultures",
]

# Existing rule-based flags (boolean)
FLAG_COLS = [
    "flag_calorie_mismatch",
    "flag_fat_mismatch",
    "flag_carb_mismatch",
    "flag_sugar_mismatch",
    "flag_missing_added_sugars",
    "flag_extra_added_sugars",
    "flag_low_sodium",
    "flag_high_sodium",
    "flag_negative_values",
    "flag_type_error",
]

# All columns to keep in the cleaned dataset
COLS_TO_KEEP = ID_COL + TEXT_COLS + NUTRIENT_COLS + TAG_COLS + FLAG_COLS + ["label_is_anomaly"]


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:

    initial_count = len(df)
    
    # 1. Subset to final columns
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[keep].copy()
    

    # 2. Convert ID + text to string
    for col in ID_COL + TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")
    

    # 2.5. Create label_is_anomaly based on carets in all columns that can have carets
    # Check for carets BEFORE processing/cleaning the columns
    df["label_is_anomaly"] = 0
    cols_with_carets = TEXT_COLS + NUTRIENT_COLS + TAG_COLS
    for col in cols_with_carets:
        if col in df.columns:
            # Convert to string, handle NaN properly, then check for caret
            col_str = df[col].astype(str).str.strip()
            # Check for caret (handle NaN as empty string which won't end with "^")
            has_caret = col_str.fillna("").str.endswith("^")
            df["label_is_anomaly"] = df["label_is_anomaly"] | has_caret.astype(int)
    

    # 3. Nutrient columns → clean float (remove carets, keep pure numbers for model)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            # Strip caret, convert to numeric (validates and cleans the number)
            numeric_val = df[col].astype(str).str.rstrip("^").str.strip()
            df[col] = pd.to_numeric(numeric_val, errors="coerce")
            # Fill NaN with -1 as sentinel value for missing/wrong data
            df[col] = df[col].fillna(-1)
            # We do NOT modify negatives — keep them as anomalies
    

    # 4. Tags + flags → bool
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
    

    # 5. Drop rows without GTIN (can't merge these)
    before_gtin = len(df)
    df = df.dropna(subset=["gtin"])
    removed_no_gtin = before_gtin - len(df)
    

    # 6. Remove rows with empty ingredients_text (null, empty string, or whitespace only)
    removed_empty_ingredients = 0


    if "ingredients_text" in df.columns:
        before_ingredients = len(df)
        # Remove rows where ingredients_text is null, empty string, or whitespace only
        df = df[
            df["ingredients_text"].notna() & 
            (df["ingredients_text"].astype(str).str.strip() != "")
        ]
        removed_empty_ingredients = before_ingredients - len(df)
    

    # 7. Remove duplicate GTINs
    before_duplicates = len(df)
    df = df.sort_values("gtin")
    df = df.drop_duplicates(subset=["gtin"], keep="first")
    removed_duplicates = before_duplicates - len(df)
    
    # 8. Return the cleaned dataframe
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
