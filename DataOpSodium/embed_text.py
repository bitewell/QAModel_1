import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

# Only sodium nutrient column for sodium-specific anomaly detection
NUTRIENT_COLS = ["sodium"]

# Columns to keep in final dataset: text embeddings + sodium + label
COLS_TO_KEEP = TEXT_COLS + NUTRIENT_COLS + ["label_is_anomaly", "gtin"]


def embed_text_data():
    # Embeds text columns using SentenceTransformer and saves as Parquet
    # Only keeps: text embeddings, sodium column, and label_is_anomaly
    data_folder = Path(__file__).parent.parent / "DataSodium" / "PreOpDataCSV"
    merged_file = data_folder / "merged.csv"
    
    print(f"Loading {merged_file}...")
    df = pd.read_csv(merged_file)
    print(f"Loaded {len(df):,} rows")
    
    # Select only the columns we need: TEXT_COLS + NUTRIENT_COLS + label + gtin
    available_cols = [c for c in COLS_TO_KEEP if c in df.columns]
    df = df[available_cols].copy()
    print(f"Selected {len(available_cols)} columns: {available_cols}")
    
    # Initialize SentenceTransformer model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed each text column (creates 384 numeric features per column)
    for col in TEXT_COLS:
        if col in df.columns:
            print(f"Embedding {col}...")
            text_data = df[col].fillna("").astype(str)
            embeddings = model.encode(text_data.tolist())
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            df = pd.concat([df, embedding_df], axis=1)
            print(f"  Added {len(embedding_cols)} embedding columns for {col}")
    
    # Drop original text columns (keep only embeddings)
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    # Fill NaN in sodium column with -1
    for col in NUTRIENT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    
    # Verify we only have: embeddings + sodium + label + gtin
    final_cols = df.columns.tolist()
    print(f"\nFinal feature set:")
    print(f"  Text embeddings: {len([c for c in final_cols if '_emb_' in c])} features")
    print(f"  Sodium column: {len([c for c in final_cols if c in NUTRIENT_COLS])} feature")
    print(f"  Label column: {'label_is_anomaly' if 'label_is_anomaly' in final_cols else 'MISSING'}")
    print(f"  ID column: {'gtin' if 'gtin' in final_cols else 'MISSING'}")
    print(f"  Total columns: {len(final_cols)}")
    
    # Save as Parquet file
    output_folder = Path(__file__).parent.parent / "DataSodium" / "PostOpData"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "merged_embedded.parquet"
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns to {output_file}")
    
    return df


if __name__ == "__main__":
    embed_text_data()
