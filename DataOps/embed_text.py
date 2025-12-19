import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Text columns to embed
TEXT_COLS = ["category_name", "product_name", "ingredients_text"]

# Nutrient columns (numeric)
NUTRIENT_COLS = [
    "calories", "total_fat", "sat_fat", "trans_fat", "unsat_fat",
    "cholesterol", "sodium", "carbs", "dietary_fiber",
    "total_sugars", "added_sugars", "protein", "potassium",
]


def embed_text_data():
    # Load merged.csv
    data_folder = Path(__file__).parent.parent / "Data" / "PreOpDataCSV"
    merged_file = data_folder / "merged.csv"
    
    print(f"Loading {merged_file}...")
    df = pd.read_csv(merged_file)
    print(f"Loaded {len(df):,} rows")
    
    # Initialize SentenceTransformer model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed each text column
    for col in TEXT_COLS:
        if col in df.columns:
            print(f"Embedding {col}...")
            # Convert to string and handle NaN
            text_data = df[col].fillna("").astype(str)
            # Generate embeddings (returns numpy array of shape [n_samples, 384])
            embeddings = model.encode(text_data.tolist())
            # Create column names for embeddings
            embedding_cols = [f"{col}_emb_{i}" for i in range(embeddings.shape[1])]
            # Create DataFrame with embedding columns
            embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
            # Add embedding columns to original dataframe
            df = pd.concat([df, embedding_df], axis=1)
            print(f"  Added {len(embedding_cols)} embedding columns for {col}")
    
    # Drop original text columns (keep only embeddings)
    df = df.drop(columns=TEXT_COLS, errors="ignore")
    
    # Fill NaN in numeric nutrition columns with -1 (explicit missing value marker)
    for col in NUTRIENT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
    
    # Save as Parquet file in PostOpData folder
    output_folder = Path(__file__).parent.parent / "Data" / "PostOpData"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "merged_embedded.parquet"
    
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns to {output_file}")
    
    return df


if __name__ == "__main__":
    embed_text_data()

