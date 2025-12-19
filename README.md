# QA Model - Anomaly Detection System

An XGBoost-based machine learning system for detecting data quality anomalies in nutrition product data.

## Overview

This project builds an anomaly detection model that identifies data entry errors and inconsistencies in product nutrition data. The model uses SentenceTransformer embeddings for text features and XGBoost for classification, trained on Before/After data pairs where human reviewers marked anomalies with carets (^).

## Project Structure

```
QAModel_1/
├── Data/
│   ├── PreOpData/          # Raw Excel files (Before/After pairs)
│   ├── PreOpDataCSV/       # Processed CSV files
│   └── PostOpData/         # Final embedded datasets
├── DataOps/
│   ├── addcaret.py         # Extract Excel data and add caret markers
│   ├── cleandata.py        # Data cleaning and preprocessing
│   ├── mergedata.py        # Merge Before/After datasets
│   └── embed_text.py       # Embed text columns using SentenceTransformer
├── ModelTraining/
│   ├── train_model.py      # Train final model with optimized hyperparameters
│   ├── optimize_model.py   # Hyperparameter optimization with Optuna
│   └── evaluate_model.py   # Model evaluation and production metrics
└── ModelTests/
    └── test_model.py       # Test model on validation datasets
```

## Data Pipeline

### 1. Data Extraction (`addcaret.py`)
- Reads Excel files from `PreOpData/` folder
- Processes "all_scores" sheet
- Adds caret (^) markers to cells with yellow or cyan background colors
- Outputs CSV files to `PreOpDataCSV/`

### 2. Data Cleaning (`cleandata.py`)
- Subsets to target columns (ID, text, nutrients, tags, flags)
- Creates `label_is_anomaly` from carets in original data
- Converts nutrient columns to numeric (removes carets, fills NaN with -1)
- Converts tags/flags to boolean
- Removes rows without GTIN or empty ingredients_text
- Removes duplicate GTINs

### 3. Data Merging (`mergedata.py`)
- Merges all Before files together
- Merges all After files together
- Aligns Before/After by GTIN
- Takes features from Before, label from After
- Outputs `merged.csv`

### 4. Text Embedding (`embed_text.py`)
- Uses SentenceTransformer model `all-MiniLM-L6-v2`
- Embeds: category_name, product_name, ingredients_text
- Each text column becomes 384 numeric embedding features (1,152 total)
- Fills NaN in nutrient columns with -1
- Outputs `merged_embedded.parquet`

## Model Training

### Hyperparameter Optimization (`optimize_model.py`)
- Uses Optuna for hyperparameter search (50 trials)
- Searches: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma
- Optimizes for Average Precision (best for rare anomaly detection)
- Best parameters saved to `best_hyperparams.json`

**Optimized Hyperparameters:**
- n_estimators: 394
- max_depth: 4
- learning_rate: 0.015
- subsample: 0.757
- colsample_bytree: 0.669
- min_child_weight: 3
- gamma: 0.442
- Best Average Precision: 0.513

### Model Training (`train_model.py`)
- Loads optimized hyperparameters
- Trains XGBoost classifier with class imbalance handling (scale_pos_weight)
- Uses 80/20 train/test split with stratification
- Evaluates multiple thresholds (0.7, 0.5, 0.3, 0.29-0.21, 0.2)
- Saves model as `final_anomaly_detector.json`

### Model Evaluation (`evaluate_model.py`)
- Precision-Recall curve visualization
- Recall@K analysis (shows how many errors caught when checking top K items)
- Precision by nutrient segments
- False positive audit export

## Results

### Test Dataset Performance (Threshold: 0.35)

**Errors Dataset (Should be 100% anomalies):**
- Catch Rate (Recall): **89.27%** (283 out of 317 detected)
- Average Probability: 0.691
- High Confidence (≥0.7): 176 items
- Medium Confidence (0.3-0.7): 116 items
- Low Confidence (<0.3): 25 items

**Approved Dataset (Should be 0% anomalies):**
- False Positive Rate: **21.45%** (68 out of 317 flagged)
- Average Probability: 0.287
- High Confidence (≥0.7): 3 items
- Medium Confidence (0.3-0.7): 124 items
- Low Confidence (<0.3): 190 items

### Summary
- **Overall Catch Rate**: 89.27% - The model successfully identifies the vast majority of known errors
- **False Positive Rate**: 21.45% - When reviewing flagged items from clean data, about 1 in 5 will be false alarms

## Features Used

- **Text Embeddings** (1,152 features): SentenceTransformer embeddings for category_name, product_name, ingredients_text
- **Nutrient Columns** (13 features): calories, total_fat, sat_fat, trans_fat, unsat_fat, cholesterol, sodium, carbs, dietary_fiber, total_sugars, added_sugars, protein, potassium
- **Tag Columns** (29 boolean features): is_whole_grain, is_omega_three, is_healthy_oils, etc.
- **Flag Columns** (10 boolean features): flag_calorie_mismatch, flag_fat_mismatch, flag_carb_mismatch, etc.

**Total Features**: ~1,204 features

## Usage

### Running the Full Pipeline

1. **Extract and process Excel files:**
   ```bash
   python DataOps/addcaret.py
   ```

2. **Clean and merge data:**
   ```bash
   python DataOps/mergedata.py
   ```

3. **Embed text columns:**
   ```bash
   python DataOps/embed_text.py
   ```

4. **Train the model:**
   ```bash
   python ModelTraining/train_model.py
   ```

5. **Evaluate the model:**
   ```bash
   python ModelTraining/evaluate_model.py
   ```

6. **Test on validation datasets:**
   ```bash
   python ModelTests/test_model.py
   ```

### Using the Model for Inference

```python
import pandas as pd
import xgboost as xgb
from pathlib import Path

# Load model
model = xgb.XGBClassifier()
model.load_model("ModelTraining/final_anomaly_detector.json")

# Process your data through the same pipeline (clean -> embed)
# Then predict:
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.35).astype(int)  # Using threshold 0.35
```

## Key Design Decisions

1. **Caret Markers as Ground Truth**: Original data had carets (^) marking anomalies, which were used to create binary labels
2. **Before Features + After Labels**: Model learns from "messy" Before data, predicts based on what was corrected in After data
3. **SentenceTransformer Embeddings**: Converts text to dense numerical representations for XGBoost
4. **Custom Threshold**: Uses 0.35 probability threshold (instead of default 0.5) to balance recall and false positives
5. **Class Imbalance Handling**: Uses scale_pos_weight to handle highly imbalanced anomaly detection task

## Dependencies

- pandas
- xgboost
- scikit-learn
- sentence-transformers
- optuna (for hyperparameter optimization)
- matplotlib (for evaluation plots)
- pyarrow (for Parquet file support)
- openpyxl (for Excel file processing)

## Notes

- The model is trained to catch ~90% of errors while maintaining reasonable false positive rates
- Threshold selection is a trade-off: lower thresholds catch more errors but increase false positives
- The 21.45% false positive rate means the model is conservative - it's better to flag potentially problematic items for review than to miss real errors

