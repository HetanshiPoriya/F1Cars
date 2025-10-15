import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import os

# --- Configuration ---
DATA_FILENAME = 'f1_dnf.csv' 
MODEL_FILENAME = 'F1_Racing.joblib'
PREPROCESSOR_FILENAME = 'F1_Racing_preprocessor.joblib'

# --- Feature Mapping (What the Model Expects) ---
NUMERICAL_FEATURES = ['year', 'round', 'grid', 'laps']
CATEGORICAL_FEATURES = ['driver', 'circuit_name', 'constructor_name']
TARGET_FEATURE = 'target_finish'
ALL_REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_FEATURE]

def train_and_save_model():
    """Loads data, trains the model, and saves the joblib files."""
    print("--- Starting Model Training & Preprocessor Creation ---")
    
    try:
        # Load data directly from your local file
        df = pd.read_csv(DATA_FILENAME)
        
        # --------------------------------------------------------------------------------
        # --- FINAL FAILSAFE FIX: Standardize, Rename, and Force Column Names ---
        # --------------------------------------------------------------------------------
        
        # 1. Standardize column names (strip whitespace and convert to lowercase)
        #    This fixes most common casing and spacing issues.
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. Force Rename: Find the column containing 'driver' or 'team'
        #    This is the ultimate attempt to ensure the required columns exist.
        
        # --- RENAME DRIVER COLUMN ---
        driver_col_candidates = [col for col in df.columns if 'driver' in col]
        if 'driver' not in df.columns and driver_col_candidates:
            # If the required name 'driver' is missing, but a candidate exists, rename the first one found.
            df = df.rename(columns={driver_col_candidates[0]: 'driver'})
            print(f"Renamed column '{driver_col_candidates[0]}' to 'driver'.")

        # --- RENAME CIRCUIT NAME COLUMN ---
        circuit_col_candidates = [col for col in df.columns if 'circuit' in col]
        if 'circuit_name' not in df.columns and circuit_col_candidates:
            df = df.rename(columns={circuit_col_candidates[0]: 'circuit_name'})
            print(f"Renamed column '{circuit_col_candidates[0]}' to 'circuit_name'.")

        # --- RENAME CONSTRUCTOR COLUMN ---
        constructor_col_candidates = [col for col in df.columns if 'constructor' in col]
        if 'constructor_name' not in df.columns and constructor_col_candidates:
            df = df.rename(columns={constructor_col_candidates[0]: 'constructor_name'})
            print(f"Renamed column '{constructor_col_candidates[0]}' to 'constructor_name'.")
        
        
        # --- FINAL CRITICAL CHECK ---
        current_cols_lower = df.columns
        missing_columns = [col for col in ALL_REQUIRED_COLUMNS if col not in current_cols_lower]
        
        if missing_columns:
            print("\n❌ FATAL ERROR: Missing Required Columns")
            print(f"The model requires columns that are not present in {DATA_FILENAME}: {missing_columns}")
            print("\nAction Required: Please ensure your CSV file contains columns that can be mapped to these names.")
            
            # Print current columns to aid debugging if the script still fails
            print("\n--- Current Columns Found (Lowercase) ---")
            print(list(current_cols_lower))
            print("-----------------------------------------")
            return

    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILENAME}' not found in the current directory ({os.getcwd()}).")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare data for Scikit-learn
    # NOTE: We use the lowercased column names here because we standardized them above.
    X = df[[col.lower() for col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES]]
    y = df[TARGET_FEATURE.lower()]
    
    # --- Create Preprocessing Pipeline (ColumnTransformer) ---
    numerical_pipeline = StandardScaler() 
    categorical_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('onehot', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    print("Fitting preprocessor to the data and training model features...")
    X_processed = preprocessor.fit_transform(X)

    # --- Train the Random Forest Model ---
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    print("Training the Random Forest model...")
    model.fit(X_processed, y)

    # --- Save Resources ---
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(preprocessor, PREPROCESSOR_FILENAME)
    
    print(f"\n✅ Success! Both deployment files have been created:")
    print(f"- {MODEL_FILENAME}")
    print(f"- {PREPROCESSOR_FILENAME}")
    print("\nNext step: Run 'python -m streamlit run app.py' to launch the dashboard.")

if __name__ == "__main__":
    train_and_save_model()
