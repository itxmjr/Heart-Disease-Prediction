"""
Main Project Runner
===================

This script runs the complete project pipeline:
1. Load and clean data
2. Run EDA
3. Train and evaluate models
4. Launch the web app
"""

import os
import sys
sys.path.append('src')

from src.dataloader import load_data, download_dataset
from src.preprocessor import (
    explore_data, 
    check_data_quality,
    handle_missing_values,
    remove_duplicates
)
from src.eda import run_complete_eda
from src.model import run_complete_ml_pipeline


def run_complete_project():
    """Run the entire project pipeline."""
    
    print("="*70)
    print("HEART DISEASE PREDICTION PROJECT")
    print("="*70)
    print()
    
    # Create directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and explore data
    print("\n" + "="*70)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*70)
    
    data_path = download_dataset()
    if not data_path:
        print("Failed to download dataset.")
        return

    # In our project structure, the symlink points to the folder containing heart.csv
    csv_path = data_path / "heart.csv"
    
    # Fallback checks
    if not csv_path.exists():
        if "heart.csv" in str(data_path):
             csv_path = data_path
        else:
             print(f"Warning: heart.csv not found in {data_path}. Checking for alternatives...")
             # List files to help debug if needed
             try:
                 print(f"Files in {data_path}: {[f.name for f in data_path.iterdir()]}")
             except:
                 pass
             csv_path = data_path / "heart.csv"

    df = load_data(str(csv_path))
    
    if df is None:
        print("Please ensure the dataset is downloaded correctly!")
        return
    
    explore_data(df)
    check_data_quality(df)
    
    # Step 2: Clean data
    print("\n" + "="*70)
    print("STEP 2: CLEANING DATA")
    print("="*70)
    
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    # Step 3: EDA
    print("\n" + "="*70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    run_complete_eda(df, save_dir='outputs/')
    
    # Step 4: ML Pipeline
    print("\n" + "="*70)
    print("STEP 4: MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # Note: We are using the cleaned dataframe directly. 
    # For this specific project/app, we treat integer categories as numerical features 
    # to maintain compatibility with the simple input form in app.py.
    
    model = run_complete_ml_pipeline(df, save_dir='outputs/')
    
    # Final summary
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print()
    print("Outputs saved to: outputs/")
    print("Model saved to: models/heart_model.pkl")
    print()
    print("To launch the web app, run:")
    print("   streamlit run app.py")
    print()
    print("="*70)
    
    return model


if __name__ == "__main__":
    run_complete_project()