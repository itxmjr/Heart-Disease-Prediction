"""
Data Preprocessing Module for Heart Disease Prediction
======================================================

This module contains functions to:
1. Explore basic information
2. Handle missing values
3. Clean and prepare data for modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform initial data exploration to understand structure and missing values.
    """
    exploration_results = {}
    
    print("\n" + "="*60)
    print("DATA EXPLORATION REPORT")
    print("="*60)
    
    # 1. First few rows
    print("\n1. FIRST 5 ROWS:")
    print("-"*40)
    print(df.head())
    
    # 2. Data types and info
    print("\n2. DATA TYPES:")
    print("-"*40)
    print(df.dtypes)
    exploration_results['dtypes'] = df.dtypes.to_dict()
    
    # 3. Missing values
    print("\n3. MISSING VALUES:")
    print("-"*40)
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 
          else "No missing values!")
    exploration_results['missing_values'] = missing.to_dict()
    
    # 4. Basic statistics
    print("\n4. STATISTICAL SUMMARY:")
    print("-"*40)
    print(df.describe())
    
    # 5. Target distribution
    print("\n5. TARGET VARIABLE DISTRIBUTION:")
    print("-"*40)
    target_counts = df['target'].value_counts()
    print(f"   No Heart Disease (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   Heart Disease (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    exploration_results['target_distribution'] = target_counts.to_dict()
    
    # 6. Check for duplicates
    print("\n6. DUPLICATE ROWS:")
    print("-"*40)
    duplicates = df.duplicated().sum()
    print(f"   {duplicates} duplicate rows found")
    exploration_results['duplicates'] = duplicates
    
    return exploration_results


def check_data_quality(df: pd.DataFrame) -> bool:
    """
    Check for data quality issues.
    Returns True if issues are found, False otherwise.
    """
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    issues = []
    
    # Age should be positive and reasonable
    if (df['age'] < 0).any() or (df['age'] > 120).any():
        issues.append("Invalid age values detected")
    
    # Sex should be 0 or 1
    if not df['sex'].isin([0, 1]).all():
        issues.append("Invalid sex values (should be 0 or 1)")
    
    # Blood pressure should be positive
    if (df['trestbps'] <= 0).any():
        issues.append("Invalid blood pressure values")
    
    # Cholesterol should be positive
    if (df['chol'] <= 0).any():
        issues.append("Invalid cholesterol values")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return True
    else:
        print("No obvious data quality issues!")
        return False


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset using specified strategy.
    """
    df_cleaned = df.copy()
    
    missing_before = df_cleaned.isnull().sum().sum()
    
    if missing_before == 0:
        print("No missing values to handle!")
        return df_cleaned
    
    if strategy == 'median':
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    elif strategy == 'mean':
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    elif strategy == 'drop':
        df_cleaned.dropna(inplace=True)
    
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Missing values handled: {missing_before} -> {missing_after}")
    
    return df_cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    """
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    
    df_cleaned.drop_duplicates(inplace=True)
    
    removed = initial_rows - len(df_cleaned)
    if removed > 0:
        print(f"Removed {removed} duplicate rows")
    else:
        print("No duplicates to remove")
    
    return df_cleaned


def get_feature_names() -> Dict[str, str]:
    """
    Get human-readable names for features.
    """
    return {
        'age': 'Age (years)',
        'sex': 'Sex (1=Male, 0=Female)',
        'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'restecg': 'Resting ECG Results',
        'thalach': 'Maximum Heart Rate',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'Slope of Peak Exercise ST',
        'ca': 'Number of Major Vessels (0-3)',
        'thal': 'Thalassemia',
        'target': 'Heart Disease (1=Yes, 0=No)'
    }

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from dataloader import load_data
    
    # Test the functions
    df = load_data("data/heart.csv")
    
    if df is not None:
        explore_data(df)
        check_data_quality(df)
        df_clean = handle_missing_values(df)
        df_clean = remove_duplicates(df_clean)
        print(f"\nFinal dataset shape: {df_clean.shape}")