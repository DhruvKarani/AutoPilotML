#!/usr/bin/env python3
"""
Debug script to test feature detection
"""

import pandas as pd
import numpy as np
from automl_pipeline import detect_cat_num

def debug_iris():
    print("🔍 DEBUGGING IRIS DATASET")
    print("="*50)
    
    # Load iris dataset
    df = pd.read_csv('Datasets/iris.csv')
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Check each column
    for col in df.columns:
        print(f"\n🔍 Column: {col}")
        print(f"   Data type: {df[col].dtype}")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Sample values: {df[col].unique()[:5]}")
        feature_type = detect_cat_num(df[col])
        print(f"   Detected as: {feature_type}")

def test_with_categorical_data():
    print("\n\n🧪 TESTING WITH SAMPLE CATEGORICAL DATA")
    print("="*50)
    
    # Create a test dataset with categorical features
    test_data = {
        'age': [25, 30, 35, 40, 45],  # numerical
        'gender': ['M', 'F', 'M', 'F', 'M'],  # categorical (object)
        'binary_feature': [0, 1, 0, 1, 0],  # binary (should be categorical)
        'multi_category': [1, 2, 3, 1, 2],  # categorical (low unique count)
        'continuous': [1.5, 2.7, 3.8, 4.2, 5.1]  # numerical
    }
    
    df_test = pd.DataFrame(test_data)
    
    print(f"📊 Test dataset shape: {df_test.shape}")
    print(f"📋 Columns: {df_test.columns.tolist()}")
    
    # Check each column
    for col in df_test.columns:
        print(f"\n🔍 Column: {col}")
        print(f"   Data type: {df_test[col].dtype}")
        print(f"   Unique values: {df_test[col].nunique()}")
        print(f"   Sample values: {df_test[col].unique()}")
        feature_type = detect_cat_num(df_test[col])
        print(f"   Detected as: {feature_type}")

if __name__ == "__main__":
    debug_iris()
    test_with_categorical_data()
