#!/usr/bin/env python3
"""
Simple debug test for categorical features
"""

import pandas as pd
import sys
import os
sys.path.append(os.getcwd())

from automl_pipeline import detect_cat_num, plot_categorical_features, plot_numerical_histograms

def simple_test():
    print("🔍 SIMPLE FEATURE DETECTION TEST")
    print("="*50)
    
    # Test with heart dataset (has categorical features)
    print("📊 Testing with heart.csv (has categorical features)")
    df = pd.read_csv('Datasets/heart.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Test first few columns
    test_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'FastingBS']
    
    numerical_features = []
    categorical_features = []
    
    for col in test_cols:
        if col in df.columns:
            feature_type = detect_cat_num(df[col])
            print(f"{col}: {feature_type} (dtype: {df[col].dtype}, unique: {df[col].nunique()})")
            
            if feature_type == 'numerical':
                numerical_features.append(col)
            elif feature_type == 'categorical':
                categorical_features.append(col)
    
    print(f"\nNumerical: {numerical_features}")
    print(f"Categorical: {categorical_features}")
    
    # Test plotting with a small subset
    if categorical_features:
        print(f"\n📈 Testing categorical plots...")
        cat_plots = plot_categorical_features(df[categorical_features], categorical_features)
        print(f"Generated {len(cat_plots)} categorical plots")
    
    if numerical_features:
        print(f"\n📊 Testing numerical plots...")
        num_plots = plot_numerical_histograms(df[numerical_features], numerical_features)
        print(f"Generated {len(num_plots)} numerical plots")

if __name__ == "__main__":
    simple_test()
