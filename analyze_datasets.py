#!/usr/bin/env python3
"""
Demonstration of feature types across different datasets
"""

import pandas as pd
from automl_pipeline import detect_cat_num

def analyze_dataset(file_path, dataset_name):
    print(f"\n🔍 ANALYZING {dataset_name.upper()}")
    print("="*50)
    
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        numerical_features = []
        categorical_features = []
        
        # Assume last column is target for this demo
        feature_cols = df.columns[:-1]  # All except last
        target_col = df.columns[-1]     # Last column
        
        print(f"\n🎯 Target: {target_col} (type: {detect_cat_num(df[target_col])})")
        print(f"🔧 Features being analyzed: {len(feature_cols)}")
        
        for col in feature_cols:
            feature_type = detect_cat_num(df[col])
            unique_count = df[col].nunique()
            print(f"   {col}: {feature_type} (unique: {unique_count})")
            
            if feature_type == 'numerical':
                numerical_features.append(col)
            elif feature_type == 'categorical':
                categorical_features.append(col)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Numerical features: {len(numerical_features)} -> {numerical_features}")
        print(f"   Categorical features: {len(categorical_features)} -> {categorical_features}")
        
        # Plotting recommendation
        if len(categorical_features) == 0:
            print(f"   ⚠️  NO categorical plots will be generated (no categorical features)")
            print(f"   ✅  {len(numerical_features)} numerical plots will be generated")
        else:
            print(f"   ✅  {len(categorical_features)} categorical plots will be generated")
            print(f"   ✅  {len(numerical_features)} numerical plots will be generated")
            
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")

def main():
    datasets = [
        ('Datasets/iris.csv', 'Iris'),
        ('Datasets/heart.csv', 'Heart Disease'),
        ('Datasets/bank.csv', 'Bank Marketing'),
        ('Datasets/insurance.csv', 'Insurance')
    ]
    
    for file_path, name in datasets:
        try:
            analyze_dataset(file_path, name)
        except:
            print(f"⚠️ Could not analyze {name} - file may not exist")

if __name__ == "__main__":
    main()
