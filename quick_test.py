#!/usr/bin/env python3
"""
Direct test of AutoML pipeline
"""

import pandas as pd
from automl_pipeline import run_automl_pipeline
import warnings
warnings.filterwarnings('ignore')

# Example 1: Using Iris dataset for classification
print("🌸 Testing with Iris Dataset (Classification)")
print("="*50)

try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    # Map to string labels
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    iris_df['species'] = iris_df['species'].map(target_names)
    
    print(f"📊 Dataset shape: {iris_df.shape}")
    print(f"🎯 Target: species")
    print("🚀 Running pipeline...")
    
    results = run_automl_pipeline(
        df=iris_df,
        target_col='species',
        model_choice='utility'
    )
    
    print("\n✅ RESULTS:")
    print(f"🏆 Best Model: {results['best_model']}")
    print(f"⭐ Rating: {results['overall_rating']:.1f}/10")
    print(f"🎯 Accuracy: {results['metrics'].get('accuracy', 'N/A')}")
    print(f"💾 Model: {results['model_path']}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🏠 Testing with Housing Dataset (Regression)")
print("="*50)

try:
    # Create synthetic regression data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    housing_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    housing_df['price'] = y
    
    print(f"📊 Dataset shape: {housing_df.shape}")
    print(f"🎯 Target: price")
    print("🚀 Running pipeline...")
    
    results = run_automl_pipeline(
        df=housing_df,
        target_col='price',
        model_choice='utility'
    )
    
    print("\n✅ RESULTS:")
    print(f"🏆 Best Model: {results['best_model']}")
    print(f"⭐ Rating: {results['overall_rating']:.1f}/10")
    print(f"📊 MAE: {results['metrics'].get('mae', 'N/A')}")
    print(f"📊 R²: {results['metrics'].get('r2', 'N/A')}")
    print(f"💾 Model: {results['model_path']}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n🎉 Testing completed!")
