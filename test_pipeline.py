#!/usr/bin/env python3
"""
Test script for the AutoML pipeline
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import sys
import traceback

# Import our AutoML pipeline
from automl_pipeline import run_automl_pipeline

def test_classification():
    """Test with Iris dataset"""
    print("🧪 Testing Classification with Iris dataset...")
    
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    print(f"✅ Loaded Iris dataset: {df.shape}")
    print(f"✅ Target distribution:\n{df['target'].value_counts()}")
    
    try:
        # Run the pipeline
        results = run_automl_pipeline(
            df=df, 
            target_col='target', 
            model_choice='utility',
            selected_class_for_roc='2'  # Test multiclass ROC
        )
        
        print("\n🎉 CLASSIFICATION TEST COMPLETED!")
        print(f"✅ Best Model: {results['best_model']}")
        print(f"✅ Task: {results['task']}")
        print(f"✅ Overall Rating: {results['overall_rating']}")
        print(f"✅ Plots Generated: {len(results['plots'])}")
        print(f"✅ Model Path: {results['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_regression():
    """Test with synthetic regression dataset"""
    print("\n🧪 Testing Regression with synthetic dataset...")
    
    # Create a simple regression dataset
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        # use np.random.choice for categorical sampling
        'feature_3': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'feature_4': np.random.randint(0, 2, n_samples),
    })
    
    # Create target with some noise
    df['target'] = (
        2 * df['feature_1'] + 
        1.5 * df['feature_2'] + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    print(f"✅ Created synthetic dataset: {df.shape}")
    print(f"✅ Target stats: mean={df['target'].mean():.2f}, std={df['target'].std():.2f}")
    
    try:
        # Run the pipeline
        results = run_automl_pipeline(
            df=df, 
            target_col='target', 
            model_choice='utility'
        )
        
        print("\n🎉 REGRESSION TEST COMPLETED!")
        print(f"✅ Best Model: {results['best_model']}")
        print(f"✅ Task: {results['task']}")
        print(f"✅ Overall Rating: {results['overall_rating']}")
        print(f"✅ Plots Generated: {len(results['plots'])}")
        print(f"✅ Model Path: {results['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Regression test failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting AutoML Pipeline Tests...\n")
    
    # Test classification
    classification_success = test_classification()
    
    # Test regression
    regression_success = test_regression()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Classification Test: {'✅ PASSED' if classification_success else '❌ FAILED'}")
    print(f"Regression Test: {'✅ PASSED' if regression_success else '❌ FAILED'}")
    
    if classification_success and regression_success:
        print("\n🎉 ALL TESTS PASSED! Your AutoML pipeline is working correctly!")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
