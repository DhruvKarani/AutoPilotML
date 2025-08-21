#!/usr/bin/env python3
"""
Simple script to run AutoML Pipeline from command line
"""

import pandas as pd
from automl_pipeline import run_automl_pipeline, test_pipeline
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🤖 AutoML Pipeline - Command Line Interface")
    print("="*50)
    
    # Method 1: Test with sample datasets
    print("\n1️⃣ Test with sample datasets")
    print("2️⃣ Use your own CSV file")
    choice = input("\nChoose an option (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🧪 Running test pipeline with sample datasets...")
        try:
            results = test_pipeline()
            print("✅ Test completed successfully!")
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    elif choice == "2":
        # Get user input
        csv_path = input("\n📁 Enter path to your CSV file: ").strip()
        target_col = input("🎯 Enter target column name: ").strip()
        
        # Optional parameters
        print("\n⚙️ Optional settings (press Enter to use defaults):")
        model_choice = input("Model choice (gridsearch/accuracy/utility) [gridsearch]: ").strip() or "gridsearch"
        force_clean = input("Force clean regression targets? (y/n) [n]: ").strip().lower() == 'y'
        roc_class = input("Class for ROC curve (multiclass only): ").strip() or None
        
        try:
            # Load data
            print(f"\n📊 Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            print(f"✅ Data loaded: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            if target_col not in df.columns:
                print(f"❌ Error: Target column '{target_col}' not found!")
                print(f"Available columns: {list(df.columns)}")
                return
            
            # Run pipeline
            print(f"\n🚀 Running AutoML pipeline...")
            print(f"Target: {target_col}")
            print(f"Model choice: {model_choice}")
            print("-" * 50)
            
            results = run_automl_pipeline(
                df=df,
                target_col=target_col,
                model_choice=model_choice,
                force_clean_regression=force_clean,
                selected_class_for_roc=roc_class
            )
            
            # Display results
            print("\n" + "="*50)
            print("🎉 PIPELINE COMPLETED!")
            print("="*50)
            print(f"🎯 Task Type: {results['task'].capitalize()}")
            print(f"🏆 Best Model: {results['best_model']}")
            print(f"⭐ Overall Rating: {results['overall_rating']:.1f}/10")
            print(f"💾 Model Saved: {results['model_path']}")
            
            # Show metrics
            print(f"\n📊 Performance Metrics:")
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    print(f"   • {key.upper()}: {value:.4f}")
                else:
                    print(f"   • {key.upper()}: {value}")
            
            # Show plots info
            print(f"\n📈 Generated {len(results['plots'])} visualizations")
            
            # Show logs option
            show_logs = input("\n📋 Show detailed logs? (y/n) [n]: ").strip().lower() == 'y'
            if show_logs:
                print("\n📋 DETAILED LOGS:")
                print("-" * 30)
                for log in results['logs']:
                    print(log)
            
            print(f"\n✅ Complete! Your model is saved at: {results['model_path']}")
            
        except FileNotFoundError:
            print(f"❌ Error: File '{csv_path}' not found!")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    else:
        print("❌ Invalid choice. Please run again and choose 1 or 2.")

if __name__ == "__main__":
    main()
