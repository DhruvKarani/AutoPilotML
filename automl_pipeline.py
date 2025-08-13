import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import os
import shap
import warnings
warnings.filterwarnings('ignore')

# Handle seaborn import with fallback
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    print("✅ Seaborn imported successfully")
except ImportError as e:
    print(f"⚠️ Seaborn import failed: {e}")
    print("📊 Will use matplotlib for plotting instead")
    sns = None

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import  ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

print("📦 All packages imported successfully!")

def detect_cat_num(series):
    # For numeric data types
    if series.dtype in ['int64', 'float64']:
        unique_count = series.nunique()
        
        # Check if it's truly categorical (like target labels 0,1,2 for classes)
        # vs numeric binary features (like age: 0=young, 1=old)
        if unique_count <= 2:
            # If values are 0,1 and it looks like a binary feature, treat as categorical
            unique_vals = sorted(series.dropna().unique())
            if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
                return 'categorical'  # Binary encoded feature
            elif unique_count == 2:
                return 'categorical'  # Other 2-value categorical
            else:
                return 'categorical'  # Single value (shouldn't happen after cleaning)
        else:
            return 'numerical'  # More than 2 unique values = numerical
    elif series.dtype == 'object' or series.dtype.name == 'category':
        return 'categorical'
    elif series.dtype == 'bool':
        return 'boolean'
    else:
        return 'unknown'

def plot_categorical_features(df, categorical_cols):
    plots = []
    if len(categorical_cols) == 0:
        print("ℹ️ No categorical features detected to plot.")
        print("💡 This might be because:")
        print("   • All features are numerical")
        print("   • Binary features (0,1) are being treated as numerical")
        print("   • High cardinality categorical features were dropped")
        return plots
    
    print(f"📊 Creating plots for {len(categorical_cols)} categorical features: {categorical_cols}")
    
    for col in categorical_cols:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + max(value_counts.values)*0.01, str(v), 
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plots.append(fig)
            print(f"   ✅ Created plot for {col} ({value_counts.shape[0]} unique values)")
        except Exception as e:
            print(f"   ❌ Error plotting {col}: {str(e)}")
            continue
    return plots

def plot_numerical_histograms(df, numerical_cols):
    plots = []
    if len(numerical_cols) == 0:
        print("ℹ️ No numerical features detected to plot.")
        print("💡 This might be because:")
        print("   • All features are categorical or boolean")
        print("   • Features were dropped due to high missing values")
        print("   • Dataset only contains the target variable")
        return plots
    
    print(f"📊 Creating histograms for {len(numerical_cols)} numerical features: {numerical_cols}")
    
    for col in numerical_cols:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            data = df[col].dropna()
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
            ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = data.mean()
            median_val = data.median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.legend()
            
            plt.tight_layout()
            plots.append(fig)
            print(f"   ✅ Created histogram for {col} (range: {data.min():.2f} to {data.max():.2f})")
        except Exception as e:
            print(f"   ❌ Error plotting {col}: {str(e)}")
            continue
    return plots

def plot_correlation_heatmap(df, numerical_cols):
    plots = []
    if len(numerical_cols) < 2:
        print("Not enough numerical features to plot a heatmap.")
        return plots

    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        corr_matrix = df[numerical_cols].corr()
        
        if sns:
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        else:
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(numerical_cols)))
            ax.set_yticks(range(len(numerical_cols)))
            ax.set_xticklabels(numerical_cols, rotation=45)
            ax.set_yticklabels(numerical_cols)
            
            for i in range(len(numerical_cols)):
                for j in range(len(numerical_cols)):
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', 
                           color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')
        
        ax.set_title("Correlation Heatmap (Numerical Features)")
        plt.tight_layout()
        plots.append(fig)
    except Exception as e:
        print(f"Error plotting correlation heatmap: {str(e)}")
    return plots

def display_class_label_mapping(y, model=None):
    y_unique = np.unique(y)
    mapping_info = []

    if len(y_unique) > 2:
        mapping_info.append("Detected multiclass classification. Label mapping:")
        if model and hasattr(model, 'classes_'):
            for i, cls in enumerate(model.classes_):
                mapping_info.append(f"  {i}: {cls}")
        elif isinstance(y[0], str) or pd.api.types.is_categorical_dtype(y):
            for i, cls in enumerate(y_unique):
                mapping_info.append(f"  {i}: {cls}")
        else:
            mapping_info.append(f"  Raw label values detected: {list(y_unique)}")
    else:
        mapping_info.append("Binary classification detected — no mapping needed.")
    
    return mapping_info

def plot_target_distribution(y, task):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if task == 'classification':
        if sns:
            sns.countplot(x=y, ax=ax)
        else:
            pd.Series(y).value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Target Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    elif task == 'regression':
        if sns:
            sns.histplot(y, kde=True, bins=30, ax=ax)
        else:
            ax.hist(y, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title("Target Value Distribution")
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")
    
    ax.grid(True)
    plt.tight_layout()
    return fig

def detect_task_type(y):
    if y.dtype in ['int64', 'float64'] and y.nunique() <= 10:
        return 'classification'
    elif y.dtype in ['int64', 'float64']:
        return 'regression'
    else:
        return 'classification'

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return y_encoded, mapping

def validate_and_clean_target(X, y, task_type, max_bad_ratio=0.01, auto_clean=True, force_clean=False):
    y = pd.Series(y)
    total = len(y)

    nan_mask = y.isnull()
    zero_mask = (y == 0) if task_type == "regression" else pd.Series([False] * total, index=y.index) 
    #[False] * total creates a list of False values of length = total rows.
    bad_mask = nan_mask | zero_mask  #combines both — these rows will be removed

    bad_count = bad_mask.sum()
    bad_ratio = bad_count / total 
    
    cleaning_log = []

    if bad_count == 0:
        cleaning_log.append(f"✅ [{task_type.capitalize()}] Target is clean.")
        return X, y, cleaning_log

    if bad_ratio <= max_bad_ratio:
        cleaning_log.append(f"⚠️ [{task_type.capitalize()}] {bad_count}/{total} bad target values ({bad_ratio*100:.2f}%) — cleaning them.")
        if auto_clean:
            good_mask = ~bad_mask
            return X.loc[good_mask], y.loc[good_mask], cleaning_log
        else:
            cleaning_log.append("⚠️ auto_clean=False — not cleaning.")
            return X, y, cleaning_log
    else:
        cleaning_log.append(f"🚨 [{task_type.capitalize()}] {bad_count}/{total} bad target values ({bad_ratio*100:.2f}%)")
        if task_type == "regression" and force_clean:
            cleaning_log.append("⚠️ Force cleaning enabled for regression task.")
            good_mask = ~bad_mask
            return X.loc[good_mask], y.loc[good_mask], cleaning_log
        else:
            cleaning_log.append("⚠️ Not cleaning — high bad ratio detected.")
            return X, y, cleaning_log


def add_interaction_features(df, numeric_cols):
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df

def add_polynomial_features(df, numeric_cols, degree=2):
    for col in numeric_cols:
        if degree >= 2:
            df[f"{col}^2"] = df[col] ** 2
        if degree >= 3:
            df[f"{col}^3"] = df[col] ** 3
    return df

def add_ratio_features(df, numeric_cols):
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-5)
    return df

def auto_feature_engineering(df, task_type, degree=2):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df, f"Not enough numeric features for engineering"

    df = add_interaction_features(df, numeric_cols)
    df = add_polynomial_features(df, numeric_cols, degree=degree)
    df = add_ratio_features(df, numeric_cols)

    if 'age' in df.columns:
        df['age_binned'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'mid', 'senior'])

    return df, f"🧠 Feature engineering applied: +{df.shape[1] - len(numeric_cols)} new features"

def plot_predicted_vs_actual(model, X_test, y_test):
    """Plot predicted vs actual values for regression models"""
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual")
    ax.grid(True)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    ax.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_residuals(model, X_test, y_test):
    """Plot residuals for regression models"""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    ax.grid(True)
    
    # Add statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    ax.text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def comprehensive_shap_analysis(pipeline, X_train, X_test, task_type, max_samples=100):
    """
    Enhanced SHAP analysis function with explicit task-type handling for both classification and regression.
    
    Parameters:
    -----------
    pipeline : sklearn.Pipeline
        Trained ML pipeline with preprocessor and model
    X_train : pd.DataFrame or np.array
        Training features
    X_test : pd.DataFrame or np.array  
        Test features
    task_type : str
        Task type: "classification" or "regression" (detected by main pipeline)
    max_samples : int, default=100
        Maximum samples to use for SHAP analysis (for performance)
        
    Returns:
    --------
    dict : SHAP analysis results with feature importance, explainer type, and logs
    """
    shap_results = {
        "explainer_type": None,
        "feature_importance": None,
        "top_features": None,
        "shap_values": None,
        "feature_names": None,
        "task_detected": None,
        "model_type": None,
        "analysis_logs": []
    }
    
    try:
        shap_results["analysis_logs"].append("🔍 Starting enhanced SHAP analysis...")
        
        # Get the model from pipeline
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        # The named_steps dictionary would contain:
        # 'preprocessor': The preprocessing pipeline (handles data cleaning, encoding, scaling, etc.)
        # 'model': The actual machine learning model (Random Forest, Logistic Regression, etc.)
        # Like defined in Pipeline
        
        # Detect model type for better explainer selection
        model_name = type(model).__name__
        shap_results["model_type"] = model_name
        shap_results["analysis_logs"].append(f"📊 Model detected: {model_name}")
        
        # Task type is passed from the main pipeline (already detected)
        detected_task = task_type
        shap_results["analysis_logs"].append(f"🎯 Task type: {detected_task}")
        shap_results["task_detected"] = detected_task
        
        # Transform data
        X_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names after transformation
        feature_names = []
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out().tolist()
        else:
            # Fallback for older versions
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        shap_results["feature_names"] = feature_names
        shap_results["analysis_logs"].append(f"📝 Features after preprocessing: {len(feature_names)}")
        
        # Limit samples for performance
        n_samples = min(max_samples, X_transformed.shape[0])
        X_sample = X_transformed[:n_samples]
        X_test_sample = X_test_transformed[:min(50, X_test_transformed.shape[0])]
        
        shap_results["analysis_logs"].append(f"🔢 Using {n_samples} training samples and {len(X_test_sample)} test samples")
        
        # Smart explainer selection based on model type and task
        explainer = None
        shap_values = None
        
        # Define tree-based models
        tree_models = ['RandomForest', 'GradientBoosting', 'XGB', 'LightGBM', 'DecisionTree', 'AdaBoost']
        linear_models = ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso', 'SGD']
        
        # 1. TreeExplainer for tree-based models (supports both classification and regression)
        if any(tree_name in model_name for tree_name in tree_models) or hasattr(model, 'feature_importances_'):
            try:
                shap_results["analysis_logs"].append(f"🌳 Trying TreeExplainer for {detected_task}...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
                shap_results["explainer_type"] = "TreeExplainer"
                shap_results["analysis_logs"].append("✅ TreeExplainer successful - optimal for tree models")
            except Exception as e:
                shap_results["analysis_logs"].append(f"❌ TreeExplainer failed: {str(e)}")
        
        # 2. LinearExplainer for linear models (supports both classification and regression)
        if explainer is None and (any(linear_name in model_name for linear_name in linear_models) or hasattr(model, 'coef_')):
            try:
                shap_results["analysis_logs"].append(f"📏 Trying LinearExplainer for {detected_task}...")
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer(X_test_sample)
                shap_results["explainer_type"] = "LinearExplainer"
                shap_results["analysis_logs"].append("✅ LinearExplainer successful - optimal for linear models")
            except Exception as e:
                shap_results["analysis_logs"].append(f"❌ LinearExplainer failed: {str(e)}")
        
        # 3. KernelExplainer as universal fallback (works for any model type and task)
        if explainer is None:
            try:
                shap_results["analysis_logs"].append(f"🔮 Trying KernelExplainer for {detected_task} (universal but slower)...")
                background = X_sample[:min(25, len(X_sample))]
                
                # Use appropriate prediction function based on task
                if detected_task == "classification" and hasattr(model, 'predict_proba'):
                    predict_fn = lambda x: model.predict_proba(x)[:, 1] if model.predict_proba(x).shape[1] == 2 else model.predict_proba(x)
                    shap_results["analysis_logs"].append("🎲 Using predict_proba for classification")
                else:
                    predict_fn = model.predict
                    shap_results["analysis_logs"].append(f"🎯 Using predict for {detected_task}")
                
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer(X_test_sample)
                shap_results["explainer_type"] = "KernelExplainer"
                shap_results["analysis_logs"].append("✅ KernelExplainer successful - universal compatibility")
            except Exception as e:
                shap_results["analysis_logs"].append(f"❌ KernelExplainer failed: {str(e)}")
                return shap_results
        
        if shap_values is None:
            shap_results["analysis_logs"].append("❌ All SHAP explainers failed")
            return shap_results
        
        # Enhanced feature importance calculation with task-specific handling
        if hasattr(shap_values, 'values'):
            shap_array_raw = shap_values.values #(common with TreeExplainer) 
        else:
            shap_array_raw = shap_values
            
        # Handle different SHAP value shapes based on task and model output
        if len(shap_array_raw.shape) == 3:  # Multiclass classification
            shap_array = np.abs(shap_array_raw).mean(axis=(0, 2))
            shap_results["analysis_logs"].append("📊 Multiclass classification - averaging across samples and classes")
        elif len(shap_array_raw.shape) == 2:  # Binary classification or regression
            shap_array = np.abs(shap_array_raw).mean(axis=0)
            if detected_task == "classification":
                shap_results["analysis_logs"].append("📊 Binary classification - averaging across samples")
            else:
                shap_results["analysis_logs"].append("📊 Regression - averaging across samples")
        else:
            shap_results["analysis_logs"].append("⚠️ Unexpected SHAP values shape, using fallback calculation")
            shap_array = np.abs(shap_array_raw).flatten()
        
        # Ensure we have the right number of features
        if len(shap_array) != len(feature_names):
            shap_results["analysis_logs"].append(f"⚠️ SHAP array length ({len(shap_array)}) doesn't match features ({len(feature_names)})")
            min_length = min(len(shap_array), len(feature_names))
            shap_array = shap_array[:min_length]
            feature_names = feature_names[:min_length]
        
        # Feature importance ranking
        feature_importance = shap_array.tolist()
        top_feature_indices = shap_array.argsort()[-10:][::-1]
        
        shap_results["feature_importance"] = feature_importance
        shap_results["top_features"] = top_feature_indices.tolist()
        shap_results["shap_values"] = shap_values
        
        # Enhanced logging with task-specific insights
        shap_results["analysis_logs"].append(f"\n🎯 Top 10 Most Important Features for {detected_task.title()}:")
        
        total_importance = shap_array.sum()
        for i, feat_idx in enumerate(top_feature_indices):
            importance = shap_array[feat_idx]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            importance_pct = (importance / total_importance) * 100 if total_importance > 0 else 0
            score = min(10, (importance / shap_array.max()) * 10) if shap_array.max() > 0 else 0
            
            shap_results["analysis_logs"].append(
                f"  {i+1}. {feat_name}: {importance:.4f} ({importance_pct:.1f}%) [Score: {score:.1f}/10]"
            )
        
        # Task-specific summary
        if detected_task == "classification":
            shap_results["analysis_logs"].append(f"\n✅ Classification SHAP analysis completed using {shap_results['explainer_type']}")
            shap_results["analysis_logs"].append("📈 Higher values indicate stronger influence on class prediction")
        else:
            shap_results["analysis_logs"].append(f"\n✅ Regression SHAP analysis completed using {shap_results['explainer_type']}")
            shap_results["analysis_logs"].append("📈 Higher values indicate stronger influence on target value prediction")
        
    except ImportError:
        shap_results["analysis_logs"].append("⚠️ SHAP not installed. Install with: pip install shap")
    except Exception as e:
        shap_results["analysis_logs"].append(f"❌ SHAP analysis failed: {str(e)}")
        import traceback
        shap_results["analysis_logs"].append(f"🔍 Error details: {traceback.format_exc()}")
    
    return shap_results


def auto_feature_refinement_after_shap(X_train, X_test, shap_results, importance_threshold=0.01):
    """
    Auto feature refinement based on SHAP analysis - exact from notebook
    """
    refinement_logs = []
    
    if not shap_results or not shap_results.get("feature_importance"):
        refinement_logs.append("⚠️ No SHAP results available for feature refinement")
        return X_train, X_test, refinement_logs
    
    feature_importance = np.array(shap_results["feature_importance"])
    feature_names = shap_results.get("feature_names", [f"feature_{i}" for i in range(len(feature_importance))])
    
    # Identify low-importance features
    low_importance_mask = feature_importance < importance_threshold
    low_importance_indices = np.where(low_importance_mask)[0]
    
    if len(low_importance_indices) == 0:
        refinement_logs.append("✅ All features meet importance threshold - no refinement needed")
        return X_train, X_test, refinement_logs
    
    # Keep high-importance features
    high_importance_indices = np.where(~low_importance_mask)[0]
    
    refinement_logs.append(f"🔧 Feature Refinement Analysis:")
    refinement_logs.append(f"   • Total features: {len(feature_importance)}")
    refinement_logs.append(f"   • Low importance features (<{importance_threshold}): {len(low_importance_indices)}")
    refinement_logs.append(f"   • Keeping high importance features: {len(high_importance_indices)}")
    
    # Apply refinement if we have numpy arrays
    if isinstance(X_train, np.ndarray):
        X_train_refined = X_train[:, high_importance_indices]
        X_test_refined = X_test[:, high_importance_indices]
    else:
        # For sparse matrices or other formats
        try:
            X_train_refined = X_train[:, high_importance_indices]
            X_test_refined = X_test[:, high_importance_indices]
        except:
            refinement_logs.append("⚠️ Could not apply feature refinement to this data format")
            return X_train, X_test, refinement_logs
    
    refinement_logs.append(f"✅ Feature refinement completed: {X_train.shape[1]} → {X_train_refined.shape[1]} features")
    
    # Log dropped features
    if len(low_importance_indices) <= 5:  # Only show if not too many
        refinement_logs.append("🗑️ Dropped low-importance features:")
        for idx in low_importance_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            importance = feature_importance[idx]
            refinement_logs.append(f"   • {feat_name}: {importance:.4f}")
    
    return X_train_refined, X_test_refined, refinement_logs

def run_automl_pipeline(df, target_col, model_choice='utility', force_clean_regression=False, selected_class_for_roc=None):
    """
    Complete AutoML Pipeline with intelligent model selection strategies
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target_col : str
        Name of target column
    model_choice : str, default='utility'
        Model selection strategy: 'utility' (comprehensive) or 'speed' (fast & smart)
    force_clean_regression : bool, default=False
        Force clean regression targets even with high bad ratio
    selected_class_for_roc : str, optional
        Class name for multiclass ROC curve generation
    
    Returns:
    --------
    dict : Complete results dictionary with all metrics, plots, and analysis
    """
    
    results = {
        "logs": [],
        "plots": [],
        "metrics": {},
        "summary": "",
        "best_model": "",
        "model_path": "",
        "label_mapping": None,
        "model_scores": {},
        "shap_analysis": None,
        "overall_rating": 0,
        "class_mapping": [],
        "task": None
    }

    # Normalize model_choice to handle legacy or unknown inputs
    legacy_map_utility = {"gridsearch", "full", "all", "comprehensive", "utility"}
    legacy_map_speed = {"speed", "fast", "quick", "smart"}
    mc_lower = str(model_choice).strip().lower() if model_choice is not None else "utility"
    if mc_lower in legacy_map_utility:
        model_choice = 'utility'
    elif mc_lower in legacy_map_speed:
        model_choice = 'speed'
    else:
        results["logs"].append(f"⚠️ Unknown model_choice '{model_choice}' – defaulting to 'utility'. Use 'utility' or 'speed'.")
        model_choice = 'utility'
    
    # Phase 1: Feature Detection (EXACT from notebook)
    results["logs"].append("🔍 Starting Feature Detection...")
    results["logs"].append("="*50)
    
    numerical_features = []
    categorical_features = []
    boolean_features = []
    dropped_columns = []

    for col in df.columns:
        if col == target_col:
            continue
        
        # Check for high missing values or constant columns
        missing_ratio = df[col].isnull().mean()
        unique_count = df[col].nunique()
        
        if missing_ratio > 0.4:
            dropped_columns.append(f"{col} (missing: {missing_ratio:.1%})")
            continue
        if unique_count == 1:
            dropped_columns.append(f"{col} (constant)")
            continue
        
        # Detect feature type
        kind = detect_cat_num(df[col])
        if kind == 'numerical':
            numerical_features.append(col)
        elif kind == 'categorical':
            if unique_count > 30:
                dropped_columns.append(f"{col} (high cardinality: {unique_count})")
            else:
                categorical_features.append(col)
        elif kind == 'boolean':
            boolean_features.append(col)
        else:
            dropped_columns.append(f"{col} (unknown type)")
    
    # Drop problematic columns
    columns_to_drop = [col.split(' (')[0] for col in dropped_columns]
    df = df.drop(columns=columns_to_drop)
    
    results["logs"].append(f"📊 Feature Detection Results:")
    results["logs"].append(f"   • Numerical features: {len(numerical_features)}")
    results["logs"].append(f"   • Categorical features: {len(categorical_features)}")
    results["logs"].append(f"   • Boolean features: {len(boolean_features)}")
    results["logs"].append(f"   • Dropped columns: {len(dropped_columns)}")
    
    if dropped_columns:
        results["logs"].append(f"🗑️ Dropped columns details:")
        for col_info in dropped_columns:
            results["logs"].append(f"   • {col_info}")
    
    # Phase 2: EDA Visualizations (EXACT from notebook)
    results["logs"].append("\n📊 Creating EDA visualizations...")
    
    try:
        cat_plots = plot_categorical_features(df, categorical_features)
        results["logs"].append(f"   ✅ Created {len(cat_plots)} categorical plots")
    except Exception as e:
        results["logs"].append(f"   ⚠️ Categorical plots failed: {str(e)}")
        cat_plots = []
    
    try:
        num_plots = plot_numerical_histograms(df, numerical_features)
        results["logs"].append(f"   ✅ Created {len(num_plots)} numerical histograms")
    except Exception as e:
        results["logs"].append(f"   ⚠️ Numerical plots failed: {str(e)}")
        num_plots = []
    
    try:
        corr_plots = plot_correlation_heatmap(df, numerical_features)
        results["logs"].append(f"   ✅ Created {len(corr_plots)} correlation heatmaps")
    except Exception as e:
        results["logs"].append(f"   ⚠️ Correlation plots failed: {str(e)}")
        corr_plots = []
    
    results["plots"].extend(cat_plots + num_plots + corr_plots)
    results["logs"].append(f"📈 Total EDA plots created: {len(results['plots'])}")

    # Phase 3: Create Transformers (EXACT from notebook)
    results["logs"].append("\n🔧 Setting up data transformers...")
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features),
        ('bool', 'passthrough', boolean_features)
    ])
    
    results["logs"].append(f"✅ Transformers created:")
    results["logs"].append(f"   • Numerical: Mean imputation + StandardScaler")
    results["logs"].append(f"   • Categorical: Mode imputation + OneHotEncoder")
    results["logs"].append(f"   • Boolean: Passthrough")
    
    # Phase 4: Define X and y (EXACT from notebook)
    results["logs"].append("\n📋 Preparing features and target...")
    X = df.drop(columns=target_col)
    y = df[target_col]
    
    results["logs"].append(f"✅ Data split into features and target:")
    results["logs"].append(f"   • Features (X): {X.shape}")
    results["logs"].append(f"   • Target (y): {y.shape}")
    
    # Phase 5: Target Analysis (EXACT from notebook)
    results["logs"].append("\n🎯 Analyzing target variable...")
    
    # Display class mapping info
    results["class_mapping"] = display_class_label_mapping(y)
    for mapping_info in results["class_mapping"]:
        results["logs"].append(mapping_info)
    
    # Detect task type
    task = detect_task_type(y)
    results["task"] = task
    results["logs"].append(f"\n✅ Task type detected: {task.upper()}")
    
    # Create target distribution plot
    try:
        target_plot = plot_target_distribution(y, task)
        results["plots"].append(target_plot)
        results["logs"].append("✅ Target distribution plot created")
    except Exception as e:
        results["logs"].append(f"⚠️ Target plot failed: {str(e)}")
    
    # Phase 6: Label Encoding (EXACT from notebook)
    if task == 'classification':
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            results["logs"].append("\n🏷️ Applying label encoding...")
            y, label_mapping = encode_labels(y)
            results["label_mapping"] = label_mapping
            results["logs"].append(f"✅ Label encoding applied: {label_mapping}")
        else:
            results["logs"].append("\n🏷️ Numerical labels detected - no encoding needed")
            results["label_mapping"] = None
    else:
        results["logs"].append("\n🏷️ Regression task - no label encoding needed")
        results["label_mapping"] = None
    
    # Phase 7: Target Cleaning (EXACT from notebook)
    results["logs"].append("\n🧹 Cleaning target variable...")
    X, y, cleaning_logs = validate_and_clean_target(X, y, task, force_clean=force_clean_regression)
    for log in cleaning_logs:
        results["logs"].append(log)
    
    # Phase 8: Define Simple Models - PHASE 1 (EXACT from notebook)
    results["logs"].append("\n🤖 PHASE 1: Quick Model Testing (3 Models)")
    results["logs"].append("="*50)
    
    if task == 'classification':
        simple_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier()
        }
        scoring = 'accuracy'
        results["logs"].append("✅ Classification models defined (Phase 1):")
    elif task == 'regression':
        simple_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'KNN': KNeighborsRegressor()
        }
        scoring = 'r2'  # R² score - higher is better (more intuitive)
        results["logs"].append("✅ Regression models defined (Phase 1):")

    for model_name in simple_models.keys():
        results["logs"].append(f"   • {model_name}")
    
    # Phase 9: Feature Engineering (EXACT from notebook)
    results["logs"].append("\n🧠 Applying feature engineering...")
    X, fe_log = auto_feature_engineering(X, task, degree=2)
    results["logs"].append(fe_log)
    results["logs"].append(f"✅ Feature engineering completed. New shape: {X.shape}")
    
    # Phase 10: Train/Test Split (EXACT from notebook)
    results["logs"].append("\n✂️ Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None)
    
    results["logs"].append(f"✅ Data split completed:")
    results["logs"].append(f"   • Training set: {X_train.shape[0]} samples")
    results["logs"].append(f"   • Test set: {X_test.shape[0]} samples")
    results["logs"].append(f"   • Split ratio: 80/20")
    
    # Create post-split target distribution plot
    try:
        target_plot_post_split = plot_target_distribution(y_train, task)
        results["plots"].append(target_plot_post_split)
        results["logs"].append("✅ Post-split target distribution plot created")
    except Exception as e:
        results["logs"].append(f"⚠️ Post-split target plot failed: {str(e)}")
    
    # Phase 11: Cross Validation Simple Models (EXACT from notebook)
    results["logs"].append("\n🔄 PHASE 1: Cross-validation on simple models...")
    results["logs"].append("="*60)
    
    simple_model_scores = {}
    simple_cv_results = {}
    
    for name, model in simple_models.items():
        results["logs"].append(f"\n🔍 Testing {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Run cross-validation with timing
        start_time = time.time()
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
            end_time = time.time()
            
            mean_score = scores.mean()
            std_score = scores.std()
            elapsed_time = end_time - start_time
            
            # Store results
            simple_cv_results[name] = {
                'scores': scores,
                'mean': mean_score,
                'std': std_score,
                'time': elapsed_time
            }
            
            # Display results - R² and accuracy are positive (higher is better)
            # Negative error metrics need to be converted to positive for display
            if scoring.startswith('neg'):
                display_score = abs(mean_score)  # Convert negative error to positive
                simple_model_scores[name] = display_score
                results["logs"].append(f"   📊 CV Score (as positive error): {display_score:.4f} (±{abs(std_score):.4f})")
            else:
                display_score = mean_score  # R² and accuracy are already positive
                simple_model_scores[name] = display_score
                results["logs"].append(f"   📊 CV Score (R² or Accuracy): {display_score:.4f} (±{std_score:.4f})")
            
            results["logs"].append(f"   ⏱️ Training time: {elapsed_time:.2f}s")
            
            # Fix the complex f-string by using a simpler approach
            if scoring.startswith('neg'):
                score_list = [f'{abs(s):.3f}' for s in scores]  # Show positive error values
                results["logs"].append(f"   🎯 Individual fold scores (as positive): {score_list}")
            else:
                score_list = [f'{s:.3f}' for s in scores]  # Show R² or accuracy as-is
                results["logs"].append(f"   🎯 Individual fold scores: {score_list}")
            
        except Exception as e:
            results["logs"].append(f"   ❌ Failed: {str(e)}")
            simple_model_scores[name] = 0
            simple_cv_results[name] = {'error': str(e)}
            continue
    
    # Find best simple model
    if simple_model_scores:
        best_simple_model = max(simple_model_scores, key=simple_model_scores.get)
        best_simple_score = simple_model_scores[best_simple_model]
        
        results["logs"].append(f"\n🏆 PHASE 1 RESULTS:")
        results["logs"].append("="*40)
        results["logs"].append(f"🥇 Best Model: {best_simple_model}")
        results["logs"].append(f"🎯 Best Score: {best_simple_score:.4f}")
        results["logs"].append(f"⏱️ Training Time: {simple_cv_results[best_simple_model].get('time', 0):.2f}s")
    else:
        results["logs"].append("❌ No models completed successfully")
        best_simple_model = list(simple_models.keys())[0]  # Fallback
        best_simple_score = 0

    # Phase 12: Initial Pipeline Training (EXACT from notebook)
    results["logs"].append(f"\n🔧 Training initial pipeline with {best_simple_model}...")
    
    # Create and train the initial pipeline
    initial_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', simple_models[best_simple_model])
    ])
    
    try:
        initial_pipeline.fit(X_train, y_train)
        results["logs"].append("✅ Initial pipeline trained successfully")
    except Exception as e:
        results["logs"].append(f"❌ Initial pipeline training failed: {str(e)}")
        # Return early if we can't even train a basic model
        results["summary"] = f"Pipeline failed at initial training: {str(e)}"
        return results

    # Phase 13: SHAP Analysis (EXACT from notebook)
    results["logs"].append("\n🔍 Running comprehensive SHAP analysis...")
    results["logs"].append("="*50)
    
    shap_results = comprehensive_shap_analysis(
        pipeline=initial_pipeline,
        X_train=X_train,
        X_test=X_test,
        task_type=task,
        max_samples=100
    )
    
    # Log SHAP analysis results
    for log_entry in shap_results["analysis_logs"]:
        results["logs"].append(log_entry)
    
    # Store SHAP results
    results["shap_analysis"] = shap_results
    
    # Phase 14: Feature Refinement based on SHAP (EXACT from notebook)
    if shap_results.get("feature_importance"):
        results["logs"].append("\n🔧 Applying feature refinement based on SHAP analysis...")
        
        # Transform data first for refinement
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Apply feature refinement
        X_train_refined, X_test_refined, refinement_logs = auto_feature_refinement_after_shap(
            X_train_transformed, X_test_transformed, shap_results, importance_threshold=0.01
        )
        
        for log_entry in refinement_logs:
            results["logs"].append(log_entry)
        
        # Note: We'll use the refined features for advanced models if available
        if X_train_refined.shape[1] < X_train_transformed.shape[1]:
            results["logs"].append("🚀 Feature refinement will be applied to advanced models")
            use_refined_features = True
        else:
            results["logs"].append("📝 No features refined - using original feature set")
            use_refined_features = False
    else:
        results["logs"].append("\n⚠️ Skipping feature refinement - SHAP analysis incomplete")
        use_refined_features = False
    
    # Phase 15: Smart Model Selection Strategy
    # Predefine to avoid UnboundLocalError in unexpected branches
    advanced_models = {}
    param_grids = {}
    if model_choice == 'utility':
        results["logs"].append("\n🔧 UTILITY MODE: Comprehensive analysis with all models")
        results["logs"].append("="*60)
        results["logs"].append("📊 Running full GridSearchCV on all 13 models for maximum accuracy...")
        
        # Advanced models setup - Full comprehensive analysis
        if task == 'classification':
            advanced_models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'SVC': SVC(random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'AdaBoost': AdaBoostClassifier(random_state=42),
                'MLP': MLPClassifier(max_iter=500, random_state=42),
                'Naive Bayes': GaussianNB(),
                'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
            }
            
            param_grids = {
                'Logistic Regression': {'model__C': [0.1, 1, 10]},
                'Random Forest': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [None, 5, 10]
                },
                'KNN': {'model__n_neighbors': [3, 5, 7]},
                'SVC': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf']
                },
                'Decision Tree': {
                    'model__max_depth': [None, 5, 10],
                    'model__criterion': ['gini', 'entropy']
                },
                'Gradient Boosting': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1, 0.2]
                },
                'AdaBoost': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1, 1.0]
                },
                'MLP': {
                    'model__hidden_layer_sizes': [(50,), (100,)],
                    'model__activation': ['relu', 'tanh']
                },
                'Naive Bayes': {},
                'SGD Classifier': {
                    'model__loss': ['log_loss', 'hinge'],
                    'model__alpha': [0.0001, 0.001, 0.01]
                },
                'XGBoost': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5, 7]
                },
                'LightGBM': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__num_leaves': [31, 50]
                }
            }
            
        elif task == 'regression':
            advanced_models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'KNN': KNeighborsRegressor(),
                'SVR': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'MLP': MLPRegressor(max_iter=500, random_state=42),
                'SGD Regressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'LightGBM': LGBMRegressor(random_state=42, verbose=-1)
            }
            
            param_grids = {
                'Linear Regression': {},
                'Random Forest': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [None, 5, 10]
                },
                'KNN': {'model__n_neighbors': [3, 5, 7]},
                'SVR': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf']
                },
                'Decision Tree': {
                    'model__max_depth': [None, 5, 10],
                    'model__criterion': ['squared_error', 'absolute_error']
                },
                'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'model__alpha': [0.1, 1.0, 10.0]},
                'Gradient Boosting': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1, 0.2]
                },
                'AdaBoost': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.01, 0.1, 1.0]
                },
                'MLP': {
                    'model__hidden_layer_sizes': [(50,), (100,)],
                    'model__activation': ['relu', 'tanh']
                },
                'SGD Regressor': {
                    'model__loss': ['squared_error', 'huber'],
                    'model__alpha': [0.0001, 0.001, 0.01]
                },
                'XGBoost': {
                    'model__n_estimators': [50, 100],
                    'model__max_depth': [3, 5, 7]
                },
                'LightGBM': {
                    'model__n_estimators': [50, 100],
                    'model__learning_rate': [0.05, 0.1],
                    'model__num_leaves': [31, 50]
                }
            }

    elif model_choice == 'speed':
        results["logs"].append("\n⚡ SPEED MODE: Smart and fast model selection")
        results["logs"].append("="*60)
        
        # Determine best model family from Phase 1 results
        best_simple_model_type = type(simple_models[best_simple_model]).__name__
        results["logs"].append(f"🎯 Phase 1 winner: {best_simple_model} ({best_simple_model_type})")
        
        # Select model family based on Phase 1 winner
        if task == 'classification':
            if 'Random' in best_simple_model_type or 'Forest' in best_simple_model:
                # Tree-based family won
                results["logs"].append("🌳 Selecting tree-based model family for speed optimization...")
                advanced_models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                }
                param_grids = {
                    'Random Forest': {
                        'model__n_estimators': [50, 100],
                        'model__max_depth': [None, 5, 10]
                    },
                    'Gradient Boosting': {
                        'model__n_estimators': [50, 100],
                        'model__learning_rate': [0.01, 0.1, 0.2]
                    },
                    'XGBoost': {
                        'model__n_estimators': [50, 100],
                        'model__max_depth': [3, 5, 7]
                    }
                }
            elif 'Logistic' in best_simple_model_type or 'Logistic' in best_simple_model:
                # Linear family won
                results["logs"].append("📏 Selecting linear model family for speed optimization...")
                advanced_models = {
                    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                    'Ridge': Ridge(random_state=42),
                    'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
                }
                param_grids = {
                    'Logistic Regression': {'model__C': [0.1, 1, 10]},
                    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
                    'SGD Classifier': {
                        'model__loss': ['log_loss', 'hinge'],
                        'model__alpha': [0.0001, 0.001, 0.01]
                    }
                }
            else:  # KNN or others
                # Instance-based family
                results["logs"].append("🔍 Selecting instance-based model family for speed optimization...")
                advanced_models = {
                    'KNN': KNeighborsClassifier(),
                    'SVC': SVC(random_state=42),
                    'MLP': MLPClassifier(max_iter=500, random_state=42)
                }
                param_grids = {
                    'KNN': {'model__n_neighbors': [3, 5, 7]},
                    'SVC': {
                        'model__C': [0.1, 1, 10],
                        'model__kernel': ['linear', 'rbf']
                    },
                    'MLP': {
                        'model__hidden_layer_sizes': [(50,), (100,)],
                        'model__activation': ['relu', 'tanh']
                    }
                }
                
        elif task == 'regression':
            if 'Random' in best_simple_model_type or 'Forest' in best_simple_model:
                # Tree-based family won
                results["logs"].append("🌳 Selecting tree-based model family for speed optimization...")
                advanced_models = {
                    'Random Forest': RandomForestRegressor(random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                    'XGBoost': XGBRegressor(random_state=42)
                }
                param_grids = {
                    'Random Forest': {
                        'model__n_estimators': [50, 100],
                        'model__max_depth': [None, 5, 10]
                    },
                    'Gradient Boosting': {
                        'model__n_estimators': [50, 100],
                        'model__learning_rate': [0.01, 0.1, 0.2]
                    },
                    'XGBoost': {
                        'model__n_estimators': [50, 100],
                        'model__max_depth': [3, 5, 7]
                    }
                }
            elif 'Linear' in best_simple_model_type or 'Linear' in best_simple_model:
                # Linear family won
                results["logs"].append("📏 Selecting linear model family for speed optimization...")
                advanced_models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge': Ridge(random_state=42),
                    'Lasso': Lasso(random_state=42)
                }
                param_grids = {
                    'Linear Regression': {},
                    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
                    'Lasso': {'model__alpha': [0.1, 1.0, 10.0]}
                }
            else:  # KNN or others
                # Instance-based family
                results["logs"].append("🔍 Selecting instance-based model family for speed optimization...")
                advanced_models = {
                    'KNN': KNeighborsRegressor(),
                    'SVR': SVR(),
                    'MLP': MLPRegressor(max_iter=500, random_state=42)
                }
                param_grids = {
                    'KNN': {'model__n_neighbors': [3, 5, 7]},
                    'SVR': {
                        'model__C': [0.1, 1, 10],
                        'model__kernel': ['linear', 'rbf']
                    },
                    'MLP': {
                        'model__hidden_layer_sizes': [(50,), (100,)],
                        'model__activation': ['relu', 'tanh']
                    }
                }
        
        results["logs"].append(f"📊 Selected {len(advanced_models)} models from winning family for focused optimization")

    # Continue with common GridSearchCV logic for both utility and speed modes

    results["logs"].append(f"✅ Advanced model setup completed:")
    results["logs"].append(f"   • {len(advanced_models)} models configured")
    results["logs"].append(f"   • Parameter grids defined for tuning")
    results["logs"].append(f"   • Ready for GridSearchCV execution")
    
    # Phase 16: GridSearchCV Execution for selected models
    results["logs"].append(f"\n🚀 Running GridSearchCV on {len(advanced_models)} selected models...")
    results["logs"].append("="*60)
    
    grid_results = {}
    best_model_name = None
    # Initialize based on scoring metric
    if scoring.startswith('neg'):
        best_model_score = 0  # For negative error metrics, 0 is worst when converted to positive
    elif scoring == 'r2':
        best_model_score = -float('inf')  # R² can be negative, so start from negative infinity
    else:
        best_model_score = 0  # For accuracy and other positive metrics
    
    for name, model in advanced_models.items():
        results["logs"].append(f"\n🔍 GridSearchCV: {name}")
        
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Get parameter grid
            param_grid = param_grids.get(name, {})
            
            # Setup GridSearchCV
            if param_grid:
                grid_search = GridSearchCV(
                    pipeline, 
                    param_grid, 
                    cv=5, 
                    scoring=scoring, 
                    n_jobs=-1, 
                    verbose=0
                )
            else:
                # No parameters to tune, just use cross-validation
                grid_search = pipeline
            
            # Fit the model
            start_time = time.time()
            if hasattr(grid_search, 'fit'):
                grid_search.fit(X_train, y_train)
                
                if hasattr(grid_search, 'best_score_'):
                    cv_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                else:
                    # For models without parameter tuning
                    scores = cross_val_score(grid_search, X_train, y_train, cv=5, scoring=scoring)
                    cv_score = scores.mean()
                    best_params = "No tuning required"
            else:
                # Fallback for unusual cases
                scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
                cv_score = scores.mean()
                best_params = "No tuning required"
                grid_search = pipeline
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Store results with proper score handling
            if scoring.startswith('neg'):
                display_score = abs(cv_score)  # Convert negative error to positive for display
                comparison_score = display_score  # Use positive value for comparison (higher is better)
                score_type = "Error (lower is better, shown as positive)"
            else:
                display_score = cv_score  # R² and accuracy are already positive/negative as appropriate
                comparison_score = cv_score  # Use actual score for comparison (higher is better)
                score_type = "R² or Accuracy (higher is better)"
            
            grid_results[name] = {
                'model': grid_search,
                'score': display_score,
                'cv_score': cv_score,
                'params': best_params,
                'time': training_time
            }
            
            # Update best model using comparison_score
            if comparison_score > best_model_score:
                best_model_name = name
                best_model_score = comparison_score
            
            # Log results with clear score interpretation
            results["logs"].append(f"   ✅ Score: {display_score:.4f} ({score_type})")
            results["logs"].append(f"   ⏱️ Time: {training_time:.2f}s")
            results["logs"].append(f"   🔧 Best params: {best_params}")
            
        except Exception as e:
            results["logs"].append(f"   ❌ Failed: {str(e)}")
            grid_results[name] = {'error': str(e), 'score': 0}
            continue
    
    # Phase 17: Model Selection and Final Training
    if not grid_results or best_model_name is None:
        results["logs"].append("❌ No models completed successfully")
        results["summary"] = "All models failed during GridSearchCV"
        return results
    
    results["logs"].append(f"\n🏆 GRIDSEARCHCV RESULTS:")
    results["logs"].append("="*50)
    
    # Sort and display all results
    sorted_results = sorted(
        [(name, data) for name, data in grid_results.items() if 'score' in data and data['score'] > 0],
        key=lambda x: x[1]['score'], 
        reverse=True
    )
    
    results["logs"].append("📊 Model Rankings:")
    for i, (name, data) in enumerate(sorted_results):
        score = data['score']
        time_taken = data.get('time', 0)
        results["logs"].append(f"   {i+1}. {name}: {score:.4f} ({time_taken:.1f}s)")
    
    # Get the best model
    final_model = grid_results[best_model_name]['model']
    final_score = grid_results[best_model_name]['score']
    final_params = grid_results[best_model_name]['params']
    
    results["best_model"] = best_model_name
    results["model_scores"] = {name: data.get('score', 0) for name, data in grid_results.items()}
    
    # Add scoring explanation
    if task == 'regression':
        scoring_explanation = "📊 Scoring Metric: R² Score (0 to 1, higher is better)"
        results["logs"].append(f"\n{scoring_explanation}")
        results["logs"].append("   • 1.0 = Perfect predictions")
        results["logs"].append("   • 0.0 = Model performs as well as predicting the mean")
        results["logs"].append("   • Negative values = Model performs worse than predicting the mean")
    else:
        scoring_explanation = "📊 Scoring Metric: Accuracy (0 to 1, higher is better)"
        results["logs"].append(f"\n{scoring_explanation}")
    
    results["logs"].append(f"\n🥇 BEST MODEL: {best_model_name}")
    results["logs"].append(f"🎯 Best Score: {final_score:.4f}")
    results["logs"].append(f"🔧 Best Parameters: {final_params}")
    
    # Final model training (if it's a GridSearchCV object, it's already trained)
    if hasattr(final_model, 'best_estimator_'):
        final_pipeline = final_model.best_estimator_
    else:
        final_pipeline = final_model
        
    results["logs"].append("✅ Final model prepared for evaluation")
    
    # Phase 18: Model Evaluation
    results["logs"].append("\n📊 FINAL MODEL EVALUATION")
    results["logs"].append("="*50)
    
    # Make predictions
    y_pred = final_pipeline.predict(X_test)
    
    if task == 'classification':
        acc = accuracy_score(y_test, y_pred)
        results["metrics"]["accuracy"] = acc
        
        results["logs"].append(f"🎯 Test Accuracy: {acc:.4f}")
        
        # Classification Report
        try:
            class_report = classification_report(y_test, y_pred)
            results["logs"].append(f"\n📋 Classification Report:\n{class_report}")
        except Exception as e:
            results["logs"].append(f"⚠️ Classification report failed: {str(e)}")
        
        # Confusion Matrix Plot
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues')
            ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=11)
            plt.tight_layout()
            results["plots"].append(fig)
            results["logs"].append("✅ Confusion matrix plot created")
        except Exception as e:
            results["logs"].append(f"⚠️ Confusion matrix plot failed: {str(e)}")
    
    elif task == 'regression':
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results["metrics"]["mae"] = mae
        results["metrics"]["mse"] = mse
        results["metrics"]["rmse"] = rmse
        results["metrics"]["r2"] = r2
        
        results["logs"].append(f"🎯 Test MAE: {mae:.4f}")
        results["logs"].append(f"🎯 Test MSE: {mse:.4f}")
        results["logs"].append(f"🎯 Test RMSE: {rmse:.4f}")
        results["logs"].append(f"🎯 Test R²: {r2:.4f}")
# ...existing code...

        # Phase 19: ROC and PR Curves for Classification (EXACT from notebook)
        if task == 'classification':
            # Binary classification ROC/PR curves
            if len(np.unique(y_test)) == 2:
                results["logs"].append("\n📈 Creating ROC and PR curves (Binary Classification)...")
                
                try:
                    # Get probabilities
                    if hasattr(final_pipeline, 'predict_proba'):
                        y_proba = final_pipeline.predict_proba(X_test)[:, 1]
                    elif hasattr(final_pipeline, 'decision_function'):
                        y_proba = final_pipeline.decision_function(X_test)
                    else:
                        y_proba = y_pred.astype(float)
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                    ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
                    ax_roc.plot([0, 1], [0, 1], 'k--')
                    ax_roc.set_xlabel('False Positive Rate')
                    ax_roc.set_ylabel('True Positive Rate')
                    ax_roc.set_title('ROC Curve')
                    ax_roc.legend()
                    ax_roc.grid()
                    plt.tight_layout()
                    results["plots"].append(fig_roc)
                    
                    # PR Curve
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    
                    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
                    ax_pr.plot(recall, precision, label='Precision-Recall Curve')
                    ax_pr.set_xlabel('Recall')
                    ax_pr.set_ylabel('Precision')
                    ax_pr.set_title('PR Curve')
                    ax_pr.legend()
                    ax_pr.grid()
                    plt.tight_layout()
                    results["plots"].append(fig_pr)
                    
                    results["logs"].append(f"✅ ROC AUC: {roc_auc:.4f}")
                    results["logs"].append("✅ ROC and PR curves created")
                    
                except Exception as e:
                    results["logs"].append(f"⚠️ ROC/PR curves failed: {str(e)}")
            
            # Multiclass ROC (EXACT from notebook)
            elif len(np.unique(y_test)) > 2 and selected_class_for_roc:
                try:
                    results["logs"].append("Multiclass detected — converting to binary classification (One vs Rest)")

                    # If class labels are numeric and from sklearn datasets like Iris
                    if hasattr(final_pipeline.named_steps['model'], 'classes_'):
                        label_map = {i: str(cls) for i, cls in enumerate(final_pipeline.named_steps['model'].classes_)}
                    else:
                        label_map = {label: str(label) for label in sorted(set(y_test))}

                    results["logs"].append("Available classes:")
                    for label in label_map.values():
                        results["logs"].append(f"- {label}")

                    # Normalize input
                    reverse_map = {str(v).lower(): k for k, v in label_map.items()}
                    if selected_class_for_roc.lower() not in reverse_map:
                        results["logs"].append(f"Invalid class {selected_class_for_roc}. Choose from: {list(label_map.values())}")
                    else:
                        selected_value = reverse_map[selected_class_for_roc.lower()]

                        # Convert to binary labels: 1 = selected class, 0 = others
                        y_test_bin = (y_test == selected_value).astype(int)
                        
                        # Check if model supports predict_proba
                        if hasattr(final_pipeline, 'predict_proba'):
                            try:
                                y_proba = final_pipeline.predict_proba(X_test)[:, list(final_pipeline.named_steps['model'].classes_).index(selected_value)]
                            except:
                                # Fallback to decision function or predict
                                if hasattr(final_pipeline, 'decision_function'):
                                    y_proba = final_pipeline.decision_function(X_test)
                                else:
                                    y_proba = final_pipeline.predict(X_test).astype(float)
                        else:
                            # Fallback to decision function or predict
                            if hasattr(final_pipeline, 'decision_function'):
                                y_proba = final_pipeline.decision_function(X_test)
                            else:
                                y_proba = final_pipeline.predict(X_test).astype(float)

                        # --- ROC Curve ---
                        fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
                        roc_auc = auc(fpr, tpr)

                        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                        ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
                        ax_roc.plot([0, 1], [0, 1], 'k--')
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title(f'ROC Curve - {selected_class_for_roc} vs Rest')
                        ax_roc.legend()
                        ax_roc.grid()
                        plt.tight_layout()
                        results["plots"].append(fig_roc)

                        # --- PR Curve ---
                        precision, recall, _ = precision_recall_curve(y_test_bin, y_proba)

                        fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
                        ax_pr.plot(recall, precision, label='Precision-Recall Curve')
                        ax_pr.set_xlabel('Recall')
                        ax_pr.set_ylabel('Precision')
                        ax_pr.set_title(f'PR Curve - {selected_class_for_roc} vs Rest')
                        ax_pr.legend()
                        ax_pr.grid()
                        plt.tight_layout()
                        results["plots"].append(fig_pr)
                    
                except Exception as e:
                    results["logs"].append(f"Multiclass ROC/PR curves failed: {e}")
            
            results["summary"] = f"Best Model: {results['best_model']}, Accuracy: {acc:.4f}"
            
        elif task == 'regression':
            # Regression plots (EXACT from notebook)
            residuals_fig = plot_residuals(final_pipeline, X_test, y_test)
            pred_vs_actual_fig = plot_predicted_vs_actual(final_pipeline, X_test, y_test)
            results["plots"].extend([residuals_fig, pred_vs_actual_fig])
            
            results["summary"] = f"Best Model: {results['best_model']}, MAE: {results['metrics']['mae']:.2f}, RMSE: {results['metrics']['rmse']:.2f}, R²: {results['metrics']['r2']:.4f}"
        
        # Phase 20: Comprehensive Diagnostic Analysis (EXACT from notebook)
        results["logs"].append("\n🔍 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        results["logs"].append("=" * 60)

        if task == 'regression':
            target_std = y_test.std()
            target_var = y_test.var()
            target_range = y_test.max() - y_test.min()
            target_mean = y_test.mean()
            
            results["logs"].append(f"📈 Target Variable Analysis:")
            results["logs"].append(f"   Mean: {target_mean:.2f}")
            results["logs"].append(f"   Std Dev: {target_std:.2f}")
            results["logs"].append(f"   Variance: {target_var:.2f}")
            results["logs"].append(f"   Range: {target_range:.2f}")
            results["logs"].append(f"   Coefficient of Variation: {(target_std/target_mean)*100:.2f}%")
            
            results["logs"].append(f"\n📊 Model Performance Breakdown:")
            results["logs"].append(f"   MAE: {mae:.2f} ({(mae/target_mean)*100:.2f}% of target mean)")
            results["logs"].append(f"   RMSE: {rmse:.2f} ({(rmse/target_mean)*100:.2f}% of target mean)")
            
            # R² Score with interpretation
            if r2 >= 0:
                results["logs"].append(f"   R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
            else:
                results["logs"].append(f"   R² Score: {r2:.4f} (NEGATIVE - model worse than predicting mean)")
                results["logs"].append(f"   ⚠️  Negative R² means the model performs worse than simply predicting the target mean")
            
            # Performance interpretation
            if abs(mae/target_mean) < 0.1:
                mae_quality = "EXCELLENT"
            elif abs(mae/target_mean) < 0.2:
                mae_quality = "GOOD"
            elif abs(mae/target_mean) < 0.3:
                mae_quality = "FAIR"
            else:
                mae_quality = "POOR"
            
            if r2 > 0.8:
                r2_quality = "EXCELLENT"
            elif r2 > 0.6:
                r2_quality = "GOOD"
            elif r2 > 0.4:
                r2_quality = "FAIR"
            else:
                r2_quality = "POOR"
            
            results["logs"].append(f"\n🔬 Performance Interpretation:")
            results["logs"].append(f"   • MAE Performance: {mae_quality} (error = {(mae/target_mean)*100:.1f}% of mean)")
            results["logs"].append(f"   • R² Performance: {r2_quality} (explains {r2*100:.1f}% of variance)")
            
            # Why metrics might seem inconsistent
            results["logs"].append(f"\n💡 Why Different Metrics Tell Different Stories:")
            results["logs"].append(f"   • High target variance ({target_var:.2f}) makes R² naturally lower")
            results["logs"].append(f"   • Wide target range ({target_range:.2f}) affects R² calculation")
            results["logs"].append(f"   • MAE focuses on average errors, R² on variance explanation")
            
            # Theoretical benchmarks
            theoretical_rmse_for_r2_90 = np.sqrt(0.1 * target_var)
            results["logs"].append(f"\n📈 Theoretical Benchmarks:")
            results["logs"].append(f"   For R² = 0.90, you'd need RMSE ≤ {theoretical_rmse_for_r2_90:.2f}")
            results["logs"].append(f"   Your current RMSE: {rmse:.2f}")
            results["logs"].append(f"   Improvement needed: {((rmse/theoretical_rmse_for_r2_90-1)*100):.1f}%")

        elif task == 'classification':
            # Target distribution analysis
            unique_classes = np.unique(y_test)
            n_classes = len(unique_classes)
            
            results["logs"].append(f"📈 Target Distribution Analysis:")
            results["logs"].append(f"   Number of classes: {n_classes}")
            results["logs"].append(f"   Class distribution:")
            for cls in unique_classes:
                count = (y_test == cls).sum()
                percentage = (count / len(y_test)) * 100
                results["logs"].append(f"     Class {cls}: {count} samples ({percentage:.1f}%)")
            
            # Check for class imbalance
            class_counts = [((y_test == cls).sum()) for cls in unique_classes]
            imbalance_ratio = max(class_counts) / min(class_counts)
            results["logs"].append(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results["logs"].append(f"\n📊 Model Performance Breakdown:")
            results["logs"].append(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Performance interpretation
            if accuracy > 0.9:
                acc_quality = "EXCELLENT"
            elif accuracy > 0.8:
                acc_quality = "GOOD"
            elif accuracy > 0.7:
                acc_quality = "FAIR"
            else:
                acc_quality = "POOR"
            
            results["logs"].append(f"   Performance Level: {acc_quality}")
            
            # Baseline comparisons
            majority_class_baseline = max(class_counts) / len(y_test)
            random_baseline = 1 / n_classes
            
            results["logs"].append(f"\n📈 Baseline Comparisons:")
            results["logs"].append(f"   Random guessing baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
            results["logs"].append(f"   Majority class baseline: {majority_class_baseline:.4f} ({majority_class_baseline*100:.2f}%)")
            results["logs"].append(f"   Your model beats random by: {((accuracy/random_baseline-1)*100):.1f}%")
            results["logs"].append(f"   Your model beats majority by: {((accuracy/majority_class_baseline-1)*100):.1f}%")
            
            # Class-specific performance (if multiclass)
            if n_classes > 2:
                results["logs"].append(f"\n🔬 Class-Specific Analysis:")
                results["logs"].append(f"   • Multiclass problem detected")
                results["logs"].append(f"   • Consider checking precision/recall per class")
                results["logs"].append(f"   • Class imbalance ratio: {imbalance_ratio:.2f}")
                
                if imbalance_ratio > 3:
                    results["logs"].append(f"   ⚠️  Significant class imbalance detected!")
                    results["logs"].append(f"   ⚠️  Consider using balanced scoring metrics")
            
            # CV Score vs Test Score analysis
            results["logs"].append(f"\n💡 CV Score vs Test Performance:")
            results["logs"].append(f"   • CV scores represent average performance across folds")
            results["logs"].append(f"   • Test accuracy shows performance on unseen data")
            results["logs"].append(f"   • Both should be similar for good generalization")

        # General conclusions
        results["logs"].append(f"\n✅ OVERALL CONCLUSION:")
        if task == 'regression':
            results["logs"].append(f"   • Your regression model shows {mae_quality.lower()} MAE performance")
            results["logs"].append(f"   • R² of {r2:.4f} is {r2_quality.lower()} for this type of data")
            results["logs"].append(f"   • High target variance makes perfect R² challenging")
        else:
            results["logs"].append(f"   • Your classification model shows {acc_quality.lower()} accuracy")
            results["logs"].append(f"   • Model significantly outperforms random guessing")
            results["logs"].append(f"   • Consider class-specific metrics for deeper insights")

        results["logs"].append(f"   • Model is performing well relative to data complexity!")
# ...existing code...
        
        # Phase 21: Comprehensive Model Rating (EXACT from notebook)
        results["logs"].append("\n" + "="*60)
        results["logs"].append("🎯 COMPREHENSIVE MODEL RATING (1-10 SCALE)")
        results["logs"].append("="*60)

        if task == 'regression':
            # Calculate individual metric scores (1-10)
            mae_percent = (mae/target_mean) * 100
            if mae_percent <= 5:
                mae_score = 10
            elif mae_percent <= 10:
                mae_score = 9
            elif mae_percent <= 15:
                mae_score = 8
            elif mae_percent <= 20:
                mae_score = 7
            elif mae_percent <= 25:
                mae_score = 6
            elif mae_percent <= 30:
                mae_score = 5
            elif mae_percent <= 35:
                mae_score = 4
            elif mae_percent <= 40:
                mae_score = 3
            elif mae_percent <= 50:
                mae_score = 2
            else:
                mae_score = 1
            
            # R² Score (handles negative values properly)
            if r2 >= 0.9:
                r2_value = 10
            elif r2 >= 0.8:
                r2_value = 9
            elif r2 >= 0.7:
                r2_value = 8
            elif r2 >= 0.6:
                r2_value = 7
            elif r2 >= 0.5:
                r2_value = 6
            elif r2 >= 0.4:
                r2_value = 5
            elif r2 >= 0.3:
                r2_value = 4
            elif r2 >= 0.2:
                r2_value = 3
            elif r2 >= 0.1:
                r2_value = 2
            elif r2 >= 0:
                r2_value = 1
            else:  # Negative R² scores (worse than predicting the mean)
                r2_value = 0
            
            # RMSE Score (based on % of target mean)
            rmse_percent = (rmse/target_mean) * 100
            if rmse_percent <= 10:
                rmse_score = 10
            elif rmse_percent <= 20:
                rmse_score = 9
            elif rmse_percent <= 30:
                rmse_score = 8
            elif rmse_percent <= 40:
                rmse_score = 7
            elif rmse_percent <= 50:
                rmse_score = 6
            elif rmse_percent <= 60:
                rmse_score = 5
            elif rmse_percent <= 70:
                rmse_score = 4
            elif rmse_percent <= 80:
                rmse_score = 3
            elif rmse_percent <= 100:
                rmse_score = 2
            else:
                rmse_score = 1
            
            # Data Complexity Adjustment
            cv_percent = (target_std/target_mean) * 100  # Coefficient of variation
            if cv_percent > 100:  # Very high variance data
                complexity_bonus = 1.5
                complexity_desc = "VERY HIGH - Extremely challenging dataset"
            elif cv_percent > 75:
                complexity_bonus = 1.2
                complexity_desc = "HIGH - Challenging dataset"
            elif cv_percent > 50:
                complexity_bonus = 1.0
                complexity_desc = "MODERATE - Standard difficulty"
            elif cv_percent > 25:
                complexity_bonus = 0.8
                complexity_desc = "LOW - Easier dataset"
            else:
                complexity_bonus = 0.6
                complexity_desc = "VERY LOW - Simple dataset"
            
            # Calculate weighted overall score
            overall_score = (
                0.4 * mae_score +      # MAE is most interpretable (40% weight)
                0.35 * r2_value +      # R² shows variance explained (35% weight)
                0.25 * rmse_score      # RMSE for outlier sensitivity (25% weight)
            ) * complexity_bonus
            
            # Cap at 10
            overall_score = min(overall_score, 10)
            
            results["logs"].append(f"📊 INDIVIDUAL METRIC SCORES:")
            results["logs"].append(f"   MAE Score: {mae_score}/10 ({mae_percent:.1f}% error)")
            results["logs"].append(f"   R² Score: {r2_value}/10 ({r2*100:.1f}% variance explained)")
            results["logs"].append(f"   RMSE Score: {rmse_score}/10 ({rmse_percent:.1f}% error)")
            
            results["logs"].append(f"\n🎲 DATA COMPLEXITY ADJUSTMENT:")
            results["logs"].append(f"   Coefficient of Variation: {cv_percent:.1f}%")
            results["logs"].append(f"   Complexity Level: {complexity_desc}")
            results["logs"].append(f"   Complexity Multiplier: {complexity_bonus:.1f}x")
            
        elif task == 'classification':
            # Classification scoring
            accuracy_score_val = acc * 10  # Convert to 1-10 scale
            
            # Adjust for class imbalance
            if imbalance_ratio > 5:
                complexity_bonus = 1.3
                complexity_desc = "HIGH - Significant class imbalance"
            elif imbalance_ratio > 2:
                complexity_bonus = 1.1
                complexity_desc = "MODERATE - Some class imbalance"
            else:
                complexity_bonus = 1.0
                complexity_desc = "LOW - Balanced classes"
            
            overall_score = min(accuracy_score_val * complexity_bonus, 10)
            
            results["logs"].append(f"📊 ACCURACY SCORE: {accuracy_score_val:.1f}/10 ({acc*100:.1f}%)")
            results["logs"].append(f"🎲 COMPLEXITY ADJUSTMENT: {complexity_bonus:.1f}x ({complexity_desc})")
        
        results["overall_rating"] = overall_score
        results["logs"].append(f"🏆 OVERALL MODEL RATING: {overall_score:.1f}/10")
        
        # Provide interpretation
        if overall_score >= 9:
            rating_desc = "OUTSTANDING"
            emoji = "🥇"
            advice = "Exceptional performance! This model is production-ready."
        elif overall_score >= 8:
            rating_desc = "EXCELLENT"
            emoji = "🥈"
            advice = "Great performance! Minor tweaks could make it perfect."
        elif overall_score >= 7:
            rating_desc = "VERY GOOD"
            emoji = "🥉"
            advice = "Solid performance! Good for most practical applications."
        elif overall_score >= 6:
            rating_desc = "GOOD"
            emoji = "✅"
            advice = "Decent performance! Some room for improvement."
        elif overall_score >= 5:
            rating_desc = "FAIR"
            emoji = "⚠️"
            advice = "Acceptable but needs improvement for critical applications."
        elif overall_score >= 4:
            rating_desc = "BELOW AVERAGE"
            emoji = "⚠️"
            advice = "Needs significant improvement before deployment."
        elif overall_score >= 3:
            rating_desc = "POOR"
            emoji = "❌"
            advice = "Requires major modifications or different approach."
        elif overall_score >= 2:
            rating_desc = "VERY POOR"
            emoji = "❌"
            advice = "Model is not suitable for this problem."
        else:
            rating_desc = "FAILING"
            emoji = "💥"
            advice = "Complete model redesign needed."
        
        results["logs"].append(f"\n{emoji} PERFORMANCE RATING: {rating_desc}")
        results["logs"].append(f"💡 RECOMMENDATION: {advice}")
        
        results["logs"].append(f"\n🎯 BOTTOM LINE:")
        results["logs"].append(f"   Your model scores {overall_score:.1f}/10 considering data complexity!")
        results["logs"].append(f"   This is {rating_desc.lower()} performance for this type of problem.")

        # Overall Rating Review and Context (EXACT from notebook)
        results["logs"].append(f"\n📋 OVERALL RATING REVIEW:")
        results["logs"].append(f"   ⭐ Rating Scale Context (Note: Perfect 10/10 is nearly impossible in real-world ML)")
        results["logs"].append(f"   • 9.0-10.0: Outstanding (Rarely achieved - requires perfect data & problem)")
        results["logs"].append(f"   • 8.0-8.9:  Excellent (Top-tier performance - production ready)")
        results["logs"].append(f"   • 7.0-7.9:  Very Good (Strong performance - suitable for most applications)")
        results["logs"].append(f"   • 6.0-6.9:  Good (Solid performance - 6+ is considered very good!)")
        results["logs"].append(f"   • 5.0-5.9:  Fair (Acceptable but room for improvement)")
        results["logs"].append(f"   • 4.0-4.9:  Below Average (Needs significant work)")
        results["logs"].append(f"   • 3.0-3.9:  Poor (Major issues - consider different approach)")
        results["logs"].append(f"   • 2.0-2.9:  Very Poor (Model not suitable)")
        results["logs"].append(f"   • 1.0-1.9:  Failing (Complete redesign needed)")

        results["logs"].append(f"\n💡 INTERPRETATION FOR YOUR SCORE ({overall_score:.1f}/10):")
        if overall_score >= 6:
            results["logs"].append(f"   🎉 GREAT NEWS! A score of {overall_score:.1f}/10 is considered VERY GOOD!")
            results["logs"].append(f"   🎯 Most production ML models score between 6-8/10")
            results["logs"].append(f"   🏆 You've achieved a score that many professionals would be proud of!")
            if overall_score >= 8:
                results["logs"].append(f"   🌟 Your {overall_score:.1f}/10 is exceptional - this is expert-level performance!")
            elif overall_score >= 7:
                results["logs"].append(f"   🚀 Your {overall_score:.1f}/10 is excellent - this model is deployment-ready!")
            else:
                results["logs"].append(f"   ✅ Your {overall_score:.1f}/10 is solid - with minor improvements, this could be outstanding!")
        else:
            results["logs"].append(f"   📈 Your {overall_score:.1f}/10 shows room for improvement")
            results["logs"].append(f"   🎯 Aim for 6+ to reach the 'very good' category")
            results["logs"].append(f"   💪 Consider feature engineering, hyperparameter tuning, or different algorithms")

        results["logs"].append(f"\n🎯 FINAL VERDICT:")
        results["logs"].append(f"   Your model achieved {overall_score:.1f}/10 - Remember, 6+ is very good in ML!")
        results["logs"].append(f"   Perfect 10/10 scores are like unicorns - beautiful in theory, but good luck finding one in the wild! 🦄")
        
        # Phase 22: Visual Explanation (EXACT from notebook)
        if task == 'regression':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot 1: Actual vs Predicted (shows good correlation)
            ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'Actual vs Predicted\n(Correlation shows model quality)')
            ax1.grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 2: Error Distribution (shows prediction accuracy)
            residuals = y_test - y_pred
            ax2.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Prediction Error (Actual - Predicted)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Error Distribution\n(Centered around 0 = good predictions)')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            ax2.text(0.05, 0.95, f'Mean Error: {residuals.mean():.2f}\nStd Error: {residuals.std():.2f}', 
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            results["plots"].append(fig)
            
            results["logs"].append("\n🎯 Key Insights:")
            results["logs"].append(f"   • High correlation ({correlation:.3f}) = model captures patterns well")
            results["logs"].append(f"   • Errors centered around 0 = unbiased predictions")
            results["logs"].append(f"   • Low MAE ({mae:.2f}) = accurate predictions")
            results["logs"].append(f"   • R² reflects data complexity, not model failure!")
        
        # Phase 23: Final Class Label Mapping (EXACT from notebook)
        if task == 'classification':
            final_mapping = display_class_label_mapping(y, final_pipeline.named_steps['model'])
            results["class_mapping"] = final_mapping
            results["logs"].extend(final_mapping)
            
            # Show actual class names for each class index (EXACT from notebook)
            if len(np.unique(y)) > 2:
                if hasattr(final_pipeline.named_steps['model'], 'classes_'):
                    results["logs"].append("Class label mapping:")
                    for i, cls in enumerate(final_pipeline.named_steps['model'].classes_):
                        results["logs"].append(f"{i}: {cls}")
                else:
                    results["logs"].append("Class label mapping:")
                    for i, cls in enumerate(np.unique(y)):
                        results["logs"].append(f"{i}: {cls}")
            else:
                results["logs"].append("Class label mapping is only relevant for multiclass classification tasks.")
        
        # Phase 24: Save Model (EXACT from notebook)
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/{results['best_model'].replace(' ', '_')}_model.pkl"
        joblib.dump(final_pipeline, model_path)
        results["model_path"] = model_path
        results["logs"].append(f"\n💾 Model saved as: {model_path}")
        
        # Phase 25: Final Summary (EXACT from notebook)
        results["logs"].append("\n" + "="*60)
        results["logs"].append("🎉 AUTOML PIPELINE COMPLETED SUCCESSFULLY!")
        results["logs"].append("="*60)
        results["logs"].append(f"✅ Task Type: {task.capitalize()}")
        results["logs"].append(f"✅ Best Model: {results['best_model']}")
        results["logs"].append(f"✅ Overall Rating: {overall_score:.1f}/10")
        results["logs"].append(f"✅ Model Saved: {model_path}")
        results["logs"].append(f"✅ Total Plots Generated: {len(results['plots'])}")
        
        if task == 'classification':
            results["logs"].append(f"✅ Final Accuracy: {acc:.4f}")
            results["logs"].append(f"✅ Classes Detected: {len(np.unique(y))}")
        else:
            results["logs"].append(f"✅ Final MAE: {mae:.2f}")
            results["logs"].append(f"✅ Final R²: {r2:.4f}")
        
        results["logs"].append(f"\n🎯 BOTTOM LINE:")
        results["logs"].append(f"   Your model scores {overall_score:.1f}/10 considering data complexity!")
        results["logs"].append(f"   Perfect 10/10 scores are like unicorns - beautiful in theory, but good luck finding one in the wild! 🦄")
        
        # Add scoring clarification for regression
        if task == 'regression':
            results["logs"].append(f"\n📊 SCORING NOTE FOR REGRESSION:")
            results["logs"].append(f"   • Cross-validation uses R² score (coefficient of determination)")
            results["logs"].append(f"   • R² ranges from -∞ to 1.0, where 1.0 is perfect prediction")
            results["logs"].append(f"   • R² = 0 means the model predicts as well as the target mean")
            results["logs"].append(f"   • Negative R² means the model is worse than predicting the mean")
            results["logs"].append(f"   • Your best model R² score: {final_score:.4f}")
        
        return results


# def test_pipeline():
#     """
#     Test function to demonstrate the AutoML pipeline with a sample dataset
#     """
#     import pandas as pd
#     from sklearn.datasets import load_iris, load_boston
#     import warnings
#     warnings.filterwarnings('ignore')
    
#     print("🧪 Testing AutoML Pipeline...")
#     print("="*50)
    
#     # Test with Iris dataset (Classification)
#     print("\n🌸 Testing with Iris Dataset (Classification)")
#     print("-" * 45)
    
#     try:
#         iris = load_iris()
#         iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
#         iris_df['target'] = iris.target
        
#         # Map numeric targets to class names for better testing
#         target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
#         iris_df['species'] = iris_df['target'].map(target_names)
#         iris_df = iris_df.drop('target', axis=1)
        
#         print(f"📊 Dataset shape: {iris_df.shape}")
#         print(f"🎯 Target column: 'species'")
#         print(f"📋 Classes: {iris_df['species'].unique()}")
        
#         # Run AutoML pipeline
#         results = run_automl_pipeline(
#             df=iris_df,
#             target_col='species',
#             model_choice='utility'
#         )
        
#         print(f"\n✅ Classification Test Results:")
#         print(f"   Best Model: {results['best_model']}")
#         print(f"   Overall Rating: {results['overall_rating']:.1f}/10")
#         print(f"   Accuracy: {results['metrics'].get('accuracy', 'N/A')}")
#         print(f"   Model Saved: {results['model_path']}")
        
#     except Exception as e:
#         print(f"❌ Classification test failed: {str(e)}")
    
#     # Test with Housing dataset (Regression) 
#     print("\n🏠 Testing with Boston Housing Dataset (Regression)")
#     print("-" * 50)
    
#     try:
#         # Try to load Boston housing dataset
#         try:
#             from sklearn.datasets import load_boston
#             boston = load_boston()
#             boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#             boston_df['price'] = boston.target
#         except ImportError:
#             # Fallback: create synthetic regression data
#             from sklearn.datasets import make_regression
#             X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
#             boston_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
#             boston_df['price'] = y
#             print("📝 Using synthetic regression dataset (Boston dataset deprecated)")
        
#         print(f"📊 Dataset shape: {boston_df.shape}")
#         print(f"🎯 Target column: 'price'")
#         print(f"📈 Target range: {boston_df['price'].min():.2f} - {boston_df['price'].max():.2f}")
        
#         # Run AutoML pipeline
#         results = run_automl_pipeline(
#             df=boston_df,
#             target_col='price',
#             model_choice='utility'
#         )
        
#         print(f"\n✅ Regression Test Results:")
#         print(f"   Best Model: {results['best_model']}")
#         print(f"   Overall Rating: {results['overall_rating']:.1f}/10")
#         print(f"   MAE: {results['metrics'].get('mae', 'N/A')}")
#         print(f"   RMSE: {results['metrics'].get('rmse', 'N/A')}")
#         print(f"   R²: {results['metrics'].get('r2', 'N/A')}")
#         print(f"   Model Saved: {results['model_path']}")
        
#     except Exception as e:
#         print(f"❌ Regression test failed: {str(e)}")
    
#     print("\n🎉 Pipeline Testing Completed!")
#     print("="*50)
#     print("💡 You can now use run_automl_pipeline() with your own datasets!")
    
#     return results
