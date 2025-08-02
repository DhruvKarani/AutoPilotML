import pandas as pd
from automl_pipeline import detect_cat_num

# Test heart dataset
df = pd.read_csv('Datasets/heart.csv')
print('HEART DATASET FEATURE DETECTION')
print('='*50)

target = 'HeartDisease'
features = [col for col in df.columns if col != target]

numerical_features = []
categorical_features = []

for col in features:
    feature_type = detect_cat_num(df[col])
    print(f'{col}: {feature_type} (dtype: {df[col].dtype}, unique: {df[col].nunique()})')
    
    if feature_type == 'numerical':
        numerical_features.append(col)
    elif feature_type == 'categorical':
        categorical_features.append(col)

print(f'\nSUMMARY:')
print(f'Numerical: {len(numerical_features)} -> {numerical_features}')
print(f'Categorical: {len(categorical_features)} -> {categorical_features}')

# Show unique values for categorical features
print(f'\nCategorical feature values:')
for col in categorical_features:
    print(f'{col}: {df[col].unique()}')
