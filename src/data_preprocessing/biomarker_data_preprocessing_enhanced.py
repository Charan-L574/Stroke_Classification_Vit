import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_merge_biomarker_datasets():
    """
    Load and intelligently merge all 3 biomarker datasets.
    
    Strategy:
    1. Use dataset.csv as base (43,401 samples, same structure)
    2. Add healthcare-dataset-stroke-data.csv (5,110 samples) for more stroke cases
    3. Map features from diabetes_data.csv and merge where possible
    
    Returns:
        merged_df: Combined dataframe with enhanced features
    """
    print("="*80)
    print("LOADING AND MERGING ALL 3 BIOMARKER DATASETS")
    print("="*80)
    
    # Load Dataset 1: healthcare-dataset-stroke-data.csv
    print("\n📂 Loading healthcare-dataset-stroke-data.csv...")
    df1 = pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')
    print(f"  ✓ Loaded {len(df1)} samples, {df1['stroke'].sum()} strokes ({df1['stroke'].mean()*100:.2f}%)")
    
    # Load Dataset 2: dataset.csv  
    print("\n📂 Loading dataset.csv...")
    df2 = pd.read_csv('dataset/dataset.csv')
    print(f"  ✓ Loaded {len(df2)} samples, {df2['stroke'].sum()} strokes ({df2['stroke'].mean()*100:.2f}%)")
    
    # Load Dataset 3: diabetes_data.csv
    print("\n📂 Loading diabetes_data.csv...")
    df3 = pd.read_csv('dataset/diabetes_data.csv')
    print(f"  ✓ Loaded {len(df3)} samples, {int(df3['Stroke'].sum())} strokes ({df3['Stroke'].mean()*100:.2f}%)")
    
    # Step 1: Merge df1 and df2 (same structure)
    print("\n🔗 Step 1: Merging healthcare-dataset + dataset.csv...")
    df_combined = pd.concat([df1, df2], ignore_index=True)
    # Remove duplicates based on id
    df_combined = df_combined.drop_duplicates(subset=['id'], keep='first')
    print(f"  ✓ Combined: {len(df_combined)} samples, {df_combined['stroke'].sum()} strokes ({df_combined['stroke'].mean()*100:.2f}%)")
    
    # Step 2: Map diabetes_data.csv features to common schema
    print("\n🔗 Step 2: Mapping diabetes_data.csv to common schema...")
    
    # Create mapping for diabetes dataset
    df3_mapped = pd.DataFrame()
    df3_mapped['age'] = df3['Age']
    df3_mapped['gender'] = df3['Sex'].map({1.0: 'Male', 0.0: 'Female'})
    df3_mapped['bmi'] = df3['BMI']
    df3_mapped['hypertension'] = df3['HighBP'].astype(int)
    df3_mapped['heart_disease'] = df3['HeartDiseaseorAttack'].astype(int)
    df3_mapped['stroke'] = df3['Stroke'].astype(int)
    df3_mapped['smoking_status'] = df3['Smoker'].map({1.0: 'smokes', 0.0: 'never smoked'})
    
    # Features not available in diabetes dataset - use defaults/NA
    df3_mapped['id'] = range(100000, 100000 + len(df3))  # Unique IDs
    df3_mapped['ever_married'] = 'Unknown'  # Not available
    df3_mapped['work_type'] = 'Unknown'  # Not available
    df3_mapped['Residence_type'] = 'Unknown'  # Not available
    df3_mapped['avg_glucose_level'] = np.nan  # Not available - will be imputed
    
    # Add extra features from diabetes dataset as new columns
    df3_mapped['high_cholesterol'] = df3['HighChol'].astype(int)
    df3_mapped['physical_activity'] = df3['PhysActivity'].astype(int)
    df3_mapped['diabetes'] = df3['Diabetes'].astype(int)
    df3_mapped['mental_health_days'] = df3['MentHlth']
    df3_mapped['physical_health_days'] = df3['PhysHlth']
    df3_mapped['difficulty_walking'] = df3['DiffWalk'].astype(int)
    
    print(f"  ✓ Mapped {len(df3_mapped)} samples from diabetes dataset")
    print(f"  ✓ Added 6 extra features: high_cholesterol, physical_activity, diabetes, mental_health_days, physical_health_days, difficulty_walking")
    
    # Step 3: Add extra columns to df_combined (fill with NA for datasets 1 & 2)
    extra_features = ['high_cholesterol', 'physical_activity', 'diabetes', 
                      'mental_health_days', 'physical_health_days', 'difficulty_walking']
    for feature in extra_features:
        df_combined[feature] = np.nan
    
    # Step 4: Combine all datasets
    print("\n🔗 Step 3: Merging with diabetes dataset...")
    df_final = pd.concat([df_combined, df3_mapped], ignore_index=True)
    
    print(f"\n✅ FINAL MERGED DATASET:")
    print(f"  Total samples: {len(df_final)}")
    print(f"  Total strokes: {df_final['stroke'].sum()}")
    print(f"  Stroke rate: {df_final['stroke'].mean()*100:.2f}%")
    print(f"  Total features: {len(df_final.columns)}")
    print(f"  Feature list: {list(df_final.columns)}")
    
    return df_final


def preprocess_biomarker_data_enhanced(use_all_datasets=True):
    """
    Enhanced biomarker preprocessing using all 3 datasets.
    
    Args:
        use_all_datasets (bool): If True, merges all 3 datasets. If False, uses only healthcare dataset.
    
    Returns:
        preprocessor: Fitted preprocessing pipeline
        X_train, X_test, y_train, y_test: Split and preprocessed data
    """
    
    if use_all_datasets:
        print("\n🚀 Using ENHANCED mode with all 3 datasets merged!")
        df = load_and_merge_biomarker_datasets()
    else:
        print("\n📊 Using STANDARD mode with healthcare-dataset-stroke-data.csv only")
        df = pd.read_csv('dataset/healthcare-dataset-stroke-data.csv')
    
    # Drop the 'id' column
    df = df.drop('id', axis=1, errors='ignore')
    
    # Handle 'Other' gender category by removing it
    if 'gender' in df.columns and 'Other' in df['gender'].unique():
        df = df[df['gender'] != 'Other']
    
    # Define features based on availability
    if use_all_datasets:
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical_features = ['age', 'avg_glucose_level', 'bmi', 'mental_health_days', 'physical_health_days']
        binary_features = ['hypertension', 'heart_disease', 'high_cholesterol', 
                          'physical_activity', 'diabetes', 'difficulty_walking']
    else:
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        binary_features = ['hypertension', 'heart_disease']
    
    # Target variable
    target = 'stroke'
    
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    print(f"\n📊 Data Split:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Class distribution: No Stroke={sum(y==0)}, Stroke={sum(y==1)}")
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # For binary features, just impute with 0 (most common for health data)
    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    
    # Create column transformer
    transformers = [
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
    
    # Add binary features transformer if using enhanced mode
    if use_all_datasets and binary_features:
        transformers.append(('bin', binary_pipeline, binary_features))
    else:
        # In standard mode, hypertension and heart_disease are passed through
        transformers.append(('bin', 'passthrough', binary_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any remaining columns
    )
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n✅ Train/Test Split:")
    print(f"  Train: {len(X_train)} samples ({sum(y_train==1)} strokes, {sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"  Test: {len(X_test)} samples ({sum(y_test==1)} strokes, {sum(y_test==1)/len(y_test)*100:.2f}%)")
    
    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING STANDARD MODE (healthcare dataset only)")
    print("="*80)
    
    preprocessor_std, X_train_std, X_test_std, y_train_std, y_test_std = preprocess_biomarker_data_enhanced(use_all_datasets=False)
    X_train_processed_std = preprocessor_std.fit_transform(X_train_std)
    X_test_processed_std = preprocessor_std.transform(X_test_std)
    
    print(f"\nStandard Mode Results:")
    print(f"  Processed train shape: {X_train_processed_std.shape}")
    print(f"  Processed test shape: {X_test_processed_std.shape}")
    
    print("\n" + "="*80)
    print("TESTING ENHANCED MODE (all 3 datasets merged)")
    print("="*80)
    
    preprocessor_enh, X_train_enh, X_test_enh, y_train_enh, y_test_enh = preprocess_biomarker_data_enhanced(use_all_datasets=True)
    X_train_processed_enh = preprocessor_enh.fit_transform(X_train_enh)
    X_test_processed_enh = preprocessor_enh.transform(X_test_enh)
    
    print(f"\nEnhanced Mode Results:")
    print(f"  Processed train shape: {X_train_processed_enh.shape}")
    print(f"  Processed test shape: {X_test_processed_enh.shape}")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Standard Mode: {len(X_train_std) + len(X_test_std)} total samples, {X_train_processed_std.shape[1]} features")
    print(f"Enhanced Mode: {len(X_train_enh) + len(X_test_enh)} total samples, {X_train_processed_enh.shape[1]} features")
    print(f"\nData increase: {((len(X_train_enh) + len(X_test_enh)) / (len(X_train_std) + len(X_test_std)) - 1) * 100:.1f}%")
    print(f"Feature increase: {X_train_processed_enh.shape[1] - X_train_processed_std.shape[1]} additional features")
