import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_clinical_data(df, target_column='stroke', preprocessor=None, fit_smote=True):
    """
    Loads and preprocesses clinical tabular data using a scikit-learn pipeline.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str, optional): The name of the target variable column. Defaults to 'stroke'.
        preprocessor (Pipeline, optional): A pre-fitted scikit-learn pipeline. 
                                           If None, a new one is created and fitted. Defaults to None.
        fit_smote (bool, optional): If True, applies SMOTE for oversampling. Defaults to True.

    Returns:
        tuple: A tuple containing:
               - The preprocessed features (np.array).
               - The target labels (np.array).
               - The fitted preprocessor pipeline.
    """
    # Drop rows with missing target values if fitting a new preprocessor
    if preprocessor is None:
        df.dropna(subset=[target_column], inplace=True)

    # Handle 'Other' gender category by removing it
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Other']

    # Separate features and target
    X = df.drop(columns=[target_column, 'id'], errors='ignore')
    y = df[target_column]
    
    if preprocessor is None:
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=['number']).columns

        # Create preprocessing pipelines for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create a column transformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep other columns (if any)
        )
        
        # Fit the preprocessor
        X_processed = preprocessor.fit_transform(X)
    else:
        # Use the provided preprocessor to transform the data
        X_processed = preprocessor.transform(X)

    # Apply SMOTE to handle class imbalance if required
    if fit_smote:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        return X_resampled, y_resampled.to_numpy(), preprocessor
    else:
        return X_processed, y.to_numpy(), preprocessor

if __name__ == '__main__':
    data_path = 'clinical_lab_data/healthcare-dataset-stroke-data.csv'
    
    try:
        df = pd.read_csv(data_path)
        X_processed, y_processed, fitted_preprocessor = preprocess_clinical_data(df)
        
        print("Shape of processed features:", X_processed.shape)
        print("Class distribution after SMOTE:\n", pd.Series(y_processed).value_counts())
        print("\nFitted Preprocessor:")
        print(fitted_preprocessor)
    except FileNotFoundError:
        print(f"The file {data_path} was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

