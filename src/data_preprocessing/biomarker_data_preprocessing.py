import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_biomarker_data(file_path):
    """
    Loads and preprocesses the biomarker data from the specified CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test: The split and preprocessed data.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop the 'id' column as it is not useful for prediction
    df = df.drop('id', axis=1)

    # Handle 'Other' gender category by removing it
    if 'Other' in df['gender'].unique():
        df = df[df['gender'] != 'Other']

    # Define categorical and numerical features
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    
    # Target variable
    target = 'stroke'

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like hypertension, heart_disease)
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage:
    preprocessor, X_train, X_test, y_train, y_test = preprocess_biomarker_data('dataset/healthcare-dataset-stroke-data.csv')
    
    # To apply the preprocessing:
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Shape of processed training data:", X_train_processed.shape)
    print("Shape of processed testing data:", X_test_processed.shape)
    print("Training labels distribution:\n", y_train.value_counts())
    print("Testing labels distribution:\n", y_test.value_counts())
