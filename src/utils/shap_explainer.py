import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BiomarkerExplainer:
    """
    SHAP-based explainer for the biomarker stroke risk model.
    Provides feature importance and explanations for individual predictions.
    """
    def __init__(self, model, preprocessor, device='cpu'):
        """
        Initialize the explainer.
        
        Args:
            model: Trained biomarker model
            preprocessor: The sklearn preprocessor used to transform input data
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.feature_names = preprocessor.get_feature_names_out()
        
    def predict_proba(self, X):
        """
        Prediction function for SHAP that returns probabilities.
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Probability predictions
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    def explain_prediction(self, patient_data, background_data=None, num_samples=100):
        """
        Explain a single patient's stroke risk prediction.
        
        Args:
            patient_data: pandas DataFrame with patient's biomarker data (single row)
            background_data: pandas DataFrame for background distribution (optional)
            num_samples: Number of samples for SHAP estimation
            
        Returns:
            shap_values: SHAP values for the prediction
            base_value: The base/expected value
            prediction: The model's prediction
        """
        # Preprocess the patient data
        X_processed = self.preprocessor.transform(patient_data)
        if not isinstance(X_processed, np.ndarray):
            X_processed = X_processed.toarray()
        
        # If no background data provided, use the patient data itself
        if background_data is None:
            background_processed = X_processed
        else:
            background_processed = self.preprocessor.transform(background_data)
            if not isinstance(background_processed, np.ndarray):
                background_processed = background_processed.toarray()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model=lambda x: self.predict_proba(x)[:, 1],  # Probability of stroke
            data=shap.sample(background_processed, min(num_samples, len(background_processed)))
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_processed)
        
        # Get prediction
        prediction = self.predict_proba(X_processed)[0]
        
        return shap_values, explainer.expected_value, prediction
    
    def plot_explanation(self, patient_data, background_data=None, save_path=None):
        """
        Create a waterfall plot showing the contribution of each feature.
        
        Args:
            patient_data: pandas DataFrame with patient's biomarker data
            background_data: pandas DataFrame for background distribution (optional)
            save_path: Path to save the plot (optional)
            
        Returns:
            fig: Matplotlib figure
        """
        shap_values, base_value, prediction = self.explain_prediction(patient_data, background_data)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the processed feature values
        X_processed = self.preprocessor.transform(patient_data)
        if not isinstance(X_processed, np.ndarray):
            X_processed = X_processed.toarray()
        
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values[0]))[::-1][:15]  # Top 15 features
        
        # Create bar plot
        feature_names = [self.feature_names[i] for i in indices]
        shap_vals = [shap_values[0][i] for i in indices]
        
        colors = ['red' if val > 0 else 'blue' for val in shap_vals]
        ax.barh(feature_names, shap_vals, color=colors)
        ax.set_xlabel('SHAP Value (Impact on Stroke Risk)', fontsize=12)
        ax.set_title(f'Feature Importance for Stroke Risk Prediction\nPredicted Probability: {prediction[1]:.2%}', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_summary(self, test_data, num_samples=100, save_path=None):
        """
        Create a summary plot showing feature importance across multiple samples.
        
        Args:
            test_data: pandas DataFrame with multiple patients' data
            num_samples: Number of samples to use
            save_path: Path to save the plot (optional)
            
        Returns:
            fig: Matplotlib figure
        """
        # Preprocess the data
        X_processed = self.preprocessor.transform(test_data.sample(min(num_samples, len(test_data))))
        if not isinstance(X_processed, np.ndarray):
            X_processed = X_processed.toarray()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            model=lambda x: self.predict_proba(x)[:, 1],
            data=shap.sample(X_processed, min(50, len(X_processed)))
        )
        
        # Calculate SHAP values for all samples
        shap_values = explainer.shap_values(X_processed)
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=self.feature_names, show=False)
        plt.title('Feature Importance Summary Across Patients', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def explain_biomarker_prediction(model_path, patient_data_dict):
    """
    High-level function to explain a biomarker prediction.
    
    Args:
        model_path: Path to the saved model checkpoint
        patient_data_dict: Dictionary with patient's biomarker values
        
    Returns:
        explanation_dict: Dictionary with prediction and explanations
    """
    # Load model and preprocessor
    checkpoint = torch.load(model_path, map_location='cpu')
    
    from src.models.biomarker_model import BiomarkerModel
    preprocessor = checkpoint['preprocessor']
    
    # Get input dimension
    input_dim = len(preprocessor.get_feature_names_out())
    model = BiomarkerModel(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create explainer
    explainer = BiomarkerExplainer(model, preprocessor)
    
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data_dict])
    
    # Get explanation
    shap_values, base_value, prediction = explainer.explain_prediction(patient_df)
    
    # Get top contributing features
    indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
    top_features = {
        explainer.feature_names[i]: float(shap_values[0][i])
        for i in indices
    }
    
    return {
        'stroke_probability': float(prediction[1]),
        'no_stroke_probability': float(prediction[0]),
        'top_risk_factors': top_features,
        'base_risk': float(base_value)
    }


if __name__ == '__main__':
    print("SHAP-based explainer for biomarker model ready.")
    print("Use the 'explain_biomarker_prediction' function to generate explanations.")
