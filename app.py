import gradio as gr
import os
from pathlib import Path
import sys
import tempfile

# Add the project root to the Python path to allow imports from src
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.prediction.predict_image_only import predict_image_only

# --- Helper function to define the Gradio UI and logic ---
def create_gradio_app():
    """
    Creates and launches the Gradio web interface for stroke prediction.
    """
    
    # Custom CSS for styling the app
    css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .gr-button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
    }
    .gr-button:hover {
        background-color: #0056b3;
    }
    #app-title {
        text-align: center;
        font-size: 2.5em;
        color: #333;
    }
    #app-subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 20px;
    }
    #footer {
        text-align: center;
        color: #888;
        font-size: 0.9em;
    }
    """

    def prediction_wrapper(image):
        """
        A wrapper function to handle inputs from Gradio, call the prediction
        function, and format the output for two separate components.
        """
        if image is None:
            raise gr.Error("Please upload an MRI image.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            temp_image_path = tmp.name

        try:
            result = predict_image_only(image_path=temp_image_path)
            
            if "error" in result:
                raise gr.Error(result["error"])

            predicted_class = result['predicted_class']
            probabilities = {label: float(prob) for label, prob in result['probabilities'].items()}
            
            # Return the main prediction and the full probability distribution
            return f"The model predicts: **{predicted_class}**", probabilities

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise gr.Error(str(e))
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    # --- Define the Gradio Interface Components ---
    with gr.Blocks(theme=gr.themes.Glass(), css=css, title="Stroke-Vs") as app:
        gr.Markdown("# Stroke-Vs: AI-Powered Stroke Classification", elem_id="app-title")
        gr.Markdown("Upload a brain MRI scan to predict the stroke type (Haemorrhagic, Ischemic, or Normal).", elem_id="app-subtitle")

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### 1. Upload MRI Image")
                image_input = gr.Image(type="pil", label="MRI Scan", height=400)
                submit_button = gr.Button("Analyze Image", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 2. Prediction Result")
                main_prediction_output = gr.Markdown(label="Main Prediction")
                label_output = gr.Label(label="Confidence Score", num_top_classes=3)
        
        with gr.Accordion("Show Examples", open=False):
            gr.Examples(
                examples=[
                    str(project_root / 'MRI_DATA' / 'Stroke_classification' / 'Ischemic' / 'Ellappan T2-12.jpg_Ischemic_10.png'),
                    str(project_root / 'MRI_DATA' / 'Stroke_classification' / 'Haemorrhagic' / 'Balu T2-17.jpg_Hemo_10.png'),
                    str(project_root / 'MRI_DATA' / 'Stroke_classification' / 'Normal' / 'Normal- (6).jpg')
                ],
                inputs=image_input,
                # The outputs must match the number and type of outputs from the function
                outputs=[main_prediction_output, label_output], 
                fn=prediction_wrapper,
                cache_examples=False
            )

        gr.Markdown("---", elem_id="footer")
        gr.Markdown("Developed by Bharat", elem_id="footer")
        
        # --- Connect the components to the prediction function ---
        submit_button.click(
            fn=prediction_wrapper,
            inputs=[image_input],
            outputs=[main_prediction_output, label_output]
        )

    return app

# --- Main execution block ---
if __name__ == "__main__":
    # Check for necessary model files before launching
    model_path = project_root / 'src' / 'prediction' / 'image_only_model_weights.pth'
    
    if not model_path.exists():
        print("Error: The required model file 'image_only_model_weights.pth' was not found.")
        print("Please run the training script first: python -m src.training.train_image_model_standalone")
    else:
        print("All model files found. Launching Gradio app...")
        app = create_gradio_app()
        app.launch(share=False)
