"""
Main application entry point for Advanced Stroke Classification & Monitoring System.
Modular implementation using inference modules and Gradio UI.
"""

import gradio as gr
import pandas as pd
import torch

# Import inference modules
from src.inference.model_loader import model_loader
from src.inference.biomarker_inference import predict_stroke_risk
from src.inference.image_inference import analyze_brain_scan
from src.inference.eeg_inference import analyze_eeg_signal
from src.inference.fusion_inference import analyze_fusion


# ===========================
# Load Models on Startup
# ===========================

print("=" * 60)
print("🚀 Starting Advanced Stroke Classification System")
print("=" * 60)

# Load all trained models
model_loader.load_all_models()

print("\n" + "=" * 60)
print("✓ System Ready - All models loaded successfully")
print("=" * 60 + "\n")


# ===========================
# EEG Pattern Loaders
# ===========================

def load_eeg_pattern(class_id, fallback_pattern):
    """Load EEG pattern from CSV or return fallback"""
    try:
        df = pd.read_csv(f'csv_samples/class_{class_id}_sample.csv')
        return ",".join([str(x) for x in df['eeg_value'].values])
    except:
        return fallback_pattern


def load_pattern_a():
    """Class 0: Normal Conscious State"""
    return load_eeg_pattern(0, "5,5,6,7,5,9,4,9,2,7,0,5,0,1,1,2,3,4,6,4," + ",".join(["4"] * 236))


def load_pattern_b():
    """Class 1: Drowsy/Sedated State"""
    return load_eeg_pattern(1, "15,9,14,6,11,5,8,4,4,4,1,4,2,5,4,6,4,6,4,7," + ",".join(["7"] * 236))


def load_pattern_c():
    """Class 2: Deep Sleep/Unconscious"""
    return load_eeg_pattern(2, "2,10,14,9,32,2,47,8,57,21,59,32,55,39,45,39,34,32,24,20," + ",".join(["14"] * 236))


def load_pattern_d():
    """Class 3: Seizure Activity Detected"""
    return load_eeg_pattern(3, "3,2,3,1,3,1,3,1,4,2,5,2,7,2,8,1,9,1,10,3," + ",".join(["2"] * 236))


def load_pattern_e():
    """Class 4: Critical Suppression"""
    return load_eeg_pattern(4, "6,9,5,11,3,11,1,9,1,7,1,3,0,1,3,3,6,4,9,4," + ",".join(["6"] * 236))


# ===========================
# Fusion Analysis Wrapper
# ===========================

def fusion_analysis_wrapper(g, a, sbp, dbp, hd, gl, bmi, sm, scan, eeg):
    """Wrapper for fusion analysis to match Gradio interface"""
    biomarker_inputs = {
        'gender': g,
        'age': a,
        'systolic_bp': sbp,
        'diastolic_bp': dbp,
        'heart_disease': hd,
        'avg_glucose_level': gl,
        'bmi': bmi,
        'smoking_status': sm
    }
    return analyze_fusion(biomarker_inputs, scan, eeg)


# ===========================
# Create Gradio Interface
# ===========================

with gr.Blocks(title="Advanced Stroke Analysis System", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🧠 Advanced Stroke Classification & Analysis System")
    gr.Markdown("*AI-powered stroke risk assessment and post-stroke monitoring*")
    
    with gr.Tabs():
        
        # ===========================
        # Tab 1: Biomarker Risk Assessment
        # ===========================
        with gr.Tab("🩺 Step 1: Biomarker Risk Assessment"):
            gr.Markdown("### Patient Clinical Data (Medical Factors Only)")
            gr.Markdown("*Enter key medical parameters to assess stroke risk*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Basic Demographics**")
                    gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    age = gr.Number(label="Age (years) - Max: 120", value=65, minimum=0, maximum=120)
                    
                    gr.Markdown("**Cardiovascular Health**")
                    systolic_bp = gr.Number(label="Systolic BP (mmHg) - Max: 250", value=120, minimum=50, maximum=250)
                    diastolic_bp = gr.Number(label="Diastolic BP (mmHg) - Max: 150", value=80, minimum=30, maximum=150)
                    heart_disease = gr.Checkbox(label="Heart Disease")
                    high_cholesterol = gr.Checkbox(label="High Cholesterol", value=False)
                
                with gr.Column():
                    gr.Markdown("**Metabolic Factors**")
                    avg_glucose_level = gr.Number(label="Average Glucose Level (mg/dL) - Max: 400", value=100, minimum=50, maximum=400)
                    bmi = gr.Number(label="BMI (Body Mass Index) - Max: 60", value=25, minimum=10, maximum=60)
                    diabetes = gr.Checkbox(label="Diabetes", value=False)
                    
                    gr.Markdown("**Lifestyle & Mobility**")
                    smoking_status = gr.Dropdown(
                        ["never smoked", "formerly smoked", "smokes", "Unknown"], 
                        label="Smoking Status", 
                        value="never smoked"
                    )
                    physical_activity = gr.Checkbox(label="Regular Physical Activity (30+ min/day)", value=True)
                    difficulty_walking = gr.Checkbox(label="Difficulty Walking", value=False)
            
            predict_btn = gr.Button("🔍 Assess Stroke Risk", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    output_text = gr.Markdown()
                with gr.Column():
                    output_plot = gr.Markdown()
            
            predict_btn.click(
                fn=predict_stroke_risk,
                inputs=[gender, age, systolic_bp, diastolic_bp, heart_disease, avg_glucose_level, bmi, smoking_status, 
                        high_cholesterol, diabetes, physical_activity, difficulty_walking],
                outputs=[output_text, output_plot]
            )
            
            gr.Markdown("---")
            gr.Markdown("*The health routine provides personalized recommendations based on your biomarkers to help prevent stroke.*")
        
        # ===========================
        # Tab 2: MRI/CT Scan Analysis
        # ===========================
        with gr.Tab("🔬 Step 2: MRI/CT Scan Analysis"):
            gr.Markdown("### Brain Imaging Analysis with AI Explainability")
            gr.Markdown("*Upload MRI or CT scan for detailed stroke detection*")
            
            # Top row: Upload and Grad-CAM side by side
            with gr.Row():
                with gr.Column():
                    scan_input = gr.Image(label="Upload Brain Scan (MRI/CT)", type="pil", height=400)
                with gr.Column():
                    scan_explanation = gr.Image(label="Grad-CAM Heatmap Overlay", height=400)
            
            analyze_scan_btn = gr.Button("🔬 Analyze Brain Scan", variant="primary", size="lg")
            
            # Bottom row: Results and Explanation side by side
            with gr.Row():
                with gr.Column():
                    scan_result = gr.Markdown()
                with gr.Column():
                    gradcam_explanation = gr.Markdown()
            
            analyze_scan_btn.click(
                fn=analyze_brain_scan,
                inputs=[scan_input],
                outputs=[scan_result, scan_explanation, gradcam_explanation]
            )
        
        # ===========================
        # Tab 3: EEG Post-Stroke Monitoring
        # ===========================
        with gr.Tab("🧪 Step 3: EEG Post-Stroke Monitoring"):
            gr.Markdown("### Post-Stroke Brain Activity Monitoring")
            gr.Markdown("*Analyze EEG signals to monitor neurological recovery and detect complications*")
            
            gr.Markdown("""
            **When to use:** After stroke event, during recovery phase
            
            **What it monitors:**
            - Consciousness level and cognitive function
            - Seizure activity (5-20% post-stroke seizure risk)
            - Brain recovery patterns
            - Early warning signs of complications
            """)
            
            # Example EEG pattern buttons
            gr.Markdown("**Quick Test Examples:**")
            with gr.Row():
                pattern_a_btn = gr.Button("✅ Normal Pattern", size="sm")
                pattern_b_btn = gr.Button("😴 Drowsy Pattern", size="sm")
                pattern_c_btn = gr.Button("💤 Sleep Pattern", size="sm")
            with gr.Row():
                pattern_d_btn = gr.Button("⚠️ Seizure Pattern", size="sm")
                pattern_e_btn = gr.Button("🔴 Suppression Pattern", size="sm")
            
            # CSV Upload option
            gr.Markdown("**Or Upload CSV File:**")
            gr.Markdown("💡 *Sample CSV files available in `csv_samples/` folder*")
            eeg_csv_upload = gr.File(
                label="Upload EEG CSV File (256 values)",
                file_types=[".csv"],
                type="filepath"
            )
            
            eeg_input = gr.Textbox(
                label="EEG Signal Data (256 samples, comma-separated)",
                placeholder="Enter 256 comma-separated EEG values, use examples above, or upload CSV file",
                lines=3,
                value=""
            )
            
            # Wire up pattern loading buttons
            pattern_a_btn.click(fn=load_pattern_a, outputs=eeg_input)
            pattern_b_btn.click(fn=load_pattern_b, outputs=eeg_input)
            pattern_c_btn.click(fn=load_pattern_c, outputs=eeg_input)
            pattern_d_btn.click(fn=load_pattern_d, outputs=eeg_input)
            pattern_e_btn.click(fn=load_pattern_e, outputs=eeg_input)
            
            analyze_eeg_btn = gr.Button("🧠 Analyze Brain Activity", variant="primary", size="lg")
            
            # Results and Guidelines side by side
            with gr.Row():
                with gr.Column():
                    eeg_result = gr.Markdown()
                with gr.Column():
                    eeg_guidelines = gr.Markdown()
            
            analyze_eeg_btn.click(
                fn=analyze_eeg_signal,
                inputs=[eeg_input, eeg_csv_upload],
                outputs=[eeg_result, eeg_guidelines]
            )
        
        # ===========================
        # Tab 4: Fusion Analysis
        # ===========================
        with gr.Tab("🔬 Step 4: Fusion Analysis"):
            gr.Markdown("### Comprehensive Multi-Modal Stroke Assessment")
            gr.Markdown("*Integrate biomarker, brain imaging, and EEG data for complete diagnosis*")
            
            gr.Markdown("""
            **Primary Analysis:** Brain MRI/CT Scan (REQUIRED) - Image model determines stroke presence and type
            
            **Supporting Analysis:** Biomarker + EEG data (Optional) - Provides additional clinical context
            
            **Benefits:**
            - **Stroke Detection**: Primary diagnosis from brain imaging with AI confidence
            - **Stroke Classification**: If stroke detected, identifies type (Ischemic/Hemorrhagic/Normal)
            - **Grad-CAM Visualization**: Shows which brain regions influenced the AI decision
            - **Multi-Modal Context**: Biomarker risk and EEG monitoring add clinical depth
            - **Integrated Recommendations**: Comprehensive action plan based on all available data
            """)
            
            # Brain Scan Input (REQUIRED)
            gr.Markdown("**🧠 Brain Scan (REQUIRED - Primary Analysis)**")
            fusion_brain_scan = gr.Image(
                label="Upload MRI/CT Scan for Stroke Detection",
                type="pil",
                height=300
            )
            
            # Biomarker Inputs (Optional)
            gr.Markdown("**🩺 Biomarker Data (Optional - Supporting Analysis)**")
            with gr.Row():
                with gr.Column():
                    fusion_gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    fusion_age = gr.Slider(1, 100, value=50, step=1, label="Age (years) - Max: 100")
                    fusion_systolic_bp = gr.Slider(80, 200, value=120, step=1, label="Systolic BP (mmHg) - Max: 200")
                    fusion_diastolic_bp = gr.Slider(40, 130, value=80, step=1, label="Diastolic BP (mmHg) - Max: 130")
                with gr.Column():
                    fusion_heart_disease = gr.Radio([0, 1], label="Heart Disease (0=No, 1=Yes)", value=0)
                    fusion_glucose = gr.Slider(50, 300, value=100, step=1, label="Avg Glucose Level (mg/dL) - Max: 300")
                    fusion_bmi = gr.Slider(10, 60, value=25, step=0.1, label="BMI (kg/m²) - Max: 60")
                    fusion_smoking = gr.Dropdown(
                        ["never smoked", "formerly smoked", "smokes"],
                        label="Smoking Status",
                        value="never smoked"
                    )
            
            # EEG Input (Optional)
            gr.Markdown("**⚡ EEG Signal (Optional - Supporting Analysis)**")
            fusion_eeg_signal = gr.Textbox(
                label="EEG Signal Data (256 samples, comma-separated)",
                placeholder="Enter EEG values or leave empty if not available",
                lines=2
            )
            
            analyze_fusion_btn = gr.Button("🔬 Perform Comprehensive Fusion Analysis", variant="primary", size="lg")
            
            # Results side by side: Analysis Results + Grad-CAM
            with gr.Row():
                with gr.Column():
                    fusion_result = gr.Markdown()
                with gr.Column():
                    fusion_gradcam = gr.Image(label="🔍 Grad-CAM Heatmap Visualization")
            
            # Wire up the fusion analysis
            analyze_fusion_btn.click(
                fn=fusion_analysis_wrapper,
                inputs=[
                    fusion_gender, fusion_age, fusion_systolic_bp, fusion_diastolic_bp,
                    fusion_heart_disease, fusion_glucose, fusion_bmi, fusion_smoking,
                    fusion_brain_scan, fusion_eeg_signal
                ],
                outputs=[fusion_result, fusion_gradcam]
            )


# ===========================
# Launch Application
# ===========================

if __name__ == "__main__":
    print("\n🌐 Launching Gradio interface...")
    print("📍 Access the application at: http://127.0.0.1:7862")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    demo.launch(share=False, server_name="127.0.0.1", server_port=7862)
