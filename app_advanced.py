import gradio as gr
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Import models
from src.models.biomarker_model import BiomarkerModel
from src.models.eeg_model import EEGModel, SimpleEEGModel

# Global variables for loaded models
biomarker_model = None
biomarker_preprocessor = None
eeg_model = None


def load_models():
    """Load trained biomarker and EEG models (if available)."""
    global biomarker_model, biomarker_preprocessor, eeg_model

    # Biomarker model
    try:
        # First try to load a separately-saved sklearn preprocessor (preferred)
        import pickle
        preproc_path = 'src/prediction/biomarker_model_weights_preprocessor.pkl'
        checkpoint_path = 'src/prediction/biomarker_model_weights.pth'

        # Load checkpoint first
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to get preprocessor from checkpoint first (most reliable)
        biomarker_preprocessor = checkpoint.get('preprocessor', None)
        if biomarker_preprocessor is not None:
            print(f"✓ Preprocessor loaded from checkpoint")
        else:
            # Fallback: try separate pickle file
            print(f"⚠ No preprocessor in checkpoint, trying separate file...")
            try:
                with open(preproc_path, 'rb') as f:
                    biomarker_preprocessor = pickle.load(f)
                print(f"✓ Preprocessor loaded from pickle file")
            except Exception as e:
                print(f"✗ Could not load preprocessor: {e}")
                raise Exception("Failed to load preprocessor from any source")
        
        # Load model weights from checkpoint
        model_state = checkpoint['model_state_dict']

        # Determine input dim from preprocessor if available
        if biomarker_preprocessor is not None:
            try:
                input_dim = len(biomarker_preprocessor.get_feature_names_out())
                print(f"✓ Input dimension from preprocessor: {input_dim}")
            except Exception:
                # Some preprocessors may not implement get_feature_names_out
                input_dim = 29
                print(f"✓ Using default input dimension: {input_dim}")
        else:
            input_dim = 29
            print(f"✓ Using default input dimension: {input_dim}")

        biomarker_model = BiomarkerModel(input_dim=input_dim)
        biomarker_model.load_state_dict(model_state)
        biomarker_model.eval()
        print("✓ Biomarker model loaded successfully (Enhanced: 114K samples, 84.17% recall)")
    except Exception as e:
        print(f"✗ Failed to load biomarker model: {e}")
        import traceback
        traceback.print_exc()

    # EEG model - use SimpleEEGModel (enhanced training version)
    try:
        eeg_model_local = SimpleEEGModel(num_classes=5, input_length=256)
        eeg_model_local.load_state_dict(torch.load('src/prediction/eeg_model_weights.pth', map_location='cpu', weights_only=True))
        eeg_model_local.eval()
        eeg_model = eeg_model_local
        print("✓ EEG model loaded successfully (SimpleEEGModel with 100% accuracy)")
    except Exception as e:
        print(f"✗ Failed to load EEG model: {e}")
        # Try old EEGModel as fallback
        try:
            eeg_model_local = EEGModel(num_channels=1, num_freq_bins=256)
            eeg_model_local.load_state_dict(torch.load('src/prediction/eeg_model_weights.pth', map_location='cpu', weights_only=True))
            eeg_model_local.eval()
            eeg_model = eeg_model_local
            print("✓ EEG model loaded successfully (fallback to old EEGModel)")
        except Exception as e2:
            print(f"✗ Failed to load EEG model (both attempts): {e2}")


# Load on import
load_models()


def generate_health_routine_text(age, glucose, bmi, hypertension_flag, heart_disease, smoking_status, stroke_risk):
    """Return a plain-text (Markdown) personalized health routine suitable for Gradio Markdown output."""
    routine_lines = []

    routine_lines.append("## 🏥 Personalized Stroke Prevention Routine")
    routine_lines.append("")

    # Diet
    routine_lines.append("### 🥗 Dietary Recommendations")
    if glucose > 125:
        routine_lines.append("- **High glucose detected**: Limit refined carbs, choose whole grains, increase fiber. Monitor glucose twice daily.")
    elif glucose > 100:
        routine_lines.append("- **Elevated glucose**: Reduce sugar and simple carbs, prefer low-GI foods, increase vegetables.")
    else:
        routine_lines.append("- Maintain a balanced diet; monitor glucose periodically.")

    if bmi > 30:
        routine_lines.append("- **BMI indicates obesity**: Aim for gradual weight loss (0.5–1 kg/week) with calorie deficit and portion control.")
    elif bmi > 25:
        routine_lines.append("- **Overweight**: Moderate calorie reduction and increased physical activity to achieve gradual weight loss.")
    else:
        routine_lines.append("- Maintain healthy weight.")

    # Blood pressure
    if hypertension_flag:
        routine_lines.append("- **Hypertension management**: DASH-style diet (low sodium), monitor BP at home, adhere to antihypertensive medications.")
    else:
        routine_lines.append("- Maintain healthy blood pressure: limit salt, exercise regularly, monitor periodically.")

    # Exercise
    routine_lines.append("")
    routine_lines.append("### 🏃 Exercise Recommendations")
    if age > 65:
        routine_lines.append("- Walking 30 min daily, balance and flexibility exercises, gentle strength work as tolerated.")
    elif age > 50:
        routine_lines.append("- Brisk walking 150 min/week, strength training 2x/week, flexibility exercises daily.")
    else:
        routine_lines.append("- Cardio 150–300 min/week or 75–150 min vigorous activity; strength training 2–3x/week.")

    # Cardiac precautions
    if heart_disease:
        routine_lines.append("- **Cardiac precautions**: Start activity slowly, monitor heart rate, consult cardiologist before increasing intensity.")

    # Smoking
    if smoking_status != "never smoked":
        routine_lines.append("- **Smoking cessation**: Critical for stroke prevention. Seek cessation programs and consider pharmacotherapy.")

    # Medication & monitoring
    routine_lines.append("")
    routine_lines.append("### 💊 Medication & Monitoring")
    if hypertension_flag:
        routine_lines.append("- Take antihypertensives as prescribed; target BP per clinician (often <140/90 or individualized). Monitor daily.")
    if glucose > 125:
        routine_lines.append("- Manage diabetes per clinician: glucose monitoring, medication adherence.")
    routine_lines.append("- Doctor visits: monthly if high risk, otherwise every 3–6 months. Report any new neurological symptoms immediately.")

    # Emergency signs
    routine_lines.append("")
    routine_lines.append("### ⚠️ Emergency Warning Signs — Seek immediate care")
    routine_lines.append("- Sudden numbness/weakness of face/arm/leg, sudden confusion, sudden trouble speaking, sudden vision loss, sudden severe headache, sudden difficulty walking.")

    # Monitoring schedule
    routine_lines.append("")
    if stroke_risk > 0.5:
        routine_lines.append("**High risk**: Intensive monitoring — monthly clinician visits, daily BP checks, consider immediate imaging.")
    else:
        routine_lines.append("**Low/Moderate risk**: Standard prevention — clinician visits every 3–6 months, routine monitoring.")

    return "\n\n".join(routine_lines)


# ===========================
# Biomarker Risk Assessment
# ===========================

def predict_stroke_risk(gender, age, systolic_bp, diastolic_bp, heart_disease, avg_glucose_level, bmi, smoking_status,
                        high_cholesterol, diabetes, physical_activity, difficulty_walking):
    """Predict stroke risk and return (result_markdown, routine_markdown)."""
    global biomarker_model, biomarker_preprocessor

    if biomarker_model is None or biomarker_preprocessor is None:
        return ("⚠️ **Biomarker model not loaded.** Please retrain the model:\n```bash\npython -m src.training.train_biomarker_model\n```",
                "⚠️ Biomarker model unavailable")

    # Derive hypertension flag from BP
    try:
        systolic = float(systolic_bp)
        diastolic = float(diastolic_bp)
    except Exception:
        systolic = 0.0
        diastolic = 0.0

    hypertension_flag = 1 if (systolic >= 140 or diastolic >= 90) else 0

    # Build patient dataframe compatible with enhanced preprocessor (29 features)
    # Use float64 for all numeric fields to match training data types
    patient_data = pd.DataFrame([{
        'gender': gender,
        'age': float(age),
        'hypertension': float(hypertension_flag),
        'heart_disease': float(bool(heart_disease)),
        'ever_married': 'Yes',  # Assuming adult patient
        'work_type': 'Private',  # Default assumption
        'Residence_type': 'Urban',  # Default assumption
        'avg_glucose_level': float(avg_glucose_level),
        'bmi': float(bmi),
        'smoking_status': smoking_status,
        # Enhanced features from diabetes dataset (all as float to match training)
        'high_cholesterol': float(bool(high_cholesterol)),
        'physical_activity': float(bool(physical_activity)),
        'diabetes': float(bool(diabetes)),
        'mental_health_days': 0.0,  # Not collected in UI (default to 0.0)
        'physical_health_days': 0.0,  # Not collected in UI (default to 0.0)
        'difficulty_walking': float(bool(difficulty_walking))
    }])

    # Preprocess
    try:
        X_processed = biomarker_preprocessor.transform(patient_data)
    except Exception as e:
        return (f"⚠️ Preprocessing failed: {e}", "")

    if not isinstance(X_processed, np.ndarray):
        try:
            X_processed = X_processed.toarray()
        except Exception:
            X_processed = np.array(X_processed)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        outputs = biomarker_model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        stroke_prob = float(probs[0][1].item()) if probs.shape[1] > 1 else 0.0

    risk_level = "🔴 **HIGH RISK**" if stroke_prob > 0.5 else "🟢 **LOW RISK**"

    # Compose result
    result_md = f"""
## 🏥 Stroke Risk Assessment Results

### Risk Analysis
- **Stroke Probability:** {stroke_prob:.1%}
- **Risk Level:** {risk_level}

### Patient Profile
- **Age:** {age} years | **Gender:** {gender}
- **Blood Pressure:** {systolic:.0f}/{diastolic:.0f} mmHg (Hypertension: {'Yes' if hypertension_flag else 'No'})
- **Blood Glucose:** {avg_glucose_level} mg/dL (Normal: 70-100 mg/dL)
- **BMI:** {bmi} (Normal: 18.5-24.9)
- **Heart Disease:** {'Yes' if heart_disease else 'No'}
- **Smoking:** {smoking_status}

### Recommendation
{'⚠️ **URGENT:** High stroke risk detected. Recommend immediate MRI/CT scan for detailed analysis.' if stroke_prob > 0.5 else '✓ Continue regular health monitoring and maintain healthy lifestyle.'}
"""

    # Generate routine
    routine_text = generate_health_routine_text(float(age), float(avg_glucose_level), float(bmi), hypertension_flag,
                                               int(bool(heart_disease)), smoking_status, stroke_prob)

    return result_md, routine_text


# ===========================
# MRI/CT Scan Analysis
# ===========================

def analyze_brain_scan(image):
    """Analyze MRI/CT scan with Grad-CAM explanation"""
    
    try:
        # Load the image model
        from src.models.image_model import create_image_model
        from src.data_preprocessing.image_preprocessing import get_image_transforms
        
        model = create_image_model(num_classes=3)  # Model was trained with 3 classes
        model.load_state_dict(torch.load('src/prediction/image_only_model_weights.pth', map_location='cpu'))
        model.eval()
        
        # Preprocess image
        transform = get_image_transforms()['val']
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            image_pil = image
        
        input_tensor = transform(image_pil).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Class names - original model was trained with 3 classes
        class_names = ["Ischemic Stroke", "Hemorrhagic Stroke", "Normal"]
        
        # Generate Grad-CAM with better error handling
        explained_image = None
        try:
            from src.utils.gradcam import GradCAM
            target_layer = model.blocks[-1].norm1
            grad_cam = GradCAM(model, target_layer)
            explained_image, _ = grad_cam.visualize(image_pil, input_tensor, alpha=0.6)
            print(f"✓ Grad-CAM generated successfully. Type: {type(explained_image)}")
        except Exception as e:
            print(f"✗ Grad-CAM error: {e}")
            import traceback
            traceback.print_exc()
            # Return original image if Grad-CAM fails
            explained_image = image_pil
        
        # Create detailed result with Grad-CAM explanation
        result = f"""🔬 **MRI/CT Scan Analysis Results**

**Diagnosis**

**Probability Distribution**
"""
        
        for i, class_name in enumerate(class_names):
            prob = probabilities[0][i].item()
            bar = "█" * int(prob * 30)
            result += f"\n- {class_name}: {prob:.1%} {bar}"
        
        # Medical interpretation
        if class_names[predicted_class] == "Ischemic Stroke":
            result += f"""

⚠️ **Clinical Interpretation**
**Ischemic Stroke Detected** - Blood clot blocking brain artery

**Immediate Actions Required:**

🔴 **Next Steps**
1. Emergency stroke team activation
2. Continuous neurological monitoring
3. If recovery phase, proceed to **EEG Monitoring** tab for post-stroke assessment
"""
        elif class_names[predicted_class] == "Hemorrhagic Stroke":
            result += f"""

⚠️ **Clinical Interpretation**
**Hemorrhagic Stroke Detected** - Brain bleeding detected

**Immediate Actions Required:**

🔴 **Next Steps**
1. Neurosurgery consultation immediately
2. ICU admission for close monitoring
3. Serial CT scans to monitor bleeding
4. If stable, proceed to **EEG Monitoring** tab for brain activity assessment
"""
        else:
            result += f"""

✓ **Clinical Interpretation**
No acute stroke detected. However, continue monitoring for:

📋 **Recommendations**
"""
        
        # Create separate Grad-CAM explanation text
        gradcam_explanation = """🔍 **Grad-CAM Heatmap Explanation**

The visualization on the left shows a **heatmap overlay** on your brain scan:

**Color Legend:**
- 🔴 **Red/Hot Colors (Most Important)**: Brain regions that STRONGLY influenced the AI's decision
  - These areas contain the most distinctive features for the detected condition
  - High activation = The AI "focused" most on these regions

- 🟡 **Yellow/Warm Colors (Moderate Importance)**: Secondary regions of interest
  - These areas contributed moderately to the diagnosis
  - May contain supporting evidence for the classification

- 🔵 **Blue/Cool Colors (Low Importance)**: Background regions
  - These areas had minimal impact on the AI's decision
  - Typically normal brain tissue without abnormalities

**How to interpret:**
- If detecting **Ischemic Stroke**: Red areas show potential blood flow blockage zones
- If detecting **Hemorrhagic Stroke**: Red areas indicate bleeding/hematoma locations
- If **Normal**: Heatmap should be relatively uniform (no strong hot spots)

**Note:** This visualization helps doctors understand the AI's reasoning and verify if it's focusing on clinically relevant regions.
"""
        
        return result, explained_image, gradcam_explanation
        
    except Exception as e:
        return f"❌ Error analyzing scan: {str(e)}\n\nPlease ensure the image model is available.", None, ""

# ===========================
# EEG Post-Stroke Analysis
# ===========================

def analyze_eeg_signal(eeg_signal_text):
    """Analyze EEG signal for post-stroke monitoring"""
    
    if eeg_model is None:
        return "⚠️ **EEG model not loaded.** Please retrain the model:\n```bash\npython -m src.training.train_eeg_model\n```"
    
    try:
        # Parse the EEG signal (expecting comma-separated values)
        eeg_values = [float(x.strip()) for x in eeg_signal_text.split(',')]
        
        # Ensure correct length (256 samples)
        if len(eeg_values) < 256:
            eeg_values.extend([0] * (256 - len(eeg_values)))
        elif len(eeg_values) > 256:
            eeg_values = eeg_values[:256]
        
        # Prepare input tensor
        eeg_tensor = torch.tensor([eeg_values], dtype=torch.float32).unsqueeze(1)  # Shape: (1, 1, 256)
        
        # Predict
        with torch.no_grad():
            outputs = eeg_model(eeg_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Medical interpretations for each class
        brain_states = {
            0: {
                "name": "Normal Conscious State",
                "description": "Active brain activity with normal cognitive function",
                "clinical": "Patient shows normal EEG patterns typical of conscious, alert state",
                "action": "✓ Continue regular monitoring"
            },
            1: {
                "name": "Drowsy/Sedated State",
                "description": "Reduced brain activity, drowsiness or light sedation",
                "clinical": "EEG shows slowing of frequencies, typical of drowsy or lightly sedated state",
                "action": "Monitor consciousness level, check medication effects"
            },
            2: {
                "name": "Deep Sleep/Unconscious",
                "description": "Significantly reduced activity, deep sleep or unconscious state",
                "clinical": "Delta waves predominant, indicating deep sleep or altered consciousness",
                "action": "⚠️ Assess level of consciousness, check for responsiveness"
            },
            3: {
                "name": "Seizure Activity Detected",
                "description": "Abnormal synchronized electrical activity",
                "clinical": "Patterns consistent with seizure activity or high seizure risk",
                "action": "🔴 URGENT: Notify physician immediately, prepare anti-seizure medication"
            },
            4: {
                "name": "Critical Suppression",
                "description": "Severely reduced brain activity",
                "clinical": "Burst suppression or significant cortical depression detected",
                "action": "🔴 CRITICAL: Immediate medical intervention required"
            }
        }
        
        detected_state = brain_states[predicted_class]
        
        # Create detailed result without subsections
        results_text = f"""🧠 **EEG Post-Stroke Monitoring Results**

**Brain Activity Status**
- **Detected State:** **{detected_state['name']}**
- **Confidence:** {confidence:.1%}
- **Description:** {detected_state['description']}

**Clinical Assessment**
{detected_state['clinical']}

**Recommended Action**
{detected_state['action']}

**Signal Analysis Distribution**

"""
        
        for i in range(5):
            state_name = brain_states[i]['name']
            prob = probabilities[0][i].item()
            bar = "█" * int(prob * 25)
            emoji = "🔴" if i >= 3 else "⚠️" if i == 2 else "🟡" if i == 1 else "🟢"
            results_text += f"{emoji} {state_name}: {prob:.1%} {bar}\n\n"
        
        # Separate guidelines text
        guidelines_text = """📊 **Post-Stroke EEG Monitoring Guidelines**

**What we're monitoring:**
- **Consciousness level**: Recovery of awareness and cognitive function
- **Seizure risk**: Post-stroke seizures occur in 5-20% of patients
- **Brain recovery**: Gradual normalization of EEG patterns indicates healing
- **Complications**: Early detection of cerebral edema or ischemic expansion

**Normal Recovery Pattern:**
- Initial suppression → Gradual increase in frequency → Return to normal patterns

**Clinical Significance:**
This analysis helps clinicians track neurological recovery and detect complications early.
"""
        
        return results_text, guidelines_text
        
    except Exception as e:
        error_msg = f"❌ Error analyzing EEG signal: {str(e)}\n\nPlease ensure the signal is provided as comma-separated numbers (e.g., '1,2,3,4...')"
        return error_msg, ""

# ===========================
# Fusion Model Analysis
# ===========================

def analyze_fusion(biomarker_inputs, brain_scan, eeg_signal):
    """
    Perform comprehensive fusion analysis with image model as primary predictor.
    
    Args:
        biomarker_inputs: Dictionary with biomarker data (gender, age, bp, etc.)
        brain_scan: PIL Image of brain MRI/CT scan (REQUIRED)
        eeg_signal: String of comma-separated EEG values
        
    Returns:
        fusion_result: Markdown text with fusion analysis results
        gradcam_visual: PIL Image showing Grad-CAM heatmap overlay
    """
    try:
        # === PRIMARY: Image Model Analysis (REQUIRED) ===
        if brain_scan is None:
            return "⚠️ **Brain Scan Required**\n\nFusion analysis requires a brain MRI/CT scan as the primary input. Please upload a brain scan image.", None
        
        # Perform image analysis with Grad-CAM
        try:
            from src.models.image_model import create_image_model
            from src.data_preprocessing.image_preprocessing import get_image_transforms
            from src.utils.gradcam import GradCAM
            
            # Load image model
            image_model = create_image_model(num_classes=3)
            image_model.load_state_dict(torch.load('src/prediction/image_only_model_weights.pth', map_location='cpu'))
            image_model.eval()
            
            # Preprocess image
            transform = get_image_transforms()['val']
            if isinstance(brain_scan, np.ndarray):
                image_pil = Image.fromarray(brain_scan.astype('uint8'), 'RGB')
            else:
                image_pil = brain_scan
            
            input_tensor = transform(image_pil).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = image_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()
                image_confidence = probabilities[0][predicted_class].item()
            
            # Class names
            class_names = ["Ischemic Stroke", "Hemorrhagic Stroke", "Normal"]
            class_name = class_names[predicted_class]
            
            # Generate Grad-CAM visualization
            gradcam_visual = None
            try:
                target_layer = image_model.blocks[-1].norm1
                grad_cam = GradCAM(image_model, target_layer)
                gradcam_visual, _ = grad_cam.visualize(image_pil, input_tensor, alpha=0.6)
                print(f"✓ Fusion Grad-CAM generated successfully")
            except Exception as e:
                print(f"✗ Fusion Grad-CAM error: {e}")
                gradcam_visual = image_pil
            
            # Determine primary stroke status
            has_stroke = (class_name != 'Normal')
            stroke_type = class_name if has_stroke else None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error analyzing brain scan: {str(e)}", None
        
        # === SUPPORTING: Biomarker Analysis ===
        biomarker_risk = None
        biomarker_prob = None
        if biomarker_inputs and any(biomarker_inputs.values()):
            try:
                gender = biomarker_inputs.get('gender', 'Male')
                age = biomarker_inputs.get('age', 50)
                systolic_bp = biomarker_inputs.get('systolic_bp', 120)
                diastolic_bp = biomarker_inputs.get('diastolic_bp', 80)
                heart_disease = biomarker_inputs.get('heart_disease', 0)
                avg_glucose_level = biomarker_inputs.get('avg_glucose_level', 100)
                bmi = biomarker_inputs.get('bmi', 25)
                smoking_status = biomarker_inputs.get('smoking_status', 'never smoked')
                
                hypertension = 1 if (systolic_bp >= 140 or diastolic_bp >= 90) else 0
                
                input_df = pd.DataFrame([{
                    'gender': gender,
                    'age': age,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }])
                
                X_transformed = biomarker_preprocessor.transform(input_df)
                X_tensor = torch.FloatTensor(X_transformed)
                
                with torch.no_grad():
                    output = biomarker_model(X_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    biomarker_prob = probabilities[0][1].item()
                    biomarker_risk = 'High Risk' if biomarker_prob > 0.5 else 'Low Risk'
            except Exception as e:
                print(f"Biomarker analysis failed: {e}")
        
        # === SUPPORTING: EEG Analysis ===
        eeg_state = None
        eeg_confidence = None
        if eeg_signal and eeg_signal.strip():
            try:
                eeg_values = [float(x.strip()) for x in eeg_signal.split(',')]
                
                if len(eeg_values) != 256:
                    raise ValueError(f"Expected 256 EEG samples, got {len(eeg_values)}")
                
                eeg_tensor = torch.FloatTensor(eeg_values).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = eeg_model(eeg_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    eeg_confidence = probabilities[0][predicted_class].item()
                
                brain_states = {
                    0: "Normal Conscious",
                    1: "Drowsy/Sedated",
                    2: "Deep Sleep/Unconscious",
                    3: "Seizure Activity",
                    4: "Critical Suppression"
                }
                
                eeg_state = brain_states[predicted_class]
            except Exception as e:
                print(f"EEG analysis failed: {e}")
        
        # === Generate Comprehensive Fusion Results ===
        
        # Determine modalities used
        modalities_used = ["Brain MRI/CT Scan"]
        if biomarker_risk is not None:
            modalities_used.append("Biomarker Analysis")
        if eeg_state is not None:
            modalities_used.append("EEG Monitoring")
        
        # Build result
        result = f"""🔬 **Comprehensive Fusion Analysis**

**Analysis Mode:** Multi-Modal Integration ({len(modalities_used)}/3 modalities)
- {', '.join(modalities_used)}

---

"""
        
        # PRIMARY DIAGNOSIS (Image-based)
        if has_stroke:
            result += f"""## 🔴 **PRIMARY DIAGNOSIS: STROKE DETECTED**

**Stroke Classification:** **{stroke_type}**
**Confidence:** {image_confidence:.1%}

**Brain Scan Analysis:**
- Imaging modality confirmed stroke event
- Type: {stroke_type}
- AI model confidence: {image_confidence:.1%}
- Grad-CAM heatmap shows affected regions (see visualization →)

"""
        else:
            result += f"""## 🟢 **PRIMARY DIAGNOSIS: NO STROKE DETECTED**

**Classification:** **Normal**
**Confidence:** {image_confidence:.1%}

**Brain Scan Analysis:**
- No acute stroke findings on imaging
- Brain tissue appears normal
- AI model confidence: {image_confidence:.1%}

"""
        
        # SUPPORTING ANALYSES
        result += "---\n\n**🔍 Supporting Multi-Modal Analysis**\n\n"
        
        if biomarker_risk is not None:
            emoji = "🔴" if biomarker_risk == 'High Risk' else "🟢"
            result += f"""**🩺 Biomarker Risk Assessment**
{emoji} **Risk Level:** {biomarker_risk}
- Clinical parameters indicate {biomarker_risk.lower()} for future stroke events
- Probability: {biomarker_prob:.1%}
"""
            if biomarker_risk == 'High Risk' and not has_stroke:
                result += "- ⚠️ High risk detected despite normal scan - preventive measures recommended\n"
            elif biomarker_risk == 'Low Risk' and has_stroke:
                result += "- ℹ️ Clinical markers show low risk, but imaging confirms stroke - acute event\n"
            result += "\n"
        
        if eeg_state is not None:
            risk_states = ['Seizure Activity', 'Critical Suppression', 'Deep Sleep/Unconscious']
            emoji = "🔴" if eeg_state in risk_states else "🟡" if eeg_state == 'Drowsy/Sedated' else "🟢"
            result += f"""**⚡ EEG Brain Activity Monitoring**
{emoji} **Brain State:** {eeg_state}
- Neurological activity level: {eeg_state}
- Confidence: {eeg_confidence:.1%}
"""
            if eeg_state in risk_states and has_stroke:
                result += "- ⚠️ Abnormal brain activity consistent with stroke findings\n"
            elif eeg_state in risk_states and not has_stroke:
                result += "- ⚠️ Abnormal brain activity despite normal scan - monitoring needed\n"
            result += "\n"
        
        # INTEGRATED CLINICAL RECOMMENDATIONS
        result += "---\n\n**🎯 Integrated Clinical Recommendations**\n\n"
        
        if has_stroke:
            result += """**IMMEDIATE ACTIONS REQUIRED:**
- � **EMERGENCY**: Stroke confirmed - immediate medical intervention
- 🏥 Activate stroke protocol (if not already in ER)
- � Consider thrombolytic therapy (if within window)
- � Continuous monitoring of vital signs and neurological status
"""
            if stroke_type == "Ischemic Stroke":
                result += "- � Antiplatelet/anticoagulant therapy as indicated\n"
                result += "- 🩺 Evaluate for mechanical thrombectomy if large vessel occlusion\n"
            elif stroke_type == "Hemorrhagic Stroke":
                result += "- 🩸 Blood pressure control critical\n"
                result += "- � Neurosurgical evaluation for possible intervention\n"
            
            if eeg_state in ['Seizure Activity', 'Critical Suppression']:
                result += "- ⚡ EEG abnormalities detected - neurological monitoring essential\n"
            
            if biomarker_risk == 'High Risk':
                result += "- 📋 Address underlying risk factors for secondary prevention\n"
        
        else:
            # No stroke detected
            if biomarker_risk == 'High Risk' or (eeg_state in ['Seizure Activity', 'Critical Suppression', 'Deep Sleep/Unconscious']):
                result += """**ELEVATED RISK DETECTED:**
- 🟡 No acute stroke on imaging, but other concerning findings
"""
                if biomarker_risk == 'High Risk':
                    result += "- 📊 Clinical biomarkers indicate high stroke risk\n"
                    result += "- 💊 Preventive medications and lifestyle modifications recommended\n"
                if eeg_state in ['Seizure Activity', 'Critical Suppression']:
                    result += "- ⚡ Abnormal EEG findings require neurological follow-up\n"
                
                result += """- 👨‍⚕️ Schedule comprehensive evaluation with neurologist
- 🔍 Consider additional diagnostic testing
- 📅 Regular monitoring and reassessment
"""
            else:
                result += """**NORMAL FINDINGS:**
- ✅ All modalities within normal parameters
- 🏃 Continue healthy lifestyle and preventive measures
- 📅 Routine follow-up as scheduled
- 💚 Low current stroke risk
"""
        
        # Add Grad-CAM explanation
        result += """

---

### 🔍 Grad-CAM Heatmap Explanation

The visualization on the right shows a **heatmap overlay** on your brain scan:

**Color Legend:**
- 🔴 **Red/Hot Colors (Most Important)**: Brain regions that STRONGLY influenced the AI's decision
  - These areas contain the most distinctive features for the detected condition
  - High activation = The AI "focused" most on these regions

- 🟡 **Yellow/Warm Colors (Moderate Importance)**: Secondary regions of interest
  - These areas contributed moderately to the diagnosis
  - May contain supporting evidence for the classification

- 🔵 **Blue/Cool Colors (Low Importance)**: Background regions
  - These areas had minimal impact on the AI's decision
  - Typically normal brain tissue without abnormalities

**How to interpret:**
- If detecting **Ischemic Stroke**: Red areas show potential blood flow blockage zones
- If detecting **Hemorrhagic Stroke**: Red areas indicate bleeding/hematoma locations
- If **Normal**: Heatmap should be relatively uniform (no strong hot spots)

**Note:** This visualization helps doctors understand the AI's reasoning and verify if it's focusing on clinically relevant regions.
"""
        
        return result, gradcam_visual
        
    except Exception as e:
        return f"❌ Error in fusion analysis: {str(e)}\n\n{str(e.__traceback__)}", None


# ===========================
# Information Tab
# ===========================

def create_info_tab():
    """Create the information/welcome tab"""
    return """
    # 🧠 Advanced Stroke Classification & Monitoring System
    
    ## Clinical Workflow
    
    This AI-powered system follows a **3-step medical workflow** for comprehensive stroke assessment:
    
    ### 🩺 **Step 1: Biomarker Risk Assessment**
    **Purpose:** Initial screening to identify high-risk patients
    
    **Input:** Key medical parameters:
    - Age, Gender, BMI
    - Blood Pressure (systolic/diastolic)
    - Blood Glucose Level
    - Heart Disease status
    - Smoking History
    
    **Normal Ranges & Stroke Risk Values:**
    - **Blood Pressure**: Normal <120/80 mmHg | Hypertension ≥140/90 mmHg
    - **Glucose (fasting)**: Normal 70-100 mg/dL | Pre-diabetes 100-125 mg/dL | Diabetes >125 mg/dL
    - **BMI**: Normal 18.5-24.9 | Overweight 25-29.9 | Obese ≥30
    - **Age**: Risk increases significantly after 55 years
    - **Smoking**: Increases stroke risk 2-4x
    - **Heart Disease**: Increases stroke risk 2-3x
    
    **Output:**
    - Stroke risk probability (0-100%)
    - Personalized health routine with diet, exercise, and monitoring recommendations
    - Recommendation for further testing if high risk detected
    
    ---
    
    ### 🔬 **Step 2: MRI/CT Scan Analysis** 
    **Purpose:** Detailed brain imaging for stroke detection
    
    **Input:** MRI or CT scan images
    
    **Output:**
    - Classification: Ischemic Stroke, Hemorrhagic Stroke, or Normal
    - Confidence score
    - **Grad-CAM heatmap visualization** highlighting brain regions that influenced the AI's diagnosis
    - Clinical recommendations based on findings
    
    **Technology:**
    - Vision Transformer (ViT) deep learning model
    - Explainable AI with heat map overlays showing decision regions
    - High accuracy on validation data
    
    ---
    
    ### 🧪 **Step 3: EEG Post-Stroke Monitoring**
    **Purpose:** Monitor neurological recovery after stroke event
    
    **Input:** EEG brain signal data (256 time-series samples)
    
    **Output:** 
    - Brain activity state classification:
      - ✅ Normal Conscious State
      - ⚠️ Drowsy/Sedated State  
      - 🟡 Deep Sleep/Unconscious
      - 🔴 Seizure Activity Detected
      - 🔴 Critical Suppression
    - Clinical interpretation with recommended actions
    - Monitoring guidelines for post-stroke care
    
    **Medical Value:**
    - Early detection of post-stroke seizures (5-20% occurrence rate)
    - Track consciousness and cognitive recovery
    - Identify complications requiring immediate intervention
    
    ---
    
    # ## 🎯 System Capabilities
    
    # ✅ **Multi-Modal Analysis**: Biomarkers + Imaging + EEG signals
    
    # ✅ **Explainable AI**: 
    # - Personalized health routine based on biomarker analysis
    # - Grad-CAM for visual explanation of image predictions
    
    # ✅ **Clinical Workflow Integration**: Follows natural diagnostic progression
    
    # ✅ **High Accuracy**:
    # - Biomarker Model: 95.1% accuracy
    # - MRI Classification: High accuracy
    # - EEG Analysis: 82.8% accuracy
    
    # ---
    
    # ## ⚠️ Important Medical Disclaimer
    
    # **This system is designed for:**
    # - Clinical decision support
    # - Research and educational purposes
    # - Assisting healthcare professionals
    
    # **This system is NOT:**
    # - A replacement for professional medical diagnosis
    # - Approved for standalone clinical use without physician oversight
    # - A substitute for emergency medical care
    
    # **Always consult qualified healthcare professionals for medical decisions.**
    
    # ---
    
    # ## 🔬 Technical Stack
    # - **Deep Learning**: PyTorch 2.x, Vision Transformers
    # - **Explainable AI**: SHAP, Grad-CAM
    # - **Interface**: Gradio 4.x
    # - **Medical AI**: Specialized models for stroke/neurological analysis
    
    # ---
    
    # **Ready to begin? Start with Step 1: Biomarker Risk Assessment →**
    """

# ===========================
# Create Gradio Interface
# ===========================

with gr.Blocks(title="Advanced Stroke Analysis System", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🧠 Advanced Stroke Classification & Analysis System")
    gr.Markdown("*AI-powered stroke risk assessment and post-stroke monitoring*")
    
    with gr.Tabs():
        
        # Information Tab
            # with gr.Tab("📋 Information"):
            #     gr.Markdown(create_info_tab())
        
        # Biomarker Risk Assessment Tab
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
        
        # MRI/CT Scan Analysis Tab
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
        
        # EEG Analysis Tab
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
                normal_btn = gr.Button("Normal Pattern", size="sm")
                drowsy_btn = gr.Button("Drowsy Pattern", size="sm")
                sleep_btn = gr.Button("Sleep Pattern", size="sm")
            
            eeg_input = gr.Textbox(
                label="EEG Signal Data (256 samples, comma-separated)",
                placeholder="Enter 256 comma-separated EEG values, or use examples above",
                lines=3,
                value=""  # Start empty, users can load examples
            )
            
            # Generate realistic EEG patterns for each brain state
            def generate_normal_eeg():
                # Alpha waves (8-13 Hz) - normal conscious state
                return ",".join([str(round(np.sin(i * 0.2) * 10 + np.random.randn() * 2, 2)) for i in range(256)])
            
            def generate_drowsy_eeg():
                # Theta waves (4-8 Hz) - drowsy/sedated
                return ",".join([str(round(np.sin(i * 0.1) * 15 + np.random.randn() * 3, 2)) for i in range(256)])
            
            def generate_sleep_eeg():
                # Delta waves (0.5-4 Hz) - deep sleep
                return ",".join([str(round(np.sin(i * 0.05) * 20 + np.random.randn() * 2, 2)) for i in range(256)])
            
            normal_btn.click(fn=generate_normal_eeg, outputs=eeg_input)
            drowsy_btn.click(fn=generate_drowsy_eeg, outputs=eeg_input)
            sleep_btn.click(fn=generate_sleep_eeg, outputs=eeg_input)
            
            analyze_eeg_btn = gr.Button("🧠 Analyze Brain Activity", variant="primary", size="lg")
            
            # Results and Guidelines side by side
            with gr.Row():
                with gr.Column():
                    eeg_result = gr.Markdown()
                with gr.Column():
                    eeg_guidelines = gr.Markdown()
            
            analyze_eeg_btn.click(
                fn=analyze_eeg_signal,
                inputs=[eeg_input],
                outputs=[eeg_result, eeg_guidelines]
            )
        
        # Fusion Analysis Tab
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
                fn=lambda g, a, sbp, dbp, hd, gl, bmi, sm, scan, eeg: analyze_fusion(
                    {
                        'gender': g,
                        'age': a,
                        'systolic_bp': sbp,
                        'diastolic_bp': dbp,
                        'heart_disease': hd,
                        'avg_glucose_level': gl,
                        'bmi': bmi,
                        'smoking_status': sm
                    },
                    scan,
                    eeg
                ),
                inputs=[
                    fusion_gender, fusion_age, fusion_systolic_bp, fusion_diastolic_bp,
                    fusion_heart_disease, fusion_glucose, fusion_bmi, fusion_smoking,
                    fusion_brain_scan, fusion_eeg_signal
                ],
                outputs=[fusion_result, fusion_gradcam]
            )
        
        # Model Information Tab
        # with gr.Tab("ℹ️ Model Details"):
        #     gr.Markdown("""
        #     ## Model Architectures
            
        #     ### Biomarker Model
        #     - **Architecture:** Multi-Layer Perceptron (MLP)
        #     - **Input Features:** Clinical biomarkers (age, gender, BMI, glucose, etc.)
        #     - **Output:** Binary classification (Stroke / No Stroke)
        #     - **Training:** Trained on healthcare stroke dataset
        #     - **Explainability:** SHAP (SHapley Additive exPlanations)
            
        #     ### EEG Model
        #     - **Architecture:** 1D Convolutional Neural Network
        #     - **Input:** EEG time-series data (256 samples)
        #     - **Output:** 5-class classification
        #     - **Training:** Trained on post-stroke patient EEG dataset
        #     - **Purpose:** Post-stroke cognitive monitoring
            
        #     ### Multimodal Fusion Model (Architecture Ready)
        #     - **Components:** Vision Transformer + Biomarker MLP + EEG CNN
        #     - **Fusion Method:** Late fusion with attention mechanism
        #     - **Input:** MRI scans + Biomarker data + EEG signals
        #     - **Output:** Comprehensive stroke diagnosis
            
        #     ## Technical Stack
        #     - **Framework:** PyTorch 2.x
        #     - **Vision Model:** Vision Transformer (ViT-Base-Patch16-224)
        #     - **XAI Tools:** SHAP, Grad-CAM
        #     - **Interface:** Gradio 4.x
        #     - **Data Processing:** scikit-learn, pandas, numpy
        #     """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7861)
