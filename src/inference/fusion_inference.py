import torch
import pandas as pd
import numpy as np
from PIL import Image
from src.inference.model_loader import model_loader


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
            class_names = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]
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
        biomarker_model_obj = model_loader.biomarker_model
        biomarker_preprocessor_obj = model_loader.biomarker_preprocessor
        
        if biomarker_inputs and any(biomarker_inputs.values()) and biomarker_model_obj is not None and biomarker_preprocessor_obj is not None:
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
                
                X_transformed = biomarker_preprocessor_obj.transform(input_df)
                X_tensor = torch.FloatTensor(X_transformed)
                
                with torch.no_grad():
                    output = biomarker_model_obj(X_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    biomarker_prob = probabilities[0][1].item()
                    biomarker_risk = 'High Risk' if biomarker_prob > 0.5 else 'Low Risk'
            except Exception as e:
                print(f"Biomarker analysis failed: {e}")
        
        # === SUPPORTING: EEG Analysis ===
        eeg_state = None
        eeg_confidence = None
        eeg_model_obj = model_loader.eeg_model
        
        if eeg_signal and eeg_signal.strip() and eeg_model_obj is not None:
            try:
                eeg_values = [float(x.strip()) for x in eeg_signal.split(',')]
                
                if len(eeg_values) != 256:
                    raise ValueError(f"Expected 256 EEG samples, got {len(eeg_values)}")
                
                eeg_tensor = torch.FloatTensor(eeg_values).unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = eeg_model_obj(eeg_tensor)
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
- 🚨 **EMERGENCY**: Stroke confirmed - immediate medical intervention
- 🏥 Activate stroke protocol (if not already in ER)
- 💉 Consider thrombolytic therapy (if within window)
- 📊 Continuous monitoring of vital signs and neurological status
"""
            if stroke_type == "Ischemic Stroke":
                result += "- 💊 Antiplatelet/anticoagulant therapy as indicated\n"
                result += "- 🩺 Evaluate for mechanical thrombectomy if large vessel occlusion\n"
            elif stroke_type == "Hemorrhagic Stroke":
                result += "- 🩸 Blood pressure control critical\n"
                result += "- 🔪 Neurosurgical evaluation for possible intervention\n"
            
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
        import traceback
        traceback.print_exc()
        return f"❌ Error in fusion analysis: {str(e)}", None
