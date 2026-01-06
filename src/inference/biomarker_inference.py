import torch
import pandas as pd
import numpy as np
from src.inference.model_loader import model_loader


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


def predict_stroke_risk(gender, age, systolic_bp, diastolic_bp, heart_disease, avg_glucose_level, bmi, smoking_status,
                        high_cholesterol, diabetes, physical_activity, difficulty_walking):
    """Predict stroke risk and return (result_markdown, routine_markdown)."""
    biomarker_model = model_loader.biomarker_model
    biomarker_preprocessor = model_loader.biomarker_preprocessor

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
    patient_data = pd.DataFrame([{
        'gender': gender,
        'age': float(age),
        'hypertension': float(hypertension_flag),
        'heart_disease': float(bool(heart_disease)),
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': float(avg_glucose_level),
        'bmi': float(bmi),
        'smoking_status': smoking_status,
        'high_cholesterol': float(bool(high_cholesterol)),
        'physical_activity': float(bool(physical_activity)),
        'diabetes': float(bool(diabetes)),
        'mental_health_days': 0.0,
        'physical_health_days': 0.0,
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
