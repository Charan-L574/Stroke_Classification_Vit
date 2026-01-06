import torch
import pandas as pd
from src.inference.model_loader import model_loader


def analyze_eeg_signal(eeg_signal_text, eeg_csv_file=None):
    """
    Analyze EEG signal for post-stroke monitoring.
    
    Args:
        eeg_signal_text: String of comma-separated EEG values (256 samples)
        eeg_csv_file: Optional CSV file with EEG data
        
    Returns:
        results_text: Markdown text with analysis results
        guidelines_text: Markdown text with clinical guidelines
    """
    eeg_model = model_loader.eeg_model
    
    if eeg_model is None:
        return "⚠️ **EEG model not loaded.** Please retrain the model:\n```bash\npython -m src.training.train_eeg_model\n```", ""
    
    try:
        # Check if CSV file was uploaded
        if eeg_csv_file is not None:
            try:
                df = pd.read_csv(eeg_csv_file)
                if df.shape[0] > 0 and df.shape[1] > 0:
                    eeg_values = df.iloc[:, 0].values.flatten().tolist()
                else:
                    raise ValueError("CSV file is empty")
                eeg_values = [float(x) for x in eeg_values if pd.notna(x)]
                print(f"✓ Loaded {len(eeg_values)} EEG values from CSV")
            except Exception as e:
                return f"❌ Error reading CSV file: {str(e)}\n\nPlease ensure the CSV contains numerical EEG values in the first column.", ""
        else:
            if not eeg_signal_text or not eeg_signal_text.strip():
                return "⚠️ **No EEG data provided.** Please enter EEG values, upload a CSV file, or use the sample buttons above.", ""
            eeg_values = [float(x.strip()) for x in eeg_signal_text.split(',')]
        
        # Ensure correct length (256 samples)
        if len(eeg_values) < 256:
            eeg_values.extend([0] * (256 - len(eeg_values)))
        elif len(eeg_values) > 256:
            eeg_values = eeg_values[:256]
        
        # Prepare input tensor
        eeg_tensor = torch.tensor([eeg_values], dtype=torch.float32).unsqueeze(1)
        
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
        
        # Create detailed result
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
