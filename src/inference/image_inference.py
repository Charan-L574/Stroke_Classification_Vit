import torch
import numpy as np
from PIL import Image


def analyze_brain_scan(image):
    """Analyze MRI/CT scan with Grad-CAM explanation"""
    
    try:
        # Load the image model
        from src.models.image_model import create_image_model
        from src.data_preprocessing.image_preprocessing import get_image_transforms
        
        model = create_image_model(num_classes=3)
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
        
        # Class names
        class_names = ["Hemorrhagic Stroke", "Ischemic Stroke", "Normal"]
        
        # Generate Grad-CAM
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
            explained_image = image_pil
        
        # Create detailed result
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
        
        # Grad-CAM explanation text
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
