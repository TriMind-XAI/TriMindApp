import torch
import numpy as np
import streamlit as st
from captum.attr import IntegratedGradients, Saliency, GradientShap, LayerGradCam
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import json

from utils import get_class_name


#@title XAI Explainer Class
class MultiviewExplainer:
    """Generate multiple XAI explanations"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Initialize XAI methods
        self.ig = IntegratedGradients(self.model)
        self.saliency = Saliency(self.model)
        self.gradient_shap = GradientShap(self.model)
        # Note: self.model.conv3 might need adjustment based on the actual model architecture
        # For ResNet8, the layer is 'layer3' in the forward pass, so a better choice might be model.layer3.conv2 or similar
        # For now, let's assume `conv3` is a placeholder or adjust if needed during execution.
        if st.session_state["model_name"]=="ResNet-8⭐":
            self.gradcam = LayerGradCam(self.model, self.model.layer3.conv2)
        else:
            self.gradcam = LayerGradCam(self.model, self.model.conv2)

    def explain(self, input_image, target_class=None):
        """Generate all explanations"""
        input_image = input_image.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        print(f"\n Generating explanations...")
        print(f"   Predicted: Class {predicted_class} (confidence: {confidence:.2%})")

        results = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'attributions': {}
        }

        # 1. Integrated Gradients
        print("   → Running Integrated Gradients...")
        ig_attr = self.ig.attribute(input_image, target=target_class, n_steps=50)
        results['attributions']['integrated_gradients'] = ig_attr.squeeze().detach().cpu().numpy()

        # 2. Saliency Maps
        print("   → Running Saliency...")
        saliency_attr = self.saliency.attribute(input_image, target=target_class)
        results['attributions']['saliency'] = np.abs(saliency_attr.squeeze().detach().cpu().numpy())

        # 3. GradientSHAP
        print("   → Running GradientSHAP...")
        baseline = torch.zeros_like(input_image)
        gradient_shap_attr = self.gradient_shap.attribute(
            input_image, baselines=baseline, target=target_class, n_samples=50
        )
        results['attributions']['gradient_shap'] = gradient_shap_attr.squeeze().detach().cpu().numpy()

        # 4. GradCAM
        print("   → Running GradCAM...")
        gradcam_attr = self.gradcam.attribute(input_image, target=target_class)
        gradcam_upsampled = torch.nn.functional.interpolate(
            gradcam_attr, size=(28, 28), mode='bilinear', align_corners=False
        )
        results['attributions']['gradcam'] = gradcam_upsampled.squeeze().detach().cpu().numpy()

        print("✅ All explanations generated!")
        return results
    def visualize_explanations(self, input_image, results):
        """Visualize all XAI methods"""
        
        plt.style.use("dark_background")
        attributions = results['attributions']
        n_methods = len(attributions)

        fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 8))

        original_img = input_image.squeeze().cpu().numpy()

        for idx, (method_name, attr) in enumerate(attributions.items()):

            # Top row: heatmap
            im = axes[0, idx].imshow(attr, cmap='jet')
            axes[0, idx].set_title(f'{method_name.replace("_"," ").title()} Heatmap')
            axes[0, idx].axis('off')

            fig.colorbar(im, ax=axes[0, idx], fraction=0.046)

            # Bottom row: overlay
            axes[1, idx].imshow(original_img, cmap='gray', alpha=0.7)
            axes[1, idx].imshow(attr, cmap='jet', alpha=0.5)
            axes[1, idx].set_title("Overlay")
            axes[1, idx].axis('off')

        fig.suptitle(
            f'Prediction: Class {results["prediction"]} '
            f'(Confidence: {results["confidence"]:.2%})',
            fontsize=14,
            fontweight='bold'
        )

        plt.tight_layout()
        st.pyplot(fig,width='stretch')

def analyze_consistency(results):
    """Analyze consistency across XAI methods"""


    attributions = results['attributions']
    methods = list(attributions.keys())

    # Flatten attributions
    flat_attrs = {method: attr.flatten() for method, attr in attributions.items()}

    # Calculate pairwise correlations
    correlations = {}
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            corr, _ = spearmanr(flat_attrs[method1], flat_attrs[method2])
            correlations[f"{method1}_vs_{method2}"] = corr

    avg_correlation = np.mean(list(correlations.values()))

    print(f"\n📊 Consistency Analysis:")
    print(f"   Average correlation: {avg_correlation:.3f}")
    for pair, corr in correlations.items():
        print(f"   {pair}: {corr:.3f}")

    return {
        'correlations': correlations,
        'average_correlation': avg_correlation
    }
def generate_narrative_hf(client, results, consistency_metrics, audience='Clinician'):
    pred_class = get_class_name(results["prediction"])

    context_data = {
        "diagnosis": pred_class,
        "confidence": f"{results['confidence']:.2%}",
        "xai_consistency": f"{consistency_metrics['average_correlation']:.3f}",
        "method_details": {
            "GradCAM": "Regional localization",
            "IntegratedGradients": "Pixel-level attribution",
            "Saliency": "Edge/Density sensitivity",
            "GradientSHAP": "Robust feature importance"
        }
    }

    personas = {
        'Clinician': (
            f"You are a Radiologist analysing {st.session_state['selected_dataset']} classification results."
            "Write a brief clinical finding report using terms like consolidation, opacity, and feature attribution."
            "Describe which parts of the image influenced the prediction and how strongly they contributed."
        ),
        'Researcher': (
            f"You are an AI auditor evaluating a {st.session_state['selected_dataset']} image classification model. "
            "Analyze which regions influenced the prediction, assess attention strength, "
            "and discuss reliability based only on the provided data."
        ),
        'Patient': (
            f"You are a doctor explaining {st.session_state['selected_dataset']} image classificaiton results to a patient."
            "Explain the results simply, avoid jargon, and be reassuring."
        )
    }


    user_message = f"""
        DATA TO ANALYZE:
        {json.dumps(context_data, indent=2)}

        INSTRUCTIONS:
        - Reference the confidence score and XAI consistency.
        - If consistency > 0.6, state that the explanation is reliable.
        - If consistency is low, mention possible uncertainty.
        - Explain which regions of the image MOST influenced the prediction.
        - Use GradCAM as the primary source for localization (where the model focused).
        - Refer to attention strength and whether it is concentrated or spread out.
        - Avoid describing XAI methods themselves (no generic definitions).
        - Focus only on what the model used to make the decision.
        - Do NOT make unsupported medical diagnoses.
        - Keep response to 2 short paragraphs.
    """
    response = client.chat_completion(
        messages=[
            {"role": "system", "content":  personas[audience]},
            {"role": "user", "content": user_message}
        ],
        max_tokens=400,
        temperature=0.5,
    )

    return response.choices[0].message.content