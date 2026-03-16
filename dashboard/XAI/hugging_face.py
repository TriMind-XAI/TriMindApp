from huggingface_hub import InferenceClient
import os
import json
import streamlit as st

from utils import get_class_name

def setup_huggingface(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    # print("🚀 Connecting to Hugging Face Inference API...")

    client = InferenceClient(
        model=model_name,
        token=st.secrets["HF_TOKEN"]
    )

    print(f"✅ Connected to model: {model_name}")
    return client

def generate_narrative_hf(client, results, consistency_metrics, audience='clinician'):
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
        'clinician': "You are a Radiologist. Write a brief clinical finding report using terms like consolidation, opacity, and feature attribution.",
        'patient': "You are a caring doctor. Explain the results simply, avoid jargon, and be reassuring.",
        'researcher': "You are an AI Auditor. Critique reliability using the XAI consistency score and discuss potential bias."
    }

    system_message = personas.get(audience, personas['clinician'])

    user_message = f"""
        DATA TO ANALYZE:
        {json.dumps(context_data, indent=2)}

        INSTRUCTIONS:
        - Reference the specific confidence score and XAI consistency.
        - If consistency > 0.6, state that the AI appears highly reliable.
        - Keep response to 3 short paragraphs.
    """

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=400,
        temperature=0.7,
    )

    return response.choices[0].message.content