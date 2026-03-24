from huggingface_hub import InferenceClient
import streamlit as st

def setup_huggingface(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    # print("🚀 Connecting to Hugging Face Inference API...")

    client = InferenceClient(
        model=model_name,
        token=st.secrets["HF_TOKEN"]
    )

    print(f"✅ Connected to model: {model_name}")
    return client

