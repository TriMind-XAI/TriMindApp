import streamlit as st
import torch
import numpy as np 

def apply_theme(dark_mode):
    if dark_mode:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #0E1117;
                    color: white;
                }
                section[data-testid="stSidebar"] {
                    background-color: #111827;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .stApp {
                    background-color: white;
                    color: black;
                }
                section[data-testid="stSidebar"] {
                    background-color: #F0F2F6;
                }
            </style>
        """, unsafe_allow_html=True)
class TorchModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            output = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(output).numpy()

        # convert to sklearn format [prob_0, prob_1]
        probs = np.vstack([1 - probs, probs]).T
        return probs

def get_class_name(class_idx):

    labels = st.session_state.get("class_names")

    if labels is None:
        return str(class_idx)

    try:
        return labels[str(class_idx)]
    except:
        return labels[class_idx]
def get_tabular_class_name(class_idx):

    labels = st.session_state.get("tabular_class_names")

    if labels is None:
        return str(class_idx)

    return labels.get(class_idx, str(class_idx))
def get_icon_title(title,icon):
    html=f"""
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="display:inline-block;">
            <img src="data:image/png;base64,{icon}" 
                width="24" 
                style="filter: brightness(0) saturate(100%) invert(72%) sepia(36%) saturate(746%) hue-rotate(155deg) brightness(101%) contrast(101%);">
        </div>
        <span style="font-size:22px; font-weight:600;">
            {title}
        </span>
    </div>
    """
    return html
