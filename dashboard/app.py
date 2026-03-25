import streamlit as st

from tabs.explain import render_explanation_page_image,render_explanation_page_tabular
from training.app_states import init_session_state
from utils import apply_theme
from tabs.data import render_data_page
from tabs.training_image import render_training_page_image
from tabs.training_table import render_training_page_tabular
from tabs.predicton_im import render_prediction_page_image
from tabs.prediction_tab import render_prediction_page_tabular
from components.sidebar import render_sidebar

import torch

language_mode,dark_mode = render_sidebar()
# apply_theme(dark_mode)
st.set_page_config(
    page_title="TriMind",
    layout="wide"
)
init_session_state()
if "device" not in st.session_state:
    st.session_state["device"] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
st.title("TriMind Dashboard")

st.write("Welcome to our Explainable AI App")

if "data_type" not in st.session_state:
    st.session_state["data_type"] = "Image data"
    
data_tab, training_tab, prediciton_tab,Explain_tab = st.tabs([
    "Data",
    "Training",
    "Prediction",
    "Explain"
])
with data_tab:
    render_data_page()

with training_tab:
    if st.session_state["data_type"] == "Image data":
        render_training_page_image()
    else:
        render_training_page_tabular()

with prediciton_tab:
    if st.session_state["data_type"] == "Image data":
        render_prediction_page_image()
    else:
        render_prediction_page_tabular()
with Explain_tab:
    if st.session_state["data_type"] == "Image data":
        render_explanation_page_image()
    else:
        render_explanation_page_tabular()
