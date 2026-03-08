import streamlit as st

from utils import apply_theme
from tabs.data import render_data_page
from tabs.training import render_training_page
from components.sidebar import render_sidebar

language_mode,dark_mode = render_sidebar()
# apply_theme(dark_mode)
st.set_page_config(
    page_title="TriMind",
    layout="wide"
)
st.title("TriMind Dashboard")

st.write("Welcome to our Explainable AI App")

data_tab, training_tab, prediciton_tab = st.tabs([
    "Data",
    "Training",
    "Prediction"
])
with data_tab:
    render_data_page()

with training_tab:
    render_training_page()

with prediciton_tab:
    st.write("Not Implemented")