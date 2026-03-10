import streamlit as st

from utils import apply_theme
from tabs.data import render_data_page
from tabs.training_image import render_training_page_image
from tabs.training_table import render_training_page_tabular
from components.sidebar import render_sidebar

language_mode,dark_mode = render_sidebar()
# apply_theme(dark_mode)
st.set_page_config(
    page_title="TriMind",
    layout="wide"
)
st.title("TriMind Dashboard")

st.write("Welcome to our Explainable AI App")

if "data_type" not in st.session_state:
    print("wtf?")
    st.session_state["data_type"] = "Image data"
else: 
    print("wtf double?")
data_tab, training_tab, prediciton_tab = st.tabs([
    "Data",
    "Training",
    "Prediction"
])
with data_tab:
    render_data_page()

with training_tab:
    if st.session_state["data_type"] == "Image data":
        render_training_page_image()
    else:
        render_training_page_tabular()

with prediciton_tab:
    st.write("Not Implemented")
