import streamlit as st

from tabs.data import render_data_page
# from tabs.training import render_training_page
from components.sidebar import render_sidebar

language_mode,dark_mode = render_sidebar()
st.title("TriMind Dashboard")

st.write("Welcome to our Explainable AI App")

data_tab, training_tab, prediciton_tab = st.tabs([
    "data",
    "Training",
    "prediction"
])
with data_tab:
    render_data_page()

with training_tab:
    # render_training_page()
    st.write("Not Implemented")

with prediciton_tab:
    st.write("Not Implemented")
