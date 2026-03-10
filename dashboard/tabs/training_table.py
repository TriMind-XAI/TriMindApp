import streamlit as st
import pandas as pd
import altair as alt
from training.tabular_train import train_and_evaluate
def render_training_page_tabular():
    model_option = st.selectbox(
        "Select Model",
        ["RandomForest","SVM","XGBoost","DecisionTree", "MLP","Pretrained Model"]
    )
    if st.button("Train Model"):
        with st.spinner("Training model... Please wait ⏳"):
            accuracy,report= train_and_evaluate(model_option)
            st.session_state["accuracy"] = accuracy
            st.session_state["report"] = report
    if "accuracy" in st.session_state:
        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", f"{st.session_state["accuracy"]:.3f}")

        st.subheader("Classification Report")
        st.text(st.session_state["report"])



