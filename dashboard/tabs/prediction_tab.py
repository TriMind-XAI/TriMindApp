from XAI.shap import explain_with_shap
import streamlit as st
import torch
import random
import numpy as np

def run_tabular_prediction(sample):

    model = st.session_state["model"]
    if isinstance(model, torch.nn.Module):
        model.eval()
        sample_tensor = torch.FloatTensor(sample)

        with torch.no_grad():
            output = model(sample_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

    else:
        pred = model.predict(sample)[0]
        prob = None

    if pred == 1:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")

    if prob is not None:
        st.write(f"Confidence: {prob:.3f}")

def render_prediction_page_tabular():
    if "tabular_sample" not in st.session_state:
        st.session_state["tabular_sample"]=None
    if st.button("Pick Random Sample"):
        X_test = st.session_state["test_x"]
        y_test = st.session_state["test_y"]

        idx = np.random.randint(0, len(X_test))

        sample = X_test[idx].reshape(1, -1)

        st.session_state["tabular_sample"] = sample
        st.session_state["true_label"] = y_test.iloc[idx]

        st.write("Sample Features")
        st.dataframe(sample)
    if st.session_state["true_label"] is not None:

        label = st.session_state["true_label"]

        if label == 1:
            st.info("True Label: Benign")
        else:
            st.info("True Label: Malignant")
    if st.button("Predict"):
        if st.session_state["tabular_sample"] is not None :
            sample = st.session_state["tabular_sample"]
            run_tabular_prediction(sample)
            explain_with_shap(sample)
            