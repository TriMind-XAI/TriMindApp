from XAI.tabular_xai import explain_with_shap, extract_cf_changes, generate_counterfactuals
import streamlit as st
import torch
import random
import numpy as np
import pandas as pd

from utils import get_tabular_class_name

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
        # pred = model.predict(sample)[0]
        probs = model.predict_proba(sample)
        pred = np.argmax(probs)
        prob = probs[0][pred]
    st.session_state["tabular_results"] = {
        "prediction": pred,
        "confidence": prob
    }

    label = get_tabular_class_name(pred)

    if pred == 1:
        st.success(f"Prediction: {label}")
    else:
        st.error(f"Prediction: {label}")

    if prob is not None:
        st.subheader(f"Confidence: {abs(prob*100)}%")

def render_prediction_page_tabular():
    if st.button("Pick Random Sample",type="primary"):
        X_test = st.session_state["test_x"]
        y_test = st.session_state["test_y"]

        idx = np.random.randint(0, len(X_test))
        sample = X_test[idx].reshape(1, -1)
        st.session_state["tabular_sample"] = sample
        st.session_state["true_label"] = y_test.iloc[idx]

    if st.session_state["tabular_sample"] is not None:

        sample_df = pd.DataFrame(
            st.session_state["tabular_sample"],
            columns=st.session_state["feature_names"].columns.tolist()
        )

        st.write("Sample Features")

        edited_sample = st.data_editor(
            sample_df,
            num_rows="fixed",
            width="content"
        )

        st.session_state["tabular_sample"] = edited_sample.iloc[0].values.reshape(1,-1)

    if st.session_state["true_label"] is not None:

        label = st.session_state["true_label"]

        if label == 1:
            st.info("This sample is: Benign")
        else:
            st.info("This sample is: Malignant")
    if st.button("Predict"):
        if st.session_state["tabular_sample"] is not None :
            sample = st.session_state["tabular_sample"]
            run_tabular_prediction(sample)
            with st.spinner("🔍 Generating SHAP explanations..."):
                explain_with_shap(sample)
                cf_df = generate_counterfactuals(st.session_state["model"], st.session_state["train_x"], st.session_state["train_y"], sample_df, num_cfs=1)
                cf_changes = extract_cf_changes(sample_df, cf_df)
                print("||||cf_changes||||", cf_changes)
                st.session_state["cf_changes"]= cf_changes
                st.toast("Prediciton Complete", icon="😍")
            