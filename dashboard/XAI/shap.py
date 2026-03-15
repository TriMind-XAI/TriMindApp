import shap
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def explain_with_shap(sample):
    
    plt.style.use("dark_background")
    model=st.session_state["model"]
    
    print("\nGenerating SHAP explanations...\n")

    # Use scaled data consistently
    X_background = st.session_state["train_x"][:100]
    X_sample = sample

    # Convert to DataFrame with feature names
    X_sample_df = pd.DataFrame(
        X_sample,
        columns=st.session_state["feature_names"].columns
    )

    if st.session_state["model_name"] == "MLP":
        
        model.eval()
        explainer = shap.DeepExplainer(
            model,
            torch.FloatTensor(X_background)
        )

        shap_values = explainer.shap_values(
            torch.FloatTensor(X_sample)
        )

        # DeepExplainer returns list for binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    elif st.session_state["model_name"] =="SVM":  # SVM
        explainer = shap.KernelExplainer(
            model.predict_proba,
            X_background
        )
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    if isinstance(model, torch.nn.Module):
        model.eval()
        sample_tensor = torch.FloatTensor(sample)

        with torch.no_grad():
            output = model(sample_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
    else:
        pred = model.predict(sample)[0]

    shap_values = np.array(shap_values)

    # extract single sample
    if shap_values.ndim == 3:
        shap_values = shap_values[0,:,0]   # only class available   
    else:
        shap_values = shap_values[0]

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        if len(base_value) > 1:
            base_value = base_value[pred]
        else:
            base_value = base_value[0]

    base_value = float(base_value)
    exp = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X_sample_df.iloc[0],
        feature_names=list(st.session_state["feature_names"])
    )
    shap.plots.bar(exp, show=False)

    fig = plt.gcf()
    st.pyplot(fig)