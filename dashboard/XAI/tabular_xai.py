import shap
import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import dice_ml

from utils import TorchModelWrapper, get_tabular_class_name

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
        
        st.session_state["tabular_shap_values"] = shap_values

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    elif st.session_state["model_name"] =="SVM": 
        explainer = shap.KernelExplainer(
            model.predict_proba,
            X_background
        )
        shap_values = explainer.shap_values(X_sample)
        
        st.session_state["tabular_shap_values"] = shap_values

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        st.session_state["tabular_shap_values"] = shap_values
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

    
def generate_counterfactuals(model, X_train, y_train, sample_df, num_cfs=1):
    
    X_train = pd.DataFrame(X_train, columns=sample_df.columns)
    df = X_train.copy()
    df["target"] =np.array(y_train)

    dice_data = dice_ml.Data(
        dataframe=df,
        continuous_features=list(X_train.columns),
        outcome_name="target"
    )
    if st.session_state["model_name"] == "MLP":
        wrapped_model = TorchModelWrapper(model)
        dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    else:
        dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice = dice_ml.Dice(dice_data, dice_model)

    cf = dice.generate_counterfactuals(
        sample_df,
        total_CFs=num_cfs,
        desired_class="opposite"
    )

    return cf.cf_examples_list[0].final_cfs_df

def extract_cf_changes(sample_df, cf_df):

    original = sample_df.iloc[0]
    changes_all = []

    for _, row in cf_df.iterrows():
        changes = {}

        for col in sample_df.columns:
            orig = float(original[col])
            new = float(row[col])

            if abs(orig - new) > 1e-3:
                changes[col] = {
                    "from": round(orig, 3),
                    "to": round(new, 3),
                    "delta": round(new - orig, 3)
                }

        changes_all.append(changes)

    return changes_all

def summarize_shap_tabular(shap_values, sample, feature_names, top_k=5):

    shap_values = shap_values.flatten()
    sample = sample.flatten()
    # specfic for Breast Cancer dataset
    feature_descriptions = {
        "mean radius": "tumor size",
        "mean texture": "cell texture variation",
        "mean perimeter": "tumor boundary size",
        "mean concavity": "irregularity of tumor shape",
        "mean concave points": "number of inward curves in tumor boundary",
        "smoothness error": "variation in surface smoothness",
        "compactness error": "density and compactness variation",
        "concave points": "sharp edges",
        "symmetry": "cell symmetry",
        "fractal dimension": "boundary complexity"
    }
    features = []

    features_name_list=feature_names.columns.tolist()
    for i in range(len(features_name_list)):
        name = features_name_list[i]
        # translate techinical names
        if name in feature_descriptions:
            name = f"{name} ({feature_descriptions[name]})"
        # importance
        total = np.sum(np.abs(shap_values))
        importance = abs(shap_values[i]) / total
        # explain effect if its positive or negative  
        effect=""
        if shap_values[i] > 0:
            effect = "increases"
        elif shap_values[i] < 0:
            effect = "decreases"
        else:
            effect = "no effect"
        features.append({
            "feature": name,
            "value": round(float(sample[i]),2),
            "impact": round(float(shap_values[i]),2),
            "importance": round(importance*100, 1),
            "effect":effect
        })
    # Sort by importance
    features = sorted(features, key=lambda x: abs(x["impact"]), reverse=True)

    top_features = features[:top_k]

    # Split positive vs negative
    positive = [f for f in top_features if f["impact"] > 0]
    negative = [f for f in top_features if f["impact"] < 0]

    return positive, negative
def convert_to_llm(client,shap_values,sample,results,audience="clinician"):
    pred_class = get_tabular_class_name(results["prediction"])
    positive, negative = summarize_shap_tabular(
    shap_values,
    sample,
    st.session_state["feature_names"]
    )
    context_data = {
        "diagnosis": pred_class,
        "confidence": f"{results['confidence']:.2%}",
        "supporting_features": positive,
        "contradicting_features": negative,
        "counterfactuals":st.session_state["cf_changes"]
    }
    # this is the bug , I didnt give context to positive and negative just values
    user_message = f"""
    DATA TO ANALYZE:
    {json.dumps(context_data, indent=2)}

    INSTRUCTIONS:
    - Explain which features MOST influenced the prediction.
    - Highlight features that increased the likelihood of the diagnosis.
    - Highlight features that decreased it.
    - Use feature names and their values in your explanation.
    - keep numbers short and readable 
    - Avoid repeating feature names unnecessarily
    - Keep response to 2 short paragraphs.
    - Do NOT infer medical meaning beyond given data
    - Only describe statistical influence of features
    - The "value" is the feature value, NOT the effect direction.
    - The "effect" determines whether it increases of positive value or decreases if it's negative prediction.
    - Do not infer direction from the value.
    - from counterfactuals, explain what minimal changes would alter the prediction and Highlight the most sensitive features
    - Use the counterfactuals to describe how stable or fragile the prediction is
    """
    personas = {
        'Clinician': (
            "You are a clinical expert analyzing breast cancer prediction results "
            "based on extracted tumor features (e.g., size, shape, texture). "
            "Explain how these features influence the prediction without making unsupported medical conclusions."
        ),
        'Researcher':(
            "You are an AI auditor evaluating a breast cancer prediction model. "
            "Analyze how feature importance affects the prediction, discuss whether the decision is dominated "
            "potential bias based only on the provided feature contributions,"
            "comment on reliability based only on the provided data."
        ),
        'Patient': (
            "You are a doctor explaining breast cancer prediction results to a patient. "
            "Translate technical features into simple ideas like size, shape, or texture, "
            "and focus only on the most important factors in a reassuring way."
        ),
    }
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": personas[audience]},
            {"role": "user", "content": user_message}
        ],
        max_tokens=400,
        temperature=0.5,
    )
    print("message content:",response.choices[0].message.content)

    return response.choices[0].message.content