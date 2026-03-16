import streamlit as st
def init_session_state():

    defaults = {
        "accuracy": None,
        "report": None,
        "metrics_df": None,
        "history": None,
        "train_dataset": None,
        "test_dataset": None,
        "train_x": None,
        "train_y": None,
        "test_x": None,
        "test_y": None,
        "num_classes": None,
        "class_names":None,
        "img_tensor":None,
        "model":None,
        "model_name":None,
        "img_display":None,
        "true_label":None,
        "tabular_sample":None,
        "consistency":None,
        "exp_results":None,
        "data_type": "Image data"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
def reset_data_states():
    keys = [
        "train_dataset",
        "test_dataset",
        "train_x",
        "train_y",
        "test_x",
        "test_y",
        "num_classes",
        "tabular_sample"
    ]

    for key in keys:
        st.session_state[key] = None
        
def reset_training_states():
    keys = [
        "model",
        "accuracy",
        "report",
        "metrics_df",
        "history"
    ]

    for key in keys:
        st.session_state[key] = None
def reset_prediction_states():
    keys = [
        "img_tensor",
        "img_display",
        "true_label",
        "consistency",
        "exp_results"
    ]

    for key in keys:
        st.session_state[key] = None