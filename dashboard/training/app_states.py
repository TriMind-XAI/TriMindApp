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
        "img_tensor":None,
        "model":None,
        "img_display":None,
        "true_label":None,
        "tabular_sample":None,
        "data_type": "Image data"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value