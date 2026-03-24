import streamlit as st
def apply_theme(dark_mode):
    if dark_mode:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #0E1117;
                    color: white;
                }
                section[data-testid="stSidebar"] {
                    background-color: #111827;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .stApp {
                    background-color: white;
                    color: black;
                }
                section[data-testid="stSidebar"] {
                    background-color: #F0F2F6;
                }
            </style>
        """, unsafe_allow_html=True)
def get_class_name(class_idx):

    labels = st.session_state.get("class_names")

    if labels is None:
        return str(class_idx)

    try:
        return labels[str(class_idx)]
    except:
        return labels[class_idx]
def get_tabular_class_name(class_idx):

    labels = st.session_state.get("tabular_class_names")

    if labels is None:
        return str(class_idx)

    return labels.get(class_idx, str(class_idx))