import streamlit as st
import extra_streamlit_components as stx
from tabs.explain import render_explanation_page_image,render_explanation_page_tabular
from training.app_states import init_session_state
# from utils import apply_theme
from tabs.data import render_data_page
from tabs.training_image import render_training_page_image
from tabs.training_table import render_training_page_tabular
from tabs.predicton_im import render_prediction_page_image
from tabs.prediction_tab import render_prediction_page_tabular
from tabs.compare import render_compare_page
from components.sidebar import render_sidebar
st.markdown("""
    <style>
        iframe {
            height: 82px !important;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 25px;
        }
        .st-emotion-cache-10p9htt {
            height: 1rem !important;
            margin-bottom: 0rem !important;
        }
        hr {
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
        }
        button[kind="primary"] {
            background-color: #262930;
        }
    </style>
""", unsafe_allow_html=True)
init_session_state()
if "step" not in st.session_state:
    st.session_state.step=0
render_sidebar()
st.set_page_config(
    page_title="TriMind",
    layout="wide"
)
# st.title("TriMind Dashboard")
# st.write("Welcome to our Explainable AI App")
steps_names=["Data", "Train", "Predict", "Explain","Compare"]
stx.stepper_bar(
    steps=steps_names,
    default=st.session_state.step,
    lock_sequence=True
)
# Buttons
col1, colMiddle, col2 = st.columns([1,6,1])

with col1:
    if st.button("← Back", disabled=st.session_state.step == 0):
        if st.session_state.step >0:
            st.session_state.step -= 1
            st.rerun()
# with colMiddle:
#     st.header(steps_names[st.session_state.step])
with col2:
    if st.button("Next →", disabled=st.session_state.step == 4):
        if st.session_state.step <4:
            st.session_state.step += 1
            st.rerun()

if st.session_state.step==0:
    print("rendering data")
    render_data_page()

elif st.session_state.step==1:
    print("rendering training")
    if st.session_state["data_type"] == "Image data":
        render_training_page_image()
    else:
        render_training_page_tabular()

elif st.session_state.step==2:
    print("rendering prediciton")
    if st.session_state["data_type"] == "Image data":
        render_prediction_page_image()
    else:
        render_prediction_page_tabular()
elif st.session_state.step==3:
    print("rendering explanation")
    if st.session_state["data_type"] == "Image data":
        render_explanation_page_image()
    else:
        render_explanation_page_tabular()
elif st.session_state.step==4:
    print("rendering compare")
    render_compare_page()

