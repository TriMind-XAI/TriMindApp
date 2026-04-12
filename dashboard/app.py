
import time
start = time.time()
# your imports...
import streamlit as st
import extra_streamlit_components as stx
from training.app_states import init_session_state
# from utils import apply_theme
from tabs.compare import render_compare_page
from components.sidebar import render_sidebar

# print("After imports:", time.time() - start)

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
            st.session_state.loading = True
            st.rerun()
with col2:
    if st.button("Next →", disabled=st.session_state.step == 4):
        if st.session_state.step <4:
            st.session_state.step += 1
            st.session_state.loading = True
            st.rerun()

if st.session_state.step==0:
    print("rendering data")
    from tabs.data import render_data_page
    render_data_page()

elif st.session_state.step==1:
    print("rendering training")
    if st.session_state["data_type"] == "Image data":
        from tabs.training_image import render_training_page_image
        render_training_page_image()
    else:
        from tabs.training_table import render_training_page_tabular
        render_training_page_tabular()
elif st.session_state.step==2:
    print("rendering prediciton")
    if st.session_state["data_type"] == "Image data":
        from tabs.predicton_im import render_prediction_page_image
        render_prediction_page_image()
    else:
        from tabs.prediction_tab import render_prediction_page_tabular
        render_prediction_page_tabular()
elif st.session_state.step==3:
    print("rendering explanation")
    if st.session_state["data_type"] == "Image data":
        from tabs.explain import render_explanation_page_image
        render_explanation_page_image()
    else:
        from tabs.explain import render_explanation_page_tabular
        render_explanation_page_tabular()
elif st.session_state.step==4:
    print("rendering compare")
    render_compare_page()

# print("After app.py:", time.time() - start)