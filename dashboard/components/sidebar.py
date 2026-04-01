import streamlit as st
import os
from utils import get_icon_title
def render_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    logo_path = os.path.join(BASE_DIR, "assets/logo.png")
    st.sidebar.image(logo_path,width='content')
    st.sidebar.divider()
    if st.session_state.step==0:
        data_sidebar()
    elif st.session_state.step==1:
        train_sidebar()
    elif st.session_state.step==2:
        predict_sidebar()
    elif st.session_state.step==3:
        explain_sidebar()
import base64

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def data_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_step = get_base64_image(os.path.join(BASE_DIR, "assets", "dataset.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))

    st.sidebar.markdown(get_icon_title("Step 1: Data Setup",img_step), unsafe_allow_html=True)
    st.sidebar.markdown("""  
        In this step, you will choose the type of <span style="color:#22D3EE;font-weight:700">Data</span> and load a dataset 
        to prepare it for training.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Select the <span style="color:#22D3EE;font-weight:700">data type</span> (Image or Tabular)
    - Choose a medical <span style="color:#22D3EE;font-weight:700">dataset</span>
    - Explore the <span style="color:#22D3EE;font-weight:700">data</span> preview and statistics
    """,unsafe_allow_html=True)
    
def train_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_train = get_base64_image(os.path.join(BASE_DIR, "assets", "train.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))
    img_tips = get_base64_image(os.path.join(BASE_DIR, "assets", "tips.png"))
    st.sidebar.markdown(get_icon_title("Step 2: Training",img_train), unsafe_allow_html=True)
    st.sidebar.markdown("""
    <span style="color:#22D3EE;font-weight:700">Train</span> a model using your selected dataset. Configure the model and adjust <span style="color:#22D3EE;font-weight:700">training</span> settings before starting.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Select a <b>model architecture</b></li>
    - Choose number of <b>epochs</b></li>
    - Start <span style="color:#22D3EE;font-weight:700">training</span> and evaluate the accuracy
    """,unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(get_icon_title("Tips",img_tips), unsafe_allow_html=True)
    st.sidebar.markdown("""
                        - Models with a star ⭐ are recommended.
                        - More epochs can improve accuracy but take longer to <span style="color:#22D3EE;font-weight:700">train</span> 
                        """,unsafe_allow_html=True)

def predict_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_predict = get_base64_image(os.path.join(BASE_DIR, "assets", "predict.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))
    img_tips = get_base64_image(os.path.join(BASE_DIR, "assets", "tips.png"))
    st.sidebar.markdown(get_icon_title("Step 3: predict",img_predict), unsafe_allow_html=True)
    st.sidebar.markdown("""
    Select an input sample and run the model to see its <span style="color:#22D3EE;font-weight:700">prediction</span>.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Choose a sample and see if the model <span style="color:#22D3EE;font-weight:700">predicts</span> it right</li>
    - Weird math and jargon, right?
    - Cick next to read <span style="color:#22D3EE;font-weight:700">explanations</span>
    """,unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(get_icon_title("Tips",img_tips), unsafe_allow_html=True)
    if st.session_state["data_type"]== "Image data":
        st.sidebar.markdown("""
                            - Choose a random Image (Don't act smart😈)"
                            """,unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
                            - You can change the values to see if it makes a difference 👀.
                            """,unsafe_allow_html=True)

def explain_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_explain = get_base64_image(os.path.join(BASE_DIR, "assets", "explain.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))
    img_tips = get_base64_image(os.path.join(BASE_DIR, "assets", "tips.png"))
    st.sidebar.markdown(get_icon_title("Step 3: Predict",img_explain), unsafe_allow_html=True)
    st.sidebar.markdown("""
    Select an input sample and run the model to see its <span style="color:#22D3EE;font-weight:700">prediction</span>.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Choose a sample and see if the model <span style="color:#22D3EE;font-weight:700">predicts</span> it right</li>
    - Weird math and jargon, right?
    - Cick next to read <span style="color:#22D3EE;font-weight:700">explanations</span>
    """,unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(get_icon_title("Tips",img_tips), unsafe_allow_html=True)
    if st.session_state["data_type"]== "Image data":
        st.sidebar.markdown("""
                            - Choose a random Image (Don't act smart😈)"
                            """,unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
                            - You can change the values to see if it makes a difference 👀.
                            """,unsafe_allow_html=True)
def explain_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_explain = get_base64_image(os.path.join(BASE_DIR, "assets", "explain.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))
    img_tips = get_base64_image(os.path.join(BASE_DIR, "assets", "tips.png"))
    st.sidebar.markdown(get_icon_title("Step 4: Explain",img_explain), unsafe_allow_html=True)
    st.sidebar.markdown("""
        Understand how the model made its prediction by analyzing important features or regions.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Read the <span style="color:#22D3EE;font-weight:700">explanation</span> based on your <b>specialty</b></li>
    - Assess how <b>confident</b> the model is</li>
    - Decide if it supports your <b>decision</b></li>
    """,unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(get_icon_title("Tips",img_tips), unsafe_allow_html=True)
    st.sidebar.markdown("""
                        - You can still go back and try a different sample
                        - Not confident enough? Click next to get a second opinion
                        """,unsafe_allow_html=True)
def compare_sidebar():
    BASE_DIR = os.path.dirname(__file__)
    img_compare = get_base64_image(os.path.join(BASE_DIR, "assets", "compare.png"))
    img_goal = get_base64_image(os.path.join(BASE_DIR, "assets", "goal.png"))
    img_tips = get_base64_image(os.path.join(BASE_DIR, "assets", "tips.png"))
    st.sidebar.markdown(get_icon_title("Step 5: Compare",img_compare), unsafe_allow_html=True)
    st.sidebar.markdown("""
    Select an input sample and run the model to see its <span style="color:#22D3EE;font-weight:700">prediction</span>.
    """, unsafe_allow_html=True)
    st.sidebar.divider()
    
    st.sidebar.markdown(get_icon_title("What to do",img_goal), unsafe_allow_html=True)
    st.sidebar.markdown("""
    - Choose a sample and see if the model <span style="color:#22D3EE;font-weight:700">predicts</span> it right</li>
    - Weird math and jargon, right?
    - Cick next to read <span style="color:#22D3EE;font-weight:700">explanations</span>
    """,unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(get_icon_title("Tips",img_tips), unsafe_allow_html=True)
    if st.session_state["data_type"]== "Image data":
        st.sidebar.markdown("""
                            - Choose a random Image (Don't act smart😈)"
                            """,unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
                            - You can change the values to see if it makes a difference 👀.
                            """,unsafe_allow_html=True)
