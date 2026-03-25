import streamlit as st

from XAI.hugging_face import setup_huggingface
from XAI.image_xai import generate_narrative_hf
from XAI.tabular_xai import convert_to_llm
@st.cache_resource
def get_llm_client():
    return setup_huggingface(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

def render_explanation_page_image():
    client = get_llm_client()
    if st.session_state['image_exp_results'] != None:
        for target in ['Clinician','Researcher', 'Patient']:
            st.subheader(f"{target} Explanation")
            narrative = generate_narrative_hf(client, results=st.session_state["image_exp_results"], consistency_metrics=st.session_state["consistency"], audience=target)
            st.write(narrative)
            
def render_explanation_page_tabular():
    client = get_llm_client()
    if (
        st.session_state.get("tabular_results") is not None and
        st.session_state.get("tabular_shap_values") is not None
    ):
        for target in ['Clinician','Researcher', 'Patient']:
            st.subheader(f"{target} Explanation")
            narrative = convert_to_llm(
                client,
                shap_values=st.session_state["tabular_shap_values"],
                sample=st.session_state["tabular_sample"],
                results=st.session_state["tabular_results"],
                audience=target
            )
            st.write(narrative)
    else:
        print("state error")
