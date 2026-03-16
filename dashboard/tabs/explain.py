import streamlit as st

from XAI.hugging_face import generate_narrative_hf, setup_huggingface
@st.cache_resource
def get_llm_client():
    return setup_huggingface(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

def render_explanation_page():
    client = get_llm_client()
    if st.session_state['exp_results'] != None:
        for target in ['clinician', 'patient', 'researcher']:
            st.subheader(f"{target} Explanation")
            narrative = generate_narrative_hf(client, results=st.session_state["exp_results"], consistency_metrics=st.session_state["consistency"], audience=target)
            st.write(narrative)