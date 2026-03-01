import streamlit as st

def render_sidebar():
    st.sidebar.title("TriMind")
    st.sidebar.markdown("---")

    dark_mode = st.sidebar.toggle("Dark Mode", value=False)

    st.session_state["dark_mode"] = dark_mode
    language_mode = st.sidebar.selectbox(
        "Select language",
        ["English"]
    )


    return language_mode, dark_mode