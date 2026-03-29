import streamlit as st

def render_sidebar():
    st.sidebar.title("TriMind Logo")
    st.sidebar.markdown("---")
    st.sidebar.header("Data Step:")
    # if st.session_state["current_page"]=="data":
    st.sidebar.write("1 - you can select the data type from the dropdown")
    st.sidebar.write("2 - you can select the dataset you want to train on from the dropdown")
    st.sidebar.write("3 - click train and wait")
