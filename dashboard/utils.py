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