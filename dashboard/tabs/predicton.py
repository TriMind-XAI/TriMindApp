import streamlit as st
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success("Image uploaded successfully!")
