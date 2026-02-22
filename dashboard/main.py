import streamlit as st

st.title("TriMind Dashboard")
st.write("Welcome to our Explainable AI App")

dateType_option = st.selectbox(
    "Select data type",
    ["Image data","tabular data"]
)

data_option = st.selectbox(
    "Select dataset",
    ["Pneumonia (MNIST)","Breast (MNIST)", "Chest (MNIST)"]
)

model_option = st.selectbox(
    "Select Model",
    ["SmallCNN","ResNet-8", "Pretrained Medical Model"]
)

st.write("You selected:", model_option)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success("Image uploaded successfully!")
