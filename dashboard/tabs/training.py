import streamlit as st
import torch
from models.image_models import resnet,simple_cnn
from training.image_train import train_loop
def render_training_page():
    model_option = st.selectbox(
        "Select Model",
        ["SmallCNN","ResNet-8", "Pretrained Medical Model"]
    )
    model=None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_option is "SmallCNN":
        model=simple_cnn.SmallCNN().to(device)
    if model_option is "ResNet-8":
        model=resnet.ResNet8().to(device)
    st.button("train",on_click=train_loop(model,10))


