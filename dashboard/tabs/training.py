import streamlit as st
import torch
from models.image_models import resnet,simple_cnn
from training.image_train import train_loop
import altair as alt
import pandas as pd
# def plotResults(num_epochs):
#     metrics_df = pd.DataFrame({
#         "Epoch": range(1, num_epochs + 1),
#         "Train Loss": train_losses,
#         "Train Accuracy": train_accuracies,
#         "Test Accuracy": test_accuracies
#     })
#     st.subheader("Training vs Test Accuracy")
#     st.line_chart(
#     metrics_df.set_index("Epoch")[["Train Accuracy", "Test Accuracy"]]
# )

def render_training_page():
    model_option = st.selectbox(
        "Select Model",
        ["SmallCNN","ResNet-8", "Pretrained Medical Model"]
    )
    model=None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_option is "SmallCNN":
        model=simple_cnn.SmallCNN(num_classes=st.session_state["num_classes"],in_channels=st.session_state["in_channels"]).to(device)
    if model_option is "ResNet-8":
        model=resnet.ResNet8(num_classes=st.session_state["num_classes"],in_channels=st.session_state["in_channels"]).to(device)
    
    if st.button("Train Model"):
        with st.spinner("Training model... Please wait ⏳"):
            train_accuracies, test_accuracies = train_loop(
                model,
                st.session_state["train_dataset"],
                st.session_state["test_dataset"],
                15
            )
        metrics_df = pd.DataFrame({
        "Epoch": range(1, len(train_accuracies) + 1),
        "Train Accuracy": train_accuracies,
            "Test Accuracy": test_accuracies
        })

        st.session_state["metrics_df"] = metrics_df

    if "metrics_df" in st.session_state:
        
        df = st.session_state["metrics_df"]

        accuracy_chart = alt.Chart(df).transform_fold(
            ["Train Accuracy", "Test Accuracy"],
            as_=["Metric", "Accuracy"]
        ).mark_line().encode(
            x=alt.X("Epoch:Q", title="Epoch"),
            y=alt.Y(
                "Accuracy:Q",
                scale=alt.Scale(domain=[df[["Train Accuracy", "Test Accuracy"]].min().min() - 2,
                                        df[["Train Accuracy", "Test Accuracy"]].max().max() + 2]),
                title="Accuracy (%)"
            ),
            color="Metric:N"
        ).properties(
            title="Training vs Test Accuracy"
        )

        st.altair_chart(accuracy_chart, use_container_width=True)