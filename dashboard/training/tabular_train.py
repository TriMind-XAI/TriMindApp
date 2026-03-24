import torch
import streamlit as st

from training.training_mlp import train_mlp
from models.tabular_models.ml_models import get_model

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

device = st.session_state["device"]

def train_and_evaluate(model_name):
    print(f"\nTraining {model_name}...\n")

    if model_name == "MLP":
        print("MLP")
        model, history = train_mlp(
            st.session_state["train_x"],
            st.session_state["train_y"],
            st.session_state["test_x"],
            st.session_state["test_y"],
            epochs=70
        )
        st.session_state["model"]=model
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.FloatTensor(st.session_state["test_x"]))
            ).numpy().flatten()
        preds = (probs > 0.5).astype(int)
        accuracy = accuracy_score(st.session_state["test_y"], preds)
        return accuracy,history

    else:
        model = get_model(model_name)
        st.session_state["model"]=model
        model.fit(st.session_state["train_x"], st.session_state["train_y"])
        preds = model.predict(st.session_state["test_x"])
        accuracy = accuracy_score(st.session_state["test_y"], preds)
        class_report=classification_report(st.session_state["test_y"], preds)
        return accuracy,class_report
