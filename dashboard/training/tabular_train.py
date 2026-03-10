import torch.optim as optim
import torch
import streamlit as st
import time
import torch.nn as nn
import numpy as np
from models.tabular_models.mlp import MLP
from models.tabular_models.ml_models import get_model

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    classification_report
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mlp(X_train, y_train, X_val, y_val, patience=10, epochs=100):
    model = MLP(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5
    )

    best_loss = np.inf
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "auc": []
    }

    for epoch in range(epochs):

        # Train
        model.train()
        optimizer.zero_grad()

        outputs = model(torch.FloatTensor(X_train)).squeeze()
        loss = criterion(outputs, torch.FloatTensor(y_train.values))

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val)).squeeze()
            val_loss = criterion(
                val_outputs,
                torch.FloatTensor(y_val.values)
            )

            probs = torch.sigmoid(val_outputs).numpy()
            preds = (probs > 0.5).astype(int)

            acc = accuracy_score(y_val, preds)
            auc = roc_auc_score(y_val, probs)

        scheduler.step(val_loss)

        # metrics
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["accuracy"].append(acc)
        history["auc"].append(auc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {loss.item():.4f}")
        print(f"Validation Loss: {val_loss.item():.4f}")
        print(f"Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("--------------------")

        # Early Stopping
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            break

    return model, history

def train_and_evaluate(model_name):
    print(f"\nTraining {model_name}...\n")

    if model_name == "MLP":
        print("MLP")
        model, history = train_mlp(
            st.session_state["train_x"],
            st.session_state["train_y"],
            st.session_state["test_x"],
            st.session_state["test_y"]
        )

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
        model.fit(st.session_state["train_x"], st.session_state["train_y"])
        preds = model.predict(st.session_state["test_x"])
        accuracy = accuracy_score(st.session_state["test_y"], preds)
        class_report=classification_report(st.session_state["test_y"], preds)
        return accuracy,class_report
