import torch.optim as optim
import time
import torch
import torch.nn as nn
import numpy as np
from models.tabular_models.mlp import MLP
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

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