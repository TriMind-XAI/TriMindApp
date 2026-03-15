import streamlit as st
import pandas as pd
import altair as alt
from training.tabular_train import train_and_evaluate
if "accuracy" not in st.session_state:
    st.session_state["accuracy"] = None
    
def render_training_page_tabular():
    model_option = st.selectbox(
        "Select Model",
        ["RandomForest","SVM","XGBoost","DecisionTree", "MLP","Pretrained Model"]
    )
    st.session_state["model_name"] = model_option
    st.session_state["accuracy"] = None
    st.session_state["report"] = None
    st.session_state["metrics_df"] = None
    if st.button("Train Model"):
        with st.spinner("Training model... Please wait ⏳"):
            accuracy,report= train_and_evaluate(model_option)
            st.session_state["accuracy"] = accuracy
            st.session_state["report"] = report
        if model_option =="MLP":
            metrics_df = pd.DataFrame({
            "Epoch": range(1, len(report["accuracy"]) + 1),
            "Train Loss": report["train_loss"],
            "Accuracy": report["accuracy"],
            # "Validation Loss": report["val_loss"],
            # "AUC": report["auc"]
            })
            st.session_state["metrics_df"] = metrics_df
    if st.session_state["metrics_df"] is not None and model_option =="MLP":

        df = st.session_state["metrics_df"]

        accuracy_chart = alt.Chart(df).transform_fold(
            ["Accuracy", "Train Loss"],
            as_=["Metric", "Accuracy"]
        ).mark_line().encode(
            x=alt.X("Epoch:Q", title="Epoch"),
            y=alt.Y(
                "Accuracy:Q",
                scale=alt.Scale(domain=[df[["Train Loss", "Accuracy"]].min().min() ,
                                        df[["Train Loss", "Accuracy"]].max().max() ]),
                title="Accuracy (%)"
            ),
            color="Metric:N"
        ).properties(
            title="Training vs Test Accuracy"
        )
        st.altair_chart(accuracy_chart,width="stretch")

        col1, col2 = st.columns(2)

        col1.markdown("### Accuracy")
        col1.markdown(
            f"<h2 style='margin-top:-10px;'> {st.session_state['accuracy']*100:.2f}%</h2>",
            unsafe_allow_html=True
        )

    if st.session_state["accuracy"] is not None and model_option != "MLP" and "report" in st.session_state:

        col1, col2 = st.columns(2)

        col1.markdown("### Accuracy")
        col1.markdown(
            f"<h2 style='margin-top:-10px;'> {st.session_state['accuracy']*100:.2f}%</h2>",
            unsafe_allow_html=True
        )
        st.subheader("Classification Report")
        st.text(st.session_state["report"])



