import streamlit as st
import medmnist
import os
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_available_medmnist():
    return [
        key for key, value in INFO.items()
        if value['task'] in ['binary-class', 'multi-class']
    ]
@st.cache_resource
def load_medmnist(dataset_name, size=28, transform=None,batch=64):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    root = os.path.join(BASE_DIR, "data", "medmnist")
    os.makedirs(root, exist_ok=True)

    dataset_file = os.path.join(root, f"{dataset_name}.npz")
    download_flag = not os.path.exists(dataset_file)

    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train = DataClass(
        split='train',
        download=download_flag,
        size=size,
        transform=transform,
        root=root
    )

    test = DataClass(
        split='test',
        download=download_flag,
        size=size,
        root=root,
        transform=transform
    )

    return train,test
@st.cache_data
def load_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled,y_train, y_test,data

def show_samples_streamlit(dataloader, num_samples=6):
    images, labels = next(iter(dataloader))
    class_names = dataloader.dataset.info["label"]

    cols = st.columns(6)  

    for i in range(num_samples):
        col = cols[i]

        img = images[i].numpy()
        img = img.transpose(1, 2, 0)

        # remove normalization 
        img = (img - img.min()) / (img.max() - img.min())
        label_idx = labels[i].item()
        label_name = class_names[str(label_idx)]
        col.image(img, width="stretch")
        col.markdown(
            f"""
            <div style='text-align:center; font-size:14px; 
                        word-wrap:break-word; 
                        font-weight:600;
                        margin-top:5px;'>
                {label_name}
            </div>
            """,
            unsafe_allow_html=True
        )
def metric_box(title, value):
    st.markdown(f"""
        <div style="
            background-color:#262930;
            padding:10px;
            margin:10px 0px 10px 0;
            border-radius:12px;
            text-align:center;
            box-shadow:0 0 10px rgba(0,0,0,0.3);
        ">
            <div style="font-size:20px;color:white">{title}</div>
            <div style="font-size:36px;font-weight:bold;color:#22D3EE">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def render_image_data():

    image_datasets = get_available_medmnist()
    selected_dataset = st.selectbox(
        "Select dataset",
        image_datasets
    )

    st.session_state["selected_dataset"] = selected_dataset

    train_dataset, test_dataset = load_medmnist(selected_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    st.session_state["train_dataset"] = train_loader
    st.session_state["test_dataset"] = test_loader
    st.session_state["num_classes"] = len(train_loader.dataset.info["label"])
    st.session_state["class_names"] = st.session_state["train_dataset"].dataset.info["label"]
    st.session_state["in_channels"] = train_dataset[0][0].shape[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_box("Train Size", len(train_dataset))

    with col2:
        metric_box("Test Size", len(test_dataset))

    with col3:
        metric_box("Classes", st.session_state["num_classes"])
    st.subheader("Samples")

    show_samples_streamlit(
        st.session_state["train_dataset"],
    )


def render_tabular_data():

    selected_dataset = st.selectbox(
        "Select dataset",
        ["Breast Cancer Wisconsine"]
    )
    st.session_state["selected_dataset"] = selected_dataset
    x_train, x_test, y_train, y_test, full_data = load_breast_cancer_data()
    st.session_state["tabular_class_names"] = {
        0: "Malignant",
        1: "Benign"
    }
    st.session_state["feature_names"] = pd.DataFrame(full_data.data, columns=full_data.feature_names)

    st.session_state["train_x"] = x_train
    st.session_state["test_x"] = x_test
    st.session_state["train_y"] = y_train
    st.session_state["test_y"] = y_test
    st.session_state["num_classes"] = 2
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_box("Train Size", len(x_train))

    with col2:
        metric_box("Test Size", len(x_test))

    with col3:
        metric_box("Classes", st.session_state["num_classes"])
    st.subheader("Samples")
    df = pd.DataFrame(full_data.data, columns=full_data.feature_names)
    df['target'] = full_data.target

    st.dataframe(df.head(5))


def render_data_page():

    # st.header("Dataset Selection")
    data_type = st.selectbox(
        "Select data type",
        ["Image data", "Tabular data"]
    )
    st.session_state["data_type"] = data_type

    if st.session_state["data_type"] == "Image data":
        render_image_data()

    if st.session_state["data_type"] == "Tabular data":
        render_tabular_data()