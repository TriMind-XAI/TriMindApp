import streamlit as st
import medmnist
import os
from medmnist import INFO
from torchvision import transforms

def get_available_medmnist():
    return [
        key for key, value in INFO.items()
        if value['task'] in ['binary-class', 'multi-class']
    ]
@st.cache_resource
def load_medmnist(dataset_name, size=28, transform=None):
    root = os.path.expanduser("~/.medmnist")
    download_flag = not os.path.exists(root)
    download_flag = True
    print("download_flag",download_flag)

    DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train = DataClass(
        split='train',
        download=download_flag,
        size=size,
        transform=transform
    )

    test = DataClass(
        split='test',
        download=download_flag,
        size=size,
        transform=transform
    )
    return train,test
    

def render_data_page():

    st.header("Dataset Selection")

    data_type = st.selectbox(
        "Select data type",
        ["Image data", "Tabular data"]
    )

    if data_type == "Image data":
        image_datasets = get_available_medmnist()
        selected_dataset = st.selectbox(
            "Select dataset",
            image_datasets
        )
        st.session_state["selected_dataset"] = selected_dataset
        train_dataset,test_dataset = load_medmnist(selected_dataset)
        st.write("Train size:", len(train_dataset))
        st.write("Test size:", len(test_dataset))

        return selected_dataset
    return None