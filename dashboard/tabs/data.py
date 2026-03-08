import streamlit as st
import medmnist
import os
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

@st.cache_resource
def get_available_medmnist():
    return [
        key for key, value in INFO.items()
        if value['task'] in ['binary-class', 'multi-class']
    ]
@st.cache_resource
def load_medmnist(dataset_name, size=28, transform=None,batch=64):
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


def show_samples_streamlit(dataloader, title, num_samples=6):
    images, labels = next(iter(dataloader))
    class_names = dataloader.dataset.info["label"]

    num_samples = min(num_samples, len(images))

    st.subheader(title)

    cols = st.columns(6)  

    for i in range(num_samples):
        col = cols[i]

        img = images[i].numpy()

        # remove normalization 
        img = (img - img.min()) / (img.max() - img.min())

        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        else:
            img = img.squeeze()

        label_idx = labels[i].item()
        label_name = class_names[str(label_idx)]
        col.image(img, use_container_width=True)
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
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        st.session_state["train_dataset"]=train_loader
        st.session_state["test_dataset"]=test_loader
        st.session_state["num_classes"] = len(train_loader.dataset.info["label"])
        st.session_state["in_channels"] = train_dataset[0][0].shape[0]
        st.write("Train size: ", len(train_dataset))
        st.write("Test size: ", len(test_dataset))
        st.write("classes: ", st.session_state["num_classes"])
        show_samples_streamlit(st.session_state["train_dataset"], f"{selected_dataset} Samples")

        return selected_dataset
    return None