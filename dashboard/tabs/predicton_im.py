import streamlit as st
import random
import torch
from PIL import Image
from torchvision import transforms
if "img_tensor" in st.session_state:
    st.session_state["img_tensor"]=None
def run_prediction(img_tensor):
    st.session_state["model"].eval()
    print("run_prediction")
    with torch.no_grad():
        outputs = st.session_state["model"](img_tensor)
        _, predicted = torch.max(outputs,1)

    class_names = st.session_state["train_dataset"].dataset.info["label"]

    pred_class = class_names[str(predicted.item())]

    st.subheader("Prediction Result")
    st.success(pred_class)

def render_prediction_page_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if st.button("Pick Random Sample"):
        test_loader = st.session_state["test_dataset"]

        images, labels = next(iter(test_loader))

        idx = random.randint(0, len(images) - 1)

        img = images[idx]
        label = labels[idx].item()

        img_display = img.numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        if img_display.shape[0] == 3:
            img_display = img_display.transpose(1,2,0)
        else:
            img_display = img_display.squeeze()
        
        class_names = st.session_state["train_dataset"].dataset.info["label"]
        class_name=class_names[str(label)]
        st.subheader(f"this should be {class_name}")
        st.image(img_display, caption="Random Test Image",width=200)
        st.session_state["img_tensor"] = img.unsqueeze(0).to(device)
    st.subheader("OR")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png","jpg","jpeg"]
    )
    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        st.image(image, caption="Uploaded Image", width=200)

        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        
        st.session_state["img_tensor"] = transform(image).unsqueeze(0).to(device)
    if st.session_state["img_tensor"] is not None:
        if st.button("Predict"):
            print("hi predict")
            run_prediction(st.session_state["img_tensor"])
