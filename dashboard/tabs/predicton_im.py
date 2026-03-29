import streamlit as st
import random
import torch
from PIL import Image
from torchvision import transforms

from XAI.image_xai import MultiviewExplainer, analyze_consistency
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

def run_explanation(device,sample_image):
    # Create explainer
    print("\nCreating/getting explainer...")
    explainer = get_explainer()
    # Generate explanations
    results = explainer.explain(sample_image)
    st.session_state["image_exp_results"]=results
    # Visualize
    print("\n📊 Visualizing explanations...")
    explainer.visualize_explanations(sample_image, results)
    # Analyze consistency
    st.session_state["consistency"] = analyze_consistency(results)

def get_explainer():
    if "explainer" not in st.session_state:
        st.session_state["explainer"] = MultiviewExplainer(
            st.session_state["model"],
            device=st.session_state["device"]
        )
    return st.session_state["explainer"]

def render_prediction_page_image():
    if "img_tensor" not in st.session_state:
        st.session_state["img_tensor"]=None
    device=st.session_state["device"]
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
        st.session_state["img_tensor"] = img.unsqueeze(0).to(device)
        st.session_state["img_display"] = img_display
        st.session_state["true_label"] = class_name

    st.subheader("OR")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png","jpg","jpeg"]
    )
    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")

        transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        
        st.session_state["img_tensor"] = transform(image).unsqueeze(0).to(device)
        st.session_state["img_display"] = image
        st.session_state["true_label"] = "?"
    if st.session_state["img_display"] is not None:
        if st.session_state["true_label"] == "?":
            pass
        else:
            st.subheader(f"This should be {st.session_state['true_label']}")
        st.image(st.session_state["img_display"], width=200)

    if st.button("Predict"):
        if st.session_state["img_tensor"] is not None:
            if st.session_state["model"] is not None:
                run_prediction(st.session_state["img_tensor"])
                run_explanation(device,st.session_state["img_tensor"])
                st.toast("Prediciton Complete", icon="😍")
            else:
                st.warning("Please train the model first.")
        else:
            st.warning("Please select or upload an image first.")
