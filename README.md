# TriMindApp - Developing a Multi-view XAI tool with LLM-driven narrative explanations
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B.svg)](https://streamlit.io/)

A medical AI explanation tool that translates complex machine learning predictions into clear, natural language narratives that doctors and patients can understan

##  Features

- **Multiple AI Models**: ResNet8, Custom CNN architectures (image data), ML Models and MLP (tabular data)
- **6 XAI Methods**: Integrated Gradients, GradCAM, Saliency Maps, GradientSHAP, SHAP, Counterfacuals
- **Consistency Analysis**: Cross-validation between explanation methods using Spearman
- **LLM Narratives**: Audience-specific explanations (Clinician, Patient, Researcher)
- **Interactive Dashboard**: Streamlit-based web interface

## Important: Notebooks Folder
**The Notebooks folder contains standalone experimental workflows used during development**
They can be used to review individual parts of the project without running the full dashboard.
**Used in the final project development**
- notebooks/Pneumonia_MNIST.ipynb Image classification on PneumoniaMNIST dataset
- notebooks/Image_medmnist_XAI.ipynb XAI and LLM implementaion for PneumoniaMNIST
- notebooks/Tabular_shap.ipynb SHAP experiments for tabular data models training and XAI.
- notebooks/Counterfactual_Shap_Tab.ipynb Tabular counterfactual and SHAP experiments.
**Experimental / not included in the final system**
- notebooks/3D_Image.ipynb Exploratory work with 3D image data.
- notebooks/counterfactual_3D_image.ipynb Exploratory work with 3D image data Counterfactuals.
Experimental attempt to apply counterfactual explanation ideas to 3D image data; not included due to scope and implementation limits.

## System Workflow
<img width="3026" height="164" alt="sys workflow" src="https://github.com/user-attachments/assets/24ed66cd-d949-48af-a273-fd4007727633" />

##  Quick Start

### Prerequisites

- Python 3.12 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/TriMind-XAI/TriMindApp.git
cd TriMindApp
```

2. **Create virtual environment**
```bash
python -m venv trimindvenv
source trimindvenv/bin/activate  # On Windows: trimindvenv\Scripts\activate
```

3. **Install PyTorch (with CUDA support)**
```bash
# For GPU (CUDA 11.8)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables**
```bash

#  use Hugging Face token
echo "HUGGINGFACE_TOKEN=your-token-here" >> .env
```
### Running the Application

**Option 1: Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

**Option 2: 
```bash
python -m streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` in your browser.
##  Project Structure

##  Step-by-Step User Guide

### Login(authentication)
You can either register newly or login through google
<img width="1132" height="754" alt="login" src="https://github.com/user-attachments/assets/a3fad640-0003-47ee-b3ac-fe18ace2837e" />

### Step 1: Select Data Type

When you first launch the application, you'll see the data selection interface.

<img width="1165" height="593" alt="first_select_data" src="https://github.com/user-attachments/assets/e85e98b0-2a62-4bfa-8ff1-e129faf3e1f8" />


**Choose your data type:**
- Click on the "Data" tab in the sidebar
- Choose "Tabular" or "Image" from the dropdown
- Upload your dataset or use pre-loaded samples (PneumoniaMNIST, BreastMNIST)
- 
#### **Option A: Tabular Data**
- For structured medical records (Wisconsin Breast Cancer dataset)

#### **Option B: Image Data**
- For medical imaging (MedMnist datasets)

**What Happens:**
- System automatically configures appropriate preprocessing
- Displays data preview and statistics
- Suggests compatible model architectures
---

### Step 2: Train the Model

After selecting your data, navigate to the Training tab to build your AI model.

<img width="1172" height="587" alt="second_train_model" src="https://github.com/user-attachments/assets/8d46148c-b8ca-4ea8-b3a6-8b3221d7bb6a" />

**What Happens:**
- Shows training progress
- Displays the model performance results

### Step 3: Predict & Analyze with XAI

Now that your model is trained, make predictions on new images.

<img width="1174" height="581" alt="third_predict" src="https://github.com/user-attachments/assets/5cff6eda-7c31-47fd-94cf-3fedfd1f4008" />

**What Happens:**
- Shows prediciton/expected results
- Displays XAI technical results
---

### Step 4: Get LLM Persona Explanations

Transform technical XAI results into natural language narratives for different audiences.

<img width="1185" height="588" alt="fourth_explain" src="https://github.com/user-attachments/assets/97e3344f-8168-41c9-bcad-3f76a094dbc3" />


#### **Three Persona Options:**

### **Persona 1:  Clinician**

**Target Audience:** Doctors, Radiologists, Medical Professionals


### **Persona 3:  Researcher**

**Target Audience:** AI Researchers, Data Scientists, ML Engineers, Auditors

### **Persona 3:  Patient**

**Target Audience:** Patients, Family Members, Non-Medical Individuals


## Acknowledgments

- **MedMNIST**: For providing standardized medical imaging datasets
- **Wisconsin**: For providing standardized medical imaging datasets
- **Captum**: Facebook's XAI library
- **Meta Llama**: Llama API for narrative generation
- **Hugging Face**: Open-source LLM infrastructure
- **Kapcia, M., Eshkiki, H., Duell, J., Fan, X., Zhou, S., & Mora, B. (n.d.). ExMed**: An AI tool for experimenting explainable AI techniques on medical data analytics
## Contributors
- [Aderonke Adelabu](https://www.linkedin.com/in/aderonke-kausar-adelabu/)
- [Akinpelumi Ajayi](https://www.linkedin.com/in/akinpelumi-michael-ajayi-/)
- [Amjad Alhaffar](https://www.linkedin.com/in/amjadalhaffarsyr/)

**Supervisor: Dr. Jamie Duell**

### Sheffield Hallam University

<img width="280" height="188" alt="uni logo" src="https://github.com/user-attachments/assets/6a9dd6f2-756d-48b9-978b-20636fc2a57d" />
