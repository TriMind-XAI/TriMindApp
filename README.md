# TriMindApp - Developing a Multi-view XAI tool with LLM-driven narrative explanations
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B.svg)](https://streamlit.io/)

A medical AI explanation tool that translates complex machine learning predictions into clear, natural language narratives that doctors and patients can understand

<img width="1406" height="756" alt="dashboard" src="https://github.com/user-attachments/assets/3769e8b1-8f6f-49da-92c6-f9c0d0f46836" />

##  Features

- **Multiple AI Models**: ResNet8, Custom CNN architectures 
- **4 XAI Methods**: Integrated Gradients, GradCAM, Saliency Maps, GradientSHAP
- **Consistency Analysis**: Cross-validation between explanation methods
- **LLM Narratives**: Audience-specific explanations (Clinician, Patient, Researcher)
- **Interactive Dashboard**: Streamlit-based web interface


##  Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/TriMindApp.git
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
Navigate to `http://localhost:8501` in your browser.

**Option 2: 
```bash
python -m streamlit run dashboard/app.py
```


##  Project Structure
<img width="352" height="829" alt="trimind_folder_structure" src="https://github.com/user-attachments/assets/c4ed5d1e-68f2-4778-9c9b-f8f331b5a02e" />







## 🙏 Acknowledgments

- **MedMNIST**: For providing standardized medical imaging datasets
- **Wisconsin**: For providing standardized medical imaging datasets
- **Captum**: Facebook's XAI library
- **Ollama**: Llama API for narrative generation
- **Hugging Face**: Open-source LLM infrastructure

