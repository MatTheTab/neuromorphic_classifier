# neuromorphic_classifier
This repository trains a spiking neural network (SNN) classifier on the MNIST dataset using various spike encoding techniques. It explores different encoding schemes to convert images into spike trains and evaluates their impact on classification performance with the help of the SNNTorch module.

## 📂 Directory Structure

```
📂 project_root │
├── 📂 models # Pre-trained neuromorphic models for image classification
│ ├── 📄 neuromorphic_delta_model.pth # Spiking Neural Network trained using delta encoding
│ ├── 📄 neuromorphic_rate_model.pth # Spiking Neural Network trained using rate encoding
│ ├── 📄 neuromorphic_temporal_model.pth # Spiking Neural Network trained using latency encoding
│
├── 📂 notebooks # Jupyter notebooks for training and inference
│ ├── 📄 Neuromorphic_Spiking_CNN.ipynb # Notebook for training the models
│ ├── 📄 Neuromorphic_Spiking_CNN__Gradio_App.ipynb # Notebook for running a Gradio demo app
│
├── 📄 .gitignore # gitignore file for handling external files and directories
├── 📄 neuromorphic_demo.py # Python file for running the Streamlit application
├── 📄 requirements.txt # Environment details necessary to run the experiments
├── 📄 README.md # Project documentation and instructions
```

## Live Demo

To run the Streamlit Demo simply click the link [here](https://neuromorphicclassifier-zcndlmxf4gqlnwdnk9eucj.streamlit.app).
Or if you prefer the Gradio App Demo in a Google Colab notebook, then simply run this [notebook](https://colab.research.google.com/drive/1nb5gLPCgLxLN4RBbZTS5tXKT1nDWyUo0?usp=sharing).