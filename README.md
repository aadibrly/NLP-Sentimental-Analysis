# 🎬 IMDb Sentiment Analysis using SimpleRNN & Streamlit

An end-to-end Deep Learning Natural Language Processing (NLP) web application that analyzes user-provided movie reviews and classifies their sentiment as **Positive** or **Negative** using a Simple Recurrent Neural Network (SimpleRNN) and Streamlit.

---

## 📌 Project Overview

- **Task:** Binary Text Classification (Sentiment Analysis)
- **Dataset:** IMDb Movie Reviews dataset (Top 10,000 frequent words)
- **Model Architecture:** Embedding Layer $\rightarrow$ SimpleRNN $\rightarrow$ Dense (Sigmoid)
- **Frameworks & Libraries:** TensorFlow / Keras, Streamlit, NumPy
- **Deployment:** Interactive Streamlit UI

---

## 📁 Project Structure

```text
NLP-Sentimental-Analysis/
│
├── .streamlit/
│   └── config.toml               # Streamlit configuration settings
│
├── packages/
│   └── review_operations.py      # Text preprocessing, decoding, & inference logic
│
├── simple_rnn_imdb.h5            # Pretrained SimpleRNN model weights
├── SimpleRnn.ipynb               # Model training & evaluation notebook
├── prediction.ipynb              # Model testing & validation notebook
├── main.py                       # Streamlit web application
├── requirements.txt              # Project dependencies
├── .gitignore                    # Ignored files & directories
└── README.md                     # Project documentation