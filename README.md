# 🎬 IMDb Sentiment Analysis using SimpleRNN & Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlp-sentimental-analysis-broly.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/aadibrly/NLP-Sentimental-Analysis)

An end-to-end Deep Learning Natural Language Processing (NLP) web application that analyzes movie reviews and classifies sentiment as **Positive** or **Negative** using a Simple Recurrent Neural Network (SimpleRNN) and Streamlit.

🔗 **Live Demo:** [https://nlp-sentimental-analysis-broly.streamlit.app/](https://nlp-sentimental-analysis-broly.streamlit.app/)

---

## 📌 Project Overview

- **Task:** Binary Text Classification (Sentiment Analysis)
- **Dataset:** IMDb Movie Reviews dataset (Top 10,000 frequent words)
- **Model Architecture:** Embedding Layer $\rightarrow$ SimpleRNN $\rightarrow$ Dense (Sigmoid)
- **Live Deployment:** Hosted on Streamlit Community Cloud
- **Frameworks & Libraries:** TensorFlow / Keras, Streamlit, NumPy

---

## 📁 Project Structure

```text
NLP-Sentimental-Analysis/
│
├── .streamlit/
│   └── config.toml               # Streamlit UI & theme configuration
│
├── packages/
│   └── review_operations.py      # Text preprocessing, OOV handling & prediction logic
│
├── simple_rnn_imdb.h5            # Pretrained SimpleRNN model weights
├── SimpleRnn.ipynb               # Model training & evaluation notebook
├── prediction.ipynb              # Model testing & validation notebook
├── main.py                       # Streamlit web application
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore configuration
└── README.md                     # Project documentation