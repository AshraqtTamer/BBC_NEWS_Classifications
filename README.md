
# 📰 BBC News Classification using Multiple ML Models

An advanced **Text Classification System** built to categorize news articles into five distinct topics with high precision. This project utilizes **NLP (Natural Language Processing)** and a variety of **Machine Learning** algorithms to achieve state-of-the-art accuracy.

## 🎯 Project Overview

This repository demonstrates a complete **NLP pipeline** for classifying BBC News articles. The system processes raw text, extracts meaningful features using **TF-IDF**, and evaluates multiple classifiers to find the optimal model for news categorization.

## ⚙️ Features

* 🔍 **Advanced NLP Preprocessing:** Tokenization, stopword removal, and Lemmatization.
* 📊 **Feature Engineering:** Implementation of **TF-IDF Vectorization** for text-to-numeric conversion.
* 🤖 **Multi-Model Comparison:** Evaluates SVM, Random Forest, Naive Bayes, and Decision Trees.
* 📈 **Performance Metrics:** Comprehensive evaluation using Accuracy and F1-Scores.
* 📖 **Readability Analysis:** Integration of text complexity scores (Flesch & Dale-Chall).

## 📂 Repository Structure

```text
├── bbc-news-classifications.ipynb  # Main project notebook
├── bbc_news_text_complexity.csv    # Dataset used for training
├── requirements.txt                # List of dependencies
└── README.md                       # Project documentation

```

## 🧠 Model Comparison

The project compares five different machine learning architectures. The **Support Vector Machine (SVM)** emerged as the top performer due to its effectiveness in high-dimensional text spaces.

### Performance Leaderboard:

| Rank | Model | Accuracy | F1-Score |
| --- | --- | --- | --- |
| 🥇 | **SVM** | **0.978873** | **0.978953** |
| 🥈 | **Random Forest** | 0.976526 | 0.976592 |
| 🥉 | **Multinomial NB** | 0.976526 | 0.976567 |
| 4 | Gaussian NB | 0.896714 | 0.897640 |
| 5 | Decision Tree | 0.833333 | 0.833992 |

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/BBC-News-Classification.git
cd BBC-News-Classification

# Install dependencies
pip install -r requirements.txt

```

## 🧩 Usage

1. Open `bbc-news-classifications.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure the dataset `bbc_news_text_complexity_summarization.csv` is in the same directory.
3. Run all cells to perform data cleaning, feature extraction, and model training.
4. View the final **Confusion Matrix** and **Comparison Table** to analyze results.

## 🧾 Requirements

* `numpy` & `pandas`
* `scikit-learn`
* `nltk`
* `matplotlib` & `seaborn`
* `tensorflow` (for deep learning extensions)

## 🌐 Future Improvements

* Implement **Transformer models** (BERT/RoBERTa) for even higher accuracy.
* Develop a **Streamlit Web App** for real-time article classification.
* Expand to multi-label classification for articles covering multiple topics.

## 👨‍💻 Author

**Ashraqt Tamer**
📧 ashrakat123456@gmail.com 
🔗 [GitHub](https://github.com/AshraqtTamer)
