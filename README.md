# Fake News Detection: LSTM vs. BERT vs. Hybrid BERT-LSTM

A comparative study of deep learning architectures for automated fake news classification, evaluating a sequential LSTM model, a fine-tuned BERT transformer, and a hybrid BERT+LSTM model on the FakeNewsNet dataset.

## Overview

Fake news detection is framed here as a binary text classification problem (Real vs. Fake). This project implements and benchmarks three distinct deep learning approaches to determine which architecture best captures the linguistic patterns that distinguish fake news from real news.

## Models Compared

1. **LSTM** — A bidirectional LSTM operating on word embeddings, capturing sequential dependencies in text.
2. **BERT** — A fine-tuned `bert-base-uncased` transformer, leveraging pre-trained contextual language understanding.
3. **Hybrid BERT+LSTM** — BERT-derived contextual embeddings fed into a bidirectional LSTM layer, combining transformer-level context with explicit sequence modeling.

## Dataset

[`mdepak/fakenewsnet`](https://www.kaggle.com/datasets/mdepak/fakenewsnet) on Kaggle — derived from the FakeNewsNet repository (Politifact and GossipCop sources), containing labeled real and fake news articles.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| LSTM | 0.7595 | 0.5769 | 0.7595 | 0.6557 | 0.4972 |
| **BERT** | **0.8585** | **0.8523** | **0.8585** | **0.8510** | **0.8679** |
| Hybrid BERT+LSTM | 0.8433 | 0.8351 | 0.8433 | 0.8336 | 0.8573 |

**Key finding:** Standalone BERT outperformed both the LSTM baseline and the Hybrid BERT+LSTM model across every metric, suggesting that BERT's pre-trained contextual representations alone are more effective for this task than adding a sequential LSTM layer on top.

## Project Structure

\```
fake-news-detection-bert-lstm/
├── notebooks/
│   └── fake-news-detection-notebook.ipynb
├── results/
│   └── model_comparison_summary.png
├── requirements.txt
└── README.md
\```

## Setup

\```bash
git clone https://github.com/<your-username>/fake-news-detection-bert-lstm.git
cd fake-news-detection-bert-lstm
pip install -r requirements.txt
\```

## Dataset Setup

\```bash
kaggle datasets download -d mdepak/fakenewsnet
unzip fakenewsnet.zip -d data/
\```

## Requirements

torch>=1.9.0
transformers>=4.0.0
scikit-learn>=0.24.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0


## Running the Notebook

Open `notebooks/fake-news-detection-notebook.ipynb` in Jupyter, VS Code, Google Colab, or Kaggle Notebooks. It covers data preprocessing, model definitions (LSTM, BERT, Hybrid BERT+LSTM), training/validation, and comparative evaluation.
