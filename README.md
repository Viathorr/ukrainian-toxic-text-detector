# 🛡️ Ukrainian Toxicity Detector

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-tracked-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/viathorrr/Ukrainian-Toxic-Comments-Classification)
[![Model](https://img.shields.io/badge/Model-XLM--RoBERTa-blueviolet)](https://huggingface.co/viathorr/xlm-roberta-multi-label-uk-toxic-comments-v14)
![Macro F1](https://img.shields.io/badge/Macro--F1-0.7237-brightgreen)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/viathorr/ukrainian-toxicity-detector)

A multi-label toxicity classification system for Ukrainian-language text, powered by **XLM-RoBERTa**. This project explores multiple approaches to detecting toxic content across six categories, with an interactive web demo built with Gradio.

***

## 📌 Overview

This project addresses the lack of NLP resources for the Ukrainian language by building a multi-label toxicity classifier capable of detecting the following categories:

| Category | Description |
|---|---|
| `toxic` | General toxic content |
| `severe_toxic` | Highly offensive content |
| `obscene` | Obscene language |
| `threat` | Threatening messages |
| `insult` | Insulting content |
| `identity_hate` | Hate speech targeting identity groups |

***

## 🚀 Approaches

Three approaches were implemented and compared. Approaches [1](https://www.kaggle.com/code/viathorr/training-bilstm-for-uk-toxicity-classification) and [2](https://www.kaggle.com/code/viathorr/xlm-roberta-toxicity-classification) were explored in separate Kaggle notebooks and are not part of this repository. Only the best-performing approach (3) is implemented here.

### 1. BiLSTM (Baseline)
A recurrent neural network baseline using Bidirectional LSTM with pre-trained word embeddings.

### 2. XLM-RoBERTa — Zero-Shot Cross-Lingual Transfer
The `xlm-roberta-base` model fine-tuned on the English Jigsaw dataset and evaluated directly on Ukrainian text without any Ukrainian training data.

### 3. XLM-RoBERTa — Fine-Tuned on Translated Data ⭐ Best
The `xlm-roberta-base` model fine-tuned on the Jigsaw dataset **translated into Ukrainian**. This approach achieved the best results.

***

## 📊 Results

| Model | Macro-F1 |
|---|---|
| BiLSTM | 0.5568 |
| EN XLM-R (zero-shot) | 0.5250 |
| **UK XLM-R (fine-tuned)** | **0.7237** |

### Per-Category F1-Score

| Model | toxic | severe_toxic | obscene | threat | insult | identity_hate |
|---|---|---|---|---|---|---|
| BiLSTM | 0.7507 | 0.4332 | 0.5836 | 0.5505 | 0.6629 | 0.3601 |
| EN XLM-R (zero-shot) | 0.7971 | 0.0250 | 0.4921 | 0.6014 | 0.7150 | 0.5191 |
| **UK XLM-R (fine-tuned)** | **0.8501** | **0.6383** | **0.7344** | **0.7135** | **0.7991** | **0.6066** |

***

## ⚙️ Best Model Configuration

- **Architecture:** `xlm-roberta-base`
- **Frozen layers:** Embedding layer (`embeddings`)
- **Trainable layers:** All 12 encoder transformer layers (`encoder.layer`)
- **Optimizer:** AdamW
- **Learning rate:** 2×10⁻⁵
- **Weight decay:** 10⁻²
- **Warmup:** Linear warmup for first 20% of steps
- **Epochs:** 6
- **Batch size:** 32 (train) / 64 (eval)
- **Primary metric:** Macro-F1

> 🤗 **Fine-tuned Model on Hugging Face:** [viathorr/xlm-roberta-multi-label-uk-toxic-comments-v14](https://huggingface.co/viathorr/xlm-roberta-multi-label-uk-toxic-comments-v14)

***

## 🗃️ Dataset

- **Primary training data:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) datasets from Kaggle — translated into Ukrainian
- **Auxiliary test data:** Ukrainian toxicity binary classification dataset from [Hugging Face](https://huggingface.co/datasets/ukr-detect/ukr-toxicity-dataset) (5,000 records, 1:1 balance)
- **Test set size:** 2,000 records 

> 🔜 **Coming soon**: The translated training/validation datasets and the final test set will be made publicly available on Hugging Face and/or Kaggle.
***

## 🖥️ Web Application

An interactive demo was built using **Gradio** and deployed on **Hugging Face Spaces**.

**Features:**
- Text input for Ukrainian comments
- Verdict output with detected toxicity categories
- Bar chart showing per-category probabilities
- Built-in example comments
- Toggle between Default (0.5) and Tuned (optimized) thresholds

> 🔗 **Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/viathorr/ukrainian-toxicity-detector)

***

## 📦 Installation

```bash
git clone https://github.com/Viathorr/ukrainian-toxic-text-detector.git
cd ukrainian-toxic-text-detector
pip install -r requirements.txt
```

***

## 🔬 Training

```bash
python scripts/train.py
```

Experiments are tracked via **Weights & Biases**. Make sure to set your W&B API key before training.

***

## 🌐 Run the Web App Locally

```bash
python app/app.py
```

The Gradio interface will launch at `http://localhost:7860`.
