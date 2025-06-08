# 🎯 Sentiment Analysis using Deep Learning (IMDb Dataset)

This project performs binary sentiment classification (positive/negative) on movie reviews from the IMDb dataset using various deep learning models. Inspired by the [Skillcate AI YouTube tutorial](https://www.youtube.com/watch?v=oWo9SNcyxlI&t=1s), I enhanced it further with improved architecture, hyperparameter tuning, and evaluation strategies.

---

## 🔍 Objective

Build and evaluate multiple deep learning models for sentiment analysis of IMDb reviews using:

- Simple Neural Network (SNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM) with dropout and early stopping

---

## 📌 Highlights

- **Dataset:** IMDb 50,000 reviews (25k train, 25k test)
- **Embeddings:** Pre-trained 100D [GloVe](https://nlp.stanford.edu/projects/glove/)
- **Preprocessing:** Tokenization, padding (`maxlen=200`), embedding matrix with trainable weights
- **Architecture Improvements:**
  - Used **BiLSTM with Dropout** for better sequence modeling
  - **EarlyStopping** to avoid overfitting
  - Reduced **learning rate (0.0005)** for stable convergence
- **Achieved Test Accuracy:** **88.3%**
- **Saved Final Model:** `.h5` format for reuse
- **Predicted on Unseen Test Reviews**

---

## 🧠 Model Architecture (Final - BiLSTM)

```

Embedding (trainable=True) → BiLSTM(128) + Dropout(0.5) → Dense(1, sigmoid)

````

---

## 📊 Evaluation

| Model | Test Accuracy |
|-------|---------------|
| SNN   | ~83%          |
| CNN   | ~85%          |
| LSTM  | ~86%          |
| **BiLSTM** | **88.3%** ✅ |

- Used validation split = 0.2
- Metrics: accuracy, loss
- Plotted training & validation accuracy/loss curves

---

## 📦 Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
````

Main libraries:

* `tensorflow`, `keras`
* `numpy`, `pandas`
* `matplotlib`
* `sklearn`

---

## 🚀 Run Inference on New Data

```python
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model
model = load_model("c1_lstm_model_acc_0.883.h5")

# Example review
sample_review = ["This movie was absolutely wonderful and emotional"]

# Tokenize and pad (use tokenizer from training)
seq = tokenizer.texts_to_sequences(sample_review)
padded = pad_sequences(seq, maxlen=200)

# Predict
pred = model.predict(padded)
print("Sentiment:", "Positive" if pred[0][0] > 0.5 else "Negative")
```

---

## 📁 Files

* `lstm_model.ipynb`: Notebook with all model training and evaluation
* `glove.6B.100d.txt`: GloVe embeddings (download required separately)
* `c1_lstm_model_acc_0.883.h5`: Trained model (saved)
* `accuracy_loss_plot.png`: Accuracy/loss over epochs

---

## 🙋🏻‍♂️ Author

**Vaibhav Sharma**
[GitHub](https://github.com/vaisharma16) | [LinkedIn](https://www.linkedin.com/in/vaibhavsharma16/)

---

## 📜 Acknowledgements

* Video Credit: [Skillcate AI – Sentiment Analysis with LSTM](https://www.youtube.com/watch?v=v9dOY0zLkdc)
* IMDb Dataset from Keras Datasets
* GloVe: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
* 📦 Data & Model Files

To avoid GitHub’s file size limits, large files such as:
- `IMDB_Dataset.csv`
- `glove.6B.100d.txt`
- `lstm_model_acc_0.882.h5`

can be found in the [original repository](https://github.com/skillcate/sentiment-analysis-with-deep-neural-networks).

