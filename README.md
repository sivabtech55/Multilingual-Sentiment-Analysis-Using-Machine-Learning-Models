# Multilingual Sentiment Analysis Using Machine Learning

A sentiment analysis system capable of processing and classifying text in multiple languages using ML models (Logistic Regression, SVM, Random Forest).

## 🚀 Features

- Supports multilingual input with automatic language detection and translation to English.
- Preprocessing: Normalization, stop-word removal, and cleaning.
- Classifies sentiments as positive, negative, or neutral.
- Evaluates model accuracy and F1-score.

## 🧪 Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

## 📁 Project Structure

- `src/preprocess.py`: Preprocessing and translation logic
- `src/train_models.py`: Train and save ML models
- `src/evaluate_models.py`: Evaluate models on the dataset
- `data/sample_dataset.csv`: Sample data (text + label)
- `models/`: Trained model files

## 📦 Setup

```bash
pip install -r requirements.txt
