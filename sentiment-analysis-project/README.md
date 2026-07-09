# Sentiment Analysis & Opinion Mining

Comparative sentiment classification project for Amazon product reviews using **Naive Bayes**, **Logistic Regression**, and **LSTM**.

## Overview

This project covers a full text-classification workflow:

- data collection and labeling
- text preprocessing and cleaning
- TF-IDF and sequence-based feature extraction
- model training and evaluation
- a Flask web dashboard for live predictions and comparison

## Reported Metrics

The current comparison file at `results/model_comparison.csv` reports the following scores:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| LSTM | 1.00 | 1.00 | 1.00 | 1.00 |

## Project Structure

```
sentiment-analysis-project/
├── data/                          # Raw and processed datasets
├── models/                        # Saved model weights and vectorizers
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and labeling
│   ├── 02_preprocessing.ipynb      # Text cleaning pipeline
│   ├── 03_feature_engineering.ipynb # TF-IDF and sequences
│   ├── 04-06_model_training.ipynb  # Train NB, LR, LSTM
│   └── 07_ablation_study.ipynb     # Evaluation and comparison
├── results/
│   ├── plots/                      # Confusion matrices, ROC curves
│   └── Research_Report.md         # Final report
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # TextPreprocessor
│   ├── feature_extraction.py      # FeatureExtractor for TF-IDF and sequences
│   ├── models.py                  # NaiveBayesModel, LogisticRegressionModel, LSTMModel
│   ├── evaluation.py              # ModelEvaluator
│   ├── data_collection.py         # Amazon review scraper
│   └── generate_mock_data.py      # Synthetic dataset generator
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Optional: install TensorFlow for LSTM support
pip install tensorflow>=2.15

# 4. Generate mock data
python src/generate_mock_data.py

# 5. Launch Jupyter and run notebooks 01 to 07 in order
jupyter notebook
```

## Running the Web App

Start the Flask dashboard with:

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Features

- live sentiment prediction across the trained models
- explanation view for token-level contribution in logistic regression
- side-by-side comparison of accuracy, precision, recall, and F1 score

## License

This project is for educational purposes.
