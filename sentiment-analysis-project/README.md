# Sentiment Analysis & Opinion Mining

<p align="center">
	<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&center=true&vCenter=true&width=760&lines=Sentiment+Analysis+%26+Opinion+Mining;Naive+Bayes+%7C+Logistic+Regression+%7C+LSTM;Amazon+Review+Classification+Dashboard" alt="Animated typing header" />
</p>

<p align="center">
	<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,100:1d4ed8&height=160&section=header&text=Opinion%20Mining%20Project&fontSize=38&fontColor=ffffff&animation=fadeIn" alt="Animated wave header" />
</p>

<p align="center">
	Comparative sentiment classification project for Amazon product reviews using <strong>Naive Bayes</strong>, <strong>Logistic Regression</strong>, and <strong>LSTM</strong>.
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Models-3-0f766e?style=for-the-badge" alt="Models count" />
	<img src="https://img.shields.io/badge/Task-Binary%20Sentiment%20Classification-1d4ed8?style=for-the-badge" alt="Task badge" />
	<img src="https://img.shields.io/badge/Interface-Flask%20Dashboard-111827?style=for-the-badge" alt="Interface badge" />
</p>

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
