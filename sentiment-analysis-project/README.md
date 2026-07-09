# Sentiment Analysis & Opinion Mining

A comparative study of **Naive Bayes**, **Logistic Regression**, and **LSTM** models for binary sentiment classification on Amazon product reviews.

## Project Structure

```
sentiment-analysis-project/
├── data/                          # Raw and processed datasets
├── models/                        # Saved model weights and vectorizers
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA & labeling
│   ├── 02_preprocessing.ipynb      # Text cleaning pipeline
│   ├── 03_feature_engineering.ipynb # TF-IDF & sequences
│   ├── 04-06_model_training.ipynb  # Train NB, LR, LSTM
│   └── 07_ablation_study.ipynb     # Evaluation & comparison
├── results/
│   ├── plots/                     # Confusion matrices, ROC curves
│   └── Research_Report.md         # Final research paper
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # TextPreprocessor
│   ├── feature_extraction.py      # FeatureExtractor (TF-IDF + sequences)
│   ├── models.py                  # NaiveBayesModel, LogisticRegressionModel, LSTMModel
│   ├── evaluation.py              # ModelEvaluator
│   ├── data_collection.py         # Amazon review scraper
│   └── generate_mock_data.py      # Synthetic dataset generator
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install TensorFlow for LSTM
pip install tensorflow>=2.15

# 4. Generate mock data
python src/generate_mock_data.py

# 5. Launch Jupyter and run notebooks 01 → 07 in order
jupyter notebook
```

## Models

| Model               | Features | Expected F1 |
| -------------------- | -------- | ----------- |
| Naive Bayes          | TF-IDF   | ~77%        |
| Logistic Regression  | TF-IDF   | ~85%        |
| LSTM (Bidirectional) | Sequences| ~91%        |

## Web Dashboard (Product Interface)

We built an interactive, glassmorphism-themed Single Page Application to showcase the models in action. It features:
- **Live Sentiment Tester**: Run predictions in real-time across your active models.
- **Explainability Visualization**: Color-codes and underlines tokens based on their feature weights from the Logistic Regression coefficients (positive in green, negative in red).
- **Ablation Comparison**: Interactive Chart.js and tabular performance comparison.

To start the product:
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

## License

This project is for educational purposes.
