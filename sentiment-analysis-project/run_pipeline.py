"""
run_pipeline.py - Execute the full Sentiment Analysis pipeline end-to-end.

Steps:
  1. Generate mock dataset (10,000 reviews)
  2. Label reviews (binary sentiment)
  3. Preprocess text
  4. Extract features (TF-IDF + sequences if TF available)
  5. Train models (Naive Bayes, Logistic Regression, LSTM if TF available)
  6. Evaluate and compare all models
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import NaiveBayesModel, LogisticRegressionModel
from src.evaluation import ModelEvaluator

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from src.models import LSTMModel
    HAS_TF = True
except (ImportError, RuntimeError):
    HAS_TF = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DIVIDER = "=" * 60


def step_banner(step_num, title):
    print(f"\n{DIVIDER}")
    print(f"  STEP {step_num}: {title}")
    print(DIVIDER)


# ==================================================================
# STEP 1 - Generate Data
# ==================================================================
def step1_generate_data():
    step_banner(1, "GENERATE MOCK DATASET")
    from src.generate_mock_data import generate_mock_dataset
    generate_mock_dataset(num_samples=10000, output_path='data/raw_reviews.csv', seed=42)


# ==================================================================
# STEP 2 - Label Reviews
# ==================================================================
def step2_label_data():
    step_banner(2, "LABEL REVIEWS (Binary Sentiment)")
    df = pd.read_csv('data/raw_reviews.csv')
    print(f"  Loaded {len(df)} reviews.")
    print(f"  Rating distribution:\n{df['rating'].value_counts().sort_index().to_string()}\n")

    # Drop 3-star for binary classification
    df_binary = df[df['rating'] != 3].copy()
    df_binary['sentiment'] = df_binary['rating'].apply(lambda x: 1 if x >= 4 else 0)

    print(f"  After removing 3-star: {len(df_binary)} reviews")
    print(f"  Sentiment distribution:\n{df_binary['sentiment'].value_counts().to_string()}")

    df_binary[['review_id', 'text', 'sentiment']].to_csv('data/labeled_reviews.csv', index=False)
    print("  Saved -> data/labeled_reviews.csv")
    return df_binary


# ==================================================================
# STEP 3 - Preprocess
# ==================================================================
def step3_preprocess():
    step_banner(3, "TEXT PREPROCESSING")
    df = pd.read_csv('data/labeled_reviews.csv')
    preprocessor = TextPreprocessor()

    print(f"  Processing {len(df)} reviews ...")
    start = time.time()
    df['cleaned_text'] = df['text'].apply(lambda x: preprocessor.preprocess(x))
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")

    # Show a sample
    sample = df.sample(3, random_state=1)[['text', 'cleaned_text', 'sentiment']]
    print("\n  Sample:")
    for _, row in sample.iterrows():
        print(f"    RAW : {row['text'][:80]}...")
        print(f"    CLEAN: {row['cleaned_text'][:80]}")
        print(f"    LABEL: {'Positive' if row['sentiment'] == 1 else 'Negative'}\n")

    # Drop empties
    before = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    print(f"  Dropped {before - len(df)} empty reviews after cleaning.")

    df.to_csv('data/cleaned_reviews.csv', index=False)
    print("  Saved -> data/cleaned_reviews.csv")


# ==================================================================
# STEP 4 - Feature Extraction + Train/Val/Test Split
# ==================================================================
def step4_features():
    step_banner(4, "FEATURE EXTRACTION (TF-IDF + Sequences)")
    df = pd.read_csv('data/cleaned_reviews.csv')
    df = df.dropna(subset=['cleaned_text'])

    X = df['cleaned_text'].values
    y = df['sentiment'].values

    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    extractor = FeatureExtractor(max_features=5000, max_sequence_length=100)

    # TF-IDF
    print("  Fitting TF-IDF ...")
    X_train_tfidf = extractor.fit_tfidf(X_train)
    X_val_tfidf = extractor.transform_tfidf(X_val)
    X_test_tfidf = extractor.transform_tfidf(X_test)
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")

    # Save TF-IDF
    scipy.sparse.save_npz('data/X_train_tfidf.npz', X_train_tfidf)
    scipy.sparse.save_npz('data/X_val_tfidf.npz', X_val_tfidf)
    scipy.sparse.save_npz('data/X_test_tfidf.npz', X_test_tfidf)

    # Sequences (only if TF available)
    X_train_seq = X_val_seq = X_test_seq = None
    if HAS_TF:
        print("  Fitting Keras Tokenizer ...")
        X_train_seq = extractor.fit_sequences(X_train)
        X_val_seq = extractor.transform_sequences(X_val)
        X_test_seq = extractor.transform_sequences(X_test)
        print(f"  Sequence shape: {X_train_seq.shape}")
        np.save('data/X_train_seq.npy', X_train_seq)
        np.save('data/X_val_seq.npy', X_val_seq)
        np.save('data/X_test_seq.npy', X_test_seq)
    else:
        print("  [SKIP] Sequences - TensorFlow not installed.")

    # Save labels
    np.save('data/y_train.npy', y_train)
    np.save('data/y_val.npy', y_val)
    np.save('data/y_test.npy', y_test)

    extractor.save('models')
    print("  All features saved to data/ and models/")


# ==================================================================
# STEP 5 - Model Training
# ==================================================================
def step5_train():
    step_banner(5, "MODEL TRAINING")

    y_train = np.load('data/y_train.npy')
    y_val = np.load('data/y_val.npy')
    X_train_tfidf = scipy.sparse.load_npz('data/X_train_tfidf.npz')
    X_val_tfidf = scipy.sparse.load_npz('data/X_val_tfidf.npz')

    timings = {}

    # --- Naive Bayes ---
    print("\n  [1/3] Naive Bayes")
    nb = NaiveBayesModel()
    t0 = time.time()
    nb.train(X_train_tfidf, y_train)
    timings['Naive Bayes'] = time.time() - t0
    nb.save('models/naive_bayes.pkl')
    print(f"        Trained in {timings['Naive Bayes']:.2f}s")

    # --- Logistic Regression ---
    print("\n  [2/3] Logistic Regression")
    lr = LogisticRegressionModel()
    t0 = time.time()
    lr.train(X_train_tfidf, y_train)
    timings['Logistic Regression'] = time.time() - t0
    lr.save('models/logistic_regression.pkl')
    print(f"        Trained in {timings['Logistic Regression']:.2f}s")

    # --- LSTM ---
    if HAS_TF:
        print("\n  [3/3] LSTM (Deep Learning)")
        X_train_seq = np.load('data/X_train_seq.npy')
        X_val_seq = np.load('data/X_val_seq.npy')
        lstm = LSTMModel(vocab_size=5000, embedding_dim=100, max_length=100)
        t0 = time.time()
        lstm.train(X_train_seq, y_train, X_val_seq, y_val,
                   epochs=5, batch_size=64, model_path='models/lstm_model.keras')
        timings['LSTM'] = time.time() - t0
        print(f"        Trained in {timings['LSTM']:.2f}s")
    else:
        print("\n  [3/3] LSTM - SKIPPED (TensorFlow not installed)")

    print("\n  Training Times:")
    for name, t in timings.items():
        print(f"    {name:25s}: {t:.2f}s")


# ==================================================================
# STEP 6 - Evaluation
# ==================================================================
def step6_evaluate():
    step_banner(6, "EVALUATION & COMPARISON")

    y_test = np.load('data/y_test.npy')
    X_test_tfidf = scipy.sparse.load_npz('data/X_test_tfidf.npz')

    evaluator = ModelEvaluator(output_dir='results/plots')

    # Load & predict — Naive Bayes
    nb = NaiveBayesModel()
    nb.load('models/naive_bayes.pkl')
    y_pred_nb = nb.predict(X_test_tfidf)
    y_prob_nb = nb.predict_proba(X_test_tfidf)

    # Load & predict — Logistic Regression
    lr = LogisticRegressionModel()
    lr.load('models/logistic_regression.pkl')
    y_pred_lr = lr.predict(X_test_tfidf)
    y_prob_lr = lr.predict_proba(X_test_tfidf)

    # Metrics
    metrics = {
        'Naive Bayes': evaluator.calculate_metrics(y_test, y_pred_nb),
        'Logistic Regression': evaluator.calculate_metrics(y_test, y_pred_lr),
    }

    # LSTM (if available)
    if HAS_TF and os.path.exists('models/lstm_model.keras'):
        X_test_seq = np.load('data/X_test_seq.npy')
        lstm = LSTMModel(vocab_size=5000, embedding_dim=100, max_length=100)
        lstm.load('models/lstm_model.keras')
        y_pred_lstm = lstm.predict(X_test_seq)
        y_prob_lstm = lstm.predict_proba(X_test_seq)
        metrics['LSTM'] = evaluator.calculate_metrics(y_test, y_pred_lstm)

    # Print comparison table
    comparison = evaluator.generate_comparison_table(metrics)
    print("\n  Model Comparison:")
    print(comparison.round(4).to_string())
    comparison.to_csv('results/model_comparison.csv')
    print("  Saved -> results/model_comparison.csv")

    # Classification reports
    evaluator.print_classification_report(y_test, y_pred_nb, "Naive Bayes")
    evaluator.print_classification_report(y_test, y_pred_lr, "Logistic Regression")

    # Confusion matrices
    evaluator.plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")
    evaluator.plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

    # ROC curves
    roc_probs = {'Naive Bayes': y_prob_nb, 'Logistic Regression': y_prob_lr}
    if 'LSTM' in metrics:
        evaluator.print_classification_report(y_test, y_pred_lstm, "LSTM")
        evaluator.plot_confusion_matrix(y_test, y_pred_lstm, "LSTM")
        roc_probs['LSTM'] = y_prob_lstm

    evaluator.plot_roc_curves(y_test, roc_probs)

    # Metrics bar chart
    evaluator.plot_metrics_comparison(metrics)

    print("\n  Plots saved to results/plots/")
    print(f"  Files: {os.listdir('results/plots')}")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    print(DIVIDER)
    print("  SENTIMENT ANALYSIS - FULL PIPELINE")
    print(DIVIDER)
    print(f"  TensorFlow available: {HAS_TF}")

    pipeline_start = time.time()

    step1_generate_data()
    step2_label_data()
    step3_preprocess()
    step4_features()
    step5_train()
    step6_evaluate()

    total = time.time() - pipeline_start
    print(f"\n{DIVIDER}")
    print(f"  PIPELINE COMPLETE - Total time: {total:.1f}s")
    print(DIVIDER)
