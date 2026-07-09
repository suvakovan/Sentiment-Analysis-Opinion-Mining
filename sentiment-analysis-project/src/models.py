"""
Model Definitions for Sentiment Analysis.

Provides three model wrappers with a unified API:
- NaiveBayesModel (scikit-learn MultinomialNB)
- LogisticRegressionModel (scikit-learn)
- LSTMModel (TensorFlow / Keras)
"""

import os
import pickle
import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Graceful TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (Embedding, LSTM as KerasLSTM,
                                         Dense, Dropout, Bidirectional)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.warning("TensorFlow not found. LSTMModel will be unavailable.")


# ======================================================================
# Base class
# ======================================================================
class BaseModel:
    """Abstract base class for all sklearn-based models."""

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, **kwargs):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def predict_proba(self, X_test):
        raise NotImplementedError

    def save(self, filepath):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", filepath)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("Model loaded from %s", filepath)


# ======================================================================
# Traditional ML models
# ======================================================================
class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes classifier."""

    def __init__(self, alpha=1.0):
        """
        Args:
            alpha: Additive (Laplace) smoothing parameter.
        """
        super().__init__()
        self.model = MultinomialNB(alpha=alpha)

    def train(self, X_train, y_train, **kwargs):
        logger.info("Training Naive Bayes (alpha=%.2f) ...", self.model.alpha)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""

    def __init__(self, max_iter=1000, C=1.0):
        """
        Args:
            max_iter: Maximum iterations for solver convergence.
            C: Inverse regularization strength.
        """
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)

    def train(self, X_train, y_train, **kwargs):
        logger.info("Training Logistic Regression (C=%.2f) ...", self.model.C)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]


# ======================================================================
# Deep Learning model
# ======================================================================
class LSTMModel:
    """Bidirectional LSTM model built with Keras."""

    def __init__(self, vocab_size=5000, embedding_dim=100, max_length=100,
                 lstm_units=(64, 32), dropout_rate=0.3):
        """
        Args:
            vocab_size: Size of the vocabulary (embedding input_dim).
            embedding_dim: Dimensionality of word embeddings.
            max_length: Input sequence length.
            lstm_units: Tuple of units for each stacked LSTM layer.
            dropout_rate: Dropout fraction after each LSTM layer.
        """
        if not _HAS_TF:
            raise RuntimeError(
                "TensorFlow is required for LSTMModel. "
                "Install it with: pip install tensorflow"
            )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        layers = [
            Embedding(input_dim=self.vocab_size,
                      output_dim=self.embedding_dim,
                      input_length=self.max_length),
        ]
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)
            layers.append(Bidirectional(KerasLSTM(units, return_sequences=return_seq)))
            layers.append(Dropout(self.dropout_rate))

        layers.append(Dense(32, activation='relu'))
        layers.append(Dense(1, activation='sigmoid'))

        model = Sequential(layers)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64,
              model_path='models/lstm_model.keras'):
        logger.info("Training LSTM Model (%d epochs) ...", epochs)
        dirpath = os.path.dirname(model_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return (preds > 0.5).astype(int).flatten()

    def predict_proba(self, X_test):
        return self.model.predict(X_test).flatten()

    def save(self, filepath):
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.model.save(filepath)
        logger.info("LSTM Model saved to %s", filepath)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = load_model(filepath)
        logger.info("LSTM Model loaded from %s", filepath)
