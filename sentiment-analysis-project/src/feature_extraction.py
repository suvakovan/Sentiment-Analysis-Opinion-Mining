"""
Feature Extraction Module for Sentiment Analysis.

Provides TF-IDF vectorization and Keras tokenizer/sequence generation,
with graceful fallback when TensorFlow is not installed.
"""

import os
import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Graceful TensorFlow import — allows the project to work
# for traditional ML even if TF is not installed.
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    logger.warning("TensorFlow not found. LSTM sequence features will be unavailable.")


class FeatureExtractor:
    """Extracts TF-IDF and/or sequence features from text data."""

    def __init__(self, max_features=5000, max_sequence_length=100,
                 ngram_range=(1, 2), sublinear_tf=True):
        """
        Args:
            max_features: Maximum vocabulary size for both TF-IDF and tokenizer.
            max_sequence_length: Pad/truncate sequences to this length (for LSTM).
            ngram_range: N-gram range for TF-IDF (default unigrams + bigrams).
            sublinear_tf: Apply sublinear TF scaling (log normalization).
        """
        self.max_features = max_features
        self.max_sequence_length = max_sequence_length

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )

        self.tokenizer = None
        if _HAS_TF:
            self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------
    def fit_tfidf(self, texts):
        """Fit the TF-IDF vectorizer and transform texts."""
        logger.info("Fitting TF-IDF (max_features=%d) ...", self.max_features)
        return self.tfidf_vectorizer.fit_transform(texts)

    def transform_tfidf(self, texts):
        """Transform texts with the already-fitted TF-IDF vectorizer."""
        return self.tfidf_vectorizer.transform(texts)

    # ------------------------------------------------------------------
    # Keras sequences (for LSTM / embedding models)
    # ------------------------------------------------------------------
    def _check_tf(self):
        if not _HAS_TF or self.tokenizer is None:
            raise RuntimeError(
                "TensorFlow is required for sequence features. "
                "Install it with: pip install tensorflow"
            )

    def fit_sequences(self, texts):
        """Fit the Keras tokenizer and return padded sequences."""
        self._check_tf()
        logger.info("Fitting Keras Tokenizer and generating sequences ...")
        self.tokenizer.fit_on_texts(texts)
        return self._texts_to_padded(texts)

    def transform_sequences(self, texts):
        """Convert texts to padded sequences using the fitted tokenizer."""
        self._check_tf()
        return self._texts_to_padded(texts)

    def _texts_to_padded(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_sequence_length,
                             padding='post', truncating='post')

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, model_dir='models'):
        """Save the fitted vectorizer and tokenizer to disk."""
        os.makedirs(model_dir, exist_ok=True)

        tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        if self.tokenizer is not None:
            tok_path = os.path.join(model_dir, 'tokenizer.pkl')
            with open(tok_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)

        logger.info("Extractors saved to %s/", model_dir)

    def load(self, model_dir='models'):
        """Load the vectorizer and tokenizer from disk."""
        tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        else:
            logger.warning("%s not found.", tfidf_path)

        tok_path = os.path.join(model_dir, 'tokenizer.pkl')
        if os.path.exists(tok_path):
            with open(tok_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            logger.warning("%s not found.", tok_path)

    # Backward-compatible aliases for old notebook code
    fit_word2vec_sequences = fit_sequences
    transform_word2vec_sequences = transform_sequences
    save_extractors = save
    load_extractors = load
