"""
Text Preprocessing Module for Sentiment Analysis.

Provides a configurable TextPreprocessor that handles:
- Text cleaning (HTML, URLs, special characters)
- Tokenization via NLTK
- Configurable stopword removal (preserves sentiment modifiers by default)
- Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Configurable text preprocessing pipeline for sentiment analysis."""

    # Default words to keep even though they are in NLTK stopwords,
    # because they carry sentiment-modifying meaning.
    DEFAULT_KEEP_WORDS = ['not', 'no', 'nor', 'but', 'very', 'only',
                          'against', 'too', 'few', 'more', 'most']

    def __init__(self, remove_stopwords=True, use_lemmatizer=True,
                 use_stemmer=False, keep_words=None, min_token_length=2):
        """
        Initializes the text preprocessor.

        Args:
            remove_stopwords: Whether to remove stopwords.
            use_lemmatizer: Whether to apply lemmatization.
            use_stemmer: Whether to apply stemming (mutually exclusive with lemmatizer;
                         if both are True, lemmatizer takes priority).
            keep_words: List of words to preserve even if they are stopwords.
                        Defaults to DEFAULT_KEEP_WORDS.
            min_token_length: Minimum character length for a token to be kept.
        """
        self._download_nltk_data()

        self.remove_stopwords = remove_stopwords
        self.use_lemmatizer = use_lemmatizer
        self.use_stemmer = use_stemmer and not use_lemmatizer
        self.min_token_length = min_token_length

        # Build stopword set
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
            keep = keep_words if keep_words is not None else self.DEFAULT_KEEP_WORDS
            self.stop_words -= set(keep)
        else:
            self.stop_words = set()

        self.lemmatizer = WordNetLemmatizer() if self.use_lemmatizer else None
        self.stemmer = PorterStemmer() if self.use_stemmer else None

    # ------------------------------------------------------------------
    # NLTK data download (robust across NLTK versions)
    # ------------------------------------------------------------------
    @staticmethod
    def _download_nltk_data():
        """Downloads required NLTK resources if not already present."""
        resources = {
            'punkt': 'tokenizers/punkt',
            'punkt_tab': 'tokenizers/punkt_tab',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet',
            'omw-1.4': 'corpora/omw-1.4',
        }
        for name, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info("Downloading NLTK resource: %s", name)
                nltk.download(name, quiet=True)

    # ------------------------------------------------------------------
    # Individual pipeline steps (public so users can call them standalone)
    # ------------------------------------------------------------------
    def clean_text(self, text):
        """Lowercase, strip HTML/URLs/non-alpha, collapse whitespace."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)           # HTML tags
        text = re.sub(r'http\S+|www\.\S+', '', text) # URLs
        text = re.sub(r'[^a-z\s]', ' ', text)        # non-alpha
        text = re.sub(r'\s+', ' ', text).strip()      # extra whitespace
        return text

    def tokenize(self, text):
        """Tokenize text, optionally remove stopwords and short tokens."""
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        return tokens

    def normalize_tokens(self, tokens):
        """Apply lemmatization or stemming to a list of tokens."""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(w) for w in tokens]
        if self.stemmer:
            return [self.stemmer.stem(w) for w in tokens]
        return tokens

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess(self, text, return_string=True):
        """
        Run the complete preprocessing pipeline.

        Args:
            text: Raw input string.
            return_string: If True, return a space-joined string; else a token list.

        Returns:
            Cleaned text as str or list[str].
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.normalize_tokens(tokens)

        return " ".join(tokens) if return_string else tokens


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = TextPreprocessor()
    samples = [
        "This product is AMAZING!!! Best purchase ever 😍 #5stars but it was not cheap.",
        "Terrible quality. <b>DO NOT BUY!</b> Waste of money... http://bad-link.com",
        "",
        None,
        12345,
    ]
    for s in samples:
        result = preprocessor.preprocess(s)
        print(f"Original : {s!r}")
        print(f"Processed: {result!r}\n")
