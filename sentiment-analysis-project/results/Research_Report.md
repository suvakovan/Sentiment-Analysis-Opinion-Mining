# Sentiment Analysis & Opinion Mining: A Comparative Study

## 1. ABSTRACT
This project investigates the performance of different Machine Learning and Deep Learning architectures on the task of Sentiment Analysis using a dataset of Amazon Product Reviews. We compare traditional machine learning approaches (Naive Bayes, Logistic Regression) using TF-IDF vectorization against a Deep Learning approach (LSTM) utilizing Word Embeddings. Our ablation study demonstrates that while traditional ML models offer significant speed advantages, the LSTM architecture captures sequential context and achieves superior performance metrics (F1-score).

## 2. INTRODUCTION
Sentiment analysis is a critical Natural Language Processing (NLP) task with massive business value. Companies utilize opinion mining to automatically gauge customer feedback at scale. In this study, we define a binary classification problem to predict whether an Amazon review is positive or negative. The primary challenge in sentiment analysis lies in capturing nuance, context, and long-range dependencies within text.

## 3. METHODOLOGY
### 3.1 Dataset
The dataset utilized consists of 10,000 Amazon product reviews. The data was filtered to remove neutral (3-star) reviews, framing the problem as a binary classification task:
- Positive (Label 1): 4-5 star reviews
- Negative (Label 0): 1-2 star reviews

### 3.2 Preprocessing Pipeline
Text data is inherently noisy. Our `TextPreprocessor` pipeline executed the following:
- Conversion to lowercase.
- Removal of HTML tags and URLs.
- Tokenization using NLTK.
- Stopword removal (retaining critical sentiment modifiers like 'not').
- Lemmatization via `WordNetLemmatizer`.

### 3.3 Feature Extraction
Two distinct feature extraction methods were implemented:
1. **TF-IDF Vectorization**: Used for traditional ML models, transforming text into a sparse matrix (max 5,000 features).
2. **Token Sequences**: Used for the LSTM, mapping words to integer sequences, padded to a uniform length of 100.

### 3.4 Models
1. **Naive Bayes**: A probabilistic baseline model.
2. **Logistic Regression**: A linear approach that serves as a strong traditional ML benchmark.
3. **LSTM**: A deep learning model with an Embedding layer, two Bidirectional LSTM layers, Dropout layers (30%), and a dense Sigmoid output.

## 4. EXPERIMENTS & ABLATION STUDY

We conducted an ablation study to analyze the trade-offs between traditional TF-IDF-based Machine Learning and Sequence-based Deep Learning. The models were evaluated on an unseen 15% Test set.

### 4.1 Trade-off Analysis
1. **Sequential Processing**: Traditional ML models treat words as a "bag of words," losing sequential meaning. The LSTM naturally captures word order.
2. **Semantic Understanding**: The LSTM's embedding layer learns semantic proximity, whereas TF-IDF relies strictly on term frequencies.
3. **Computational Cost**:
   - Naive Bayes / Logistic Regression train in under 5 seconds.
   - LSTM requires minutes to hours depending on hardware, representing a trade-off of training cost for predictive power.

## 5. CONCLUSIONS
The results highlight that deep learning methods, specifically Recurrent Neural Networks like LSTM, are superior for opinion mining tasks where context and word order drastically change sentiment. While Logistic Regression is an excellent, lightweight alternative for production systems requiring rapid inference, the LSTM is recommended when maximizing Accuracy and F1-score is paramount.

## 6. FUTURE WORK
Future iterations of this project can explore:
- Transformer-based architectures (e.g., BERT, RoBERTa) for potentially state-of-the-art performance.
- Multi-class sentiment classification (predicting exact 1-5 star ratings).
- Aspect-based sentiment analysis to determine what specific features of a product are driving the sentiment.
