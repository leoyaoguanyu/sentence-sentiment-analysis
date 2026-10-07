# Sentence Sentiment Analysis

A from-scratch NLP pipeline that classifies sentence sentiment (positive / negative):
tweet preprocessing → TF-IDF features → model comparison → persisted best model + CLI.

## Results

Dataset: NLTK `twitter_samples` — **10,000 labeled tweets** (5,000 positive / 5,000 negative),
stratified 80/20 split, TF-IDF with word + bigram features (5,000 max features).

| Model              | Accuracy | Precision | Recall | F1     |
|--------------------|----------|-----------|--------|--------|
| MultinomialNB      | 0.7800   | 0.8084    | 0.7340 | 0.7694 |
| **LogisticRegression** | **0.7905** | **0.8161** | **0.7500** | **0.7817** |
| LinearSVC          | 0.7695   | 0.7770    | 0.7560 | 0.7663 |

Best model (LogisticRegression, F1 0.78) is saved to `models/sentiment_model.joblib`.

## Quick start

```bash
pip install -r requirements.txt

# Train (downloads the NLTK dataset on first run) and save the best model
python -m src.train

# Classify a sentence
python -m src.predict "I absolutely love this movie!"
# -> positive (confidence 0.58)  [LogisticRegression]

# Run the test suite
python -m pytest -q
```

Explore the data in [`notebooks/eda.ipynb`](notebooks/eda.ipynb) (class balance,
tweet-length distribution, top words per class).

## How it's built

- **Preprocessing** (`src/preprocess.py`): lowercasing, URL/mention stripping,
  hashtag normalization, punctuation removal — covered by `pytest` unit tests.
- **Features** (`src/features.py`): TF-IDF with unigrams + bigrams, sublinear TF scaling.
- **Training** (`src/train.py`): stratified split, three-way model comparison on
  accuracy / precision / recall / F1, best-by-F1 model persisted with `joblib`.
- **CI** (`.github/workflows/ci.yml`): installs deps, runs `pytest`, then runs the
  full training pipeline as an end-to-end smoke test on every push/PR.

## Project layout

```
├── src/
│   ├── preprocess.py      # tweet cleaning + tokenization
│   ├── features.py        # TF-IDF vectorizer
│   ├── train.py           # dataset loading, training, evaluation, model saving
│   └── predict.py         # CLI for classifying new sentences
├── tests/                 # pytest: preprocessing + feature unit tests
├── notebooks/eda.ipynb    # exploratory data analysis (matplotlib)
├── models/                # trained artifacts (generated, git-ignored)
└── .github/workflows/ci.yml
```

## Tech stack

Python · Pandas · Scikit-learn · NLTK · Matplotlib · pytest · joblib · GitHub Actions
