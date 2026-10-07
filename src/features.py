"""TF-IDF feature engineering for the sentiment classifier."""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Word + bigram TF-IDF with sublinear term-frequency scaling.

    Input text is expected to be pre-cleaned (see preprocess.clean_text),
    so the vectorizer just splits on whitespace.
    """
    return TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
