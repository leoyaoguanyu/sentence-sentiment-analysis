"""Train and compare sentiment classifiers; persist the best model.

Dataset: NLTK twitter_samples (10k labeled tweets, balanced positive/negative).
Splits stratified 80/20, compares Naive Bayes / Logistic Regression / Linear SVM
on accuracy, precision, recall and F1, then saves the best by F1.
"""

import argparse
import json
from pathlib import Path

import joblib
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .features import build_vectorizer
from .preprocess import clean_text


def load_data() -> tuple[list[str], list[int]]:
    from nltk.corpus import twitter_samples

    try:
        pos = twitter_samples.strings("positive_tweets.json")
        neg = twitter_samples.strings("negative_tweets.json")
    except LookupError:
        nltk.download("twitter_samples")  # first run needs internet
        pos = twitter_samples.strings("positive_tweets.json")
        neg = twitter_samples.strings("negative_tweets.json")
    texts = [clean_text(t) for t in pos + neg]
    labels = [1] * len(pos) + [0] * len(neg)
    pairs = [(t, y) for t, y in zip(texts, labels) if t]  # drop empties
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def evaluate(model, x_test, y_test) -> dict[str, float]:
    pred = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models"))
    args = ap.parse_args()

    texts, labels = load_data()
    print(f"Loaded {len(texts)} labeled tweets "
          f"({sum(labels)} positive / {len(labels) - sum(labels)} negative)")

    x_train_t, x_test_t, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size,
        random_state=args.seed, stratify=labels,
    )
    vectorizer = build_vectorizer(args.max_features)
    x_train = vectorizer.fit_transform(x_train_t)
    x_test = vectorizer.transform(x_test_t)
    print(f"TF-IDF matrix: {x_train.shape[0]} train docs x {x_train.shape[1]} features")

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = evaluate(model, x_test, y_test)
        m = results[name]
        print(f"{name:20s} acc={m['accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "model": models[best_name], "name": best_name},
        args.out / "sentiment_model.joblib",
    )
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Best model: {best_name} -> {args.out / 'sentiment_model.joblib'}")


if __name__ == "__main__":
    main()
