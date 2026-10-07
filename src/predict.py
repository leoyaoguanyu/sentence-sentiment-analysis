"""CLI: classify a sentence with the trained model.

Usage:
    python -m src.predict "I absolutely love this movie!"
    python -m src.predict            # then type a sentence interactively
"""

import argparse
from pathlib import Path

import joblib

from .preprocess import clean_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Sentence to classify")
    ap.add_argument("--model", type=Path, default=Path("models/sentiment_model.joblib"))
    args = ap.parse_args()

    text = args.text if args.text else input("Enter a sentence: ")
    bundle = joblib.load(args.model)
    vectorizer, model = bundle["vectorizer"], bundle["model"]

    features = vectorizer.transform([clean_text(text)])
    pred = int(model.predict(features)[0])
    label = "positive" if pred == 1 else "negative"

    confidence = ""
    if hasattr(model, "predict_proba"):
        confidence = f" (confidence {float(model.predict_proba(features)[0].max()):.2f})"
    print(f"{label}{confidence}  [{bundle['name']}]")


if __name__ == "__main__":
    main()
