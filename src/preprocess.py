"""Text cleaning utilities for tweet sentiment analysis."""

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Normalize a tweet: lowercase, strip URLs/mentions, keep hashtag words."""
    if not text:
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)  # keep the word, drop the '#'
    text = NON_ALPHA_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def tokenize(text: str | None) -> list[str]:
    """Clean then whitespace-tokenize."""
    return clean_text(text).split()
