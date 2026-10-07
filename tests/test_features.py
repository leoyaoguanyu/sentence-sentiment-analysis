from src.features import build_vectorizer


def test_vectorizer_output_shape():
    vec = build_vectorizer(max_features=100)
    matrix = vec.fit_transform(["i love nlp", "i hate bugs", "nlp is fun"])
    assert matrix.shape[0] == 3
    assert 0 < matrix.shape[1] <= 100
    assert len(vec.vocabulary_) > 0


def test_bigrams_included():
    vec = build_vectorizer(max_features=100)
    vec.fit(["i love nlp"])
    assert "love nlp" in vec.vocabulary_  # bigram captured


def test_unseen_words_produce_empty_row():
    vec = build_vectorizer(max_features=100)
    vec.fit(["hello world"])
    matrix = vec.transform(["completely unseen words here"])
    assert matrix.nnz == 0
