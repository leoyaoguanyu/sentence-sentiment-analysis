from src.preprocess import clean_text, tokenize


def test_lowercases():
    assert clean_text("Hello WORLD") == "hello world"


def test_removes_urls():
    cleaned = clean_text("check https://example.com and www.test.org out")
    assert "http" not in cleaned and "www" not in cleaned


def test_removes_mentions():
    assert "@" not in clean_text("hi @user, how are you?")


def test_hashtag_keeps_word():
    assert clean_text("#nlp is fun") == "nlp is fun"


def test_strips_punctuation_and_extra_spaces():
    assert clean_text("Wow!!!   so   cool...") == "wow so cool"


def test_empty_and_none_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_tokenize_splits_on_whitespace():
    assert tokenize("Hello,   world!") == ["hello", "world"]
