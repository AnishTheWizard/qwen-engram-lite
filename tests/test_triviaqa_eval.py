from naive_engram.eval_triviaqa import (
    answer_aliases,
    contains_alias,
    exact_match,
    normalize_trivia_answer,
    token_f1,
    truncate_generation,
)


def test_official_normalization() -> None:
    assert normalize_trivia_answer("The U.S._President’s  Answer!") == "u s president s answer"


def test_alias_aware_metrics() -> None:
    aliases = ["David Seville", "Ross Bagdasarian Sr."]
    assert exact_match("  David Seville! ", aliases) == 1.0
    assert exact_match("It was David Seville", aliases) == 0.0
    assert contains_alias("It was David Seville", aliases) == 1.0
    assert token_f1("David Seville", aliases) == 1.0
    assert 0.0 < token_f1("Seville", aliases) < 1.0


def test_answer_schema_and_generation_truncation() -> None:
    row = {"answer": {"value": "David Seville", "aliases": ["David Seville", "Alvin's creator"]}}
    assert answer_aliases(row) == ["David Seville", "Alvin's creator"]
    assert truncate_generation(" David Seville. He created them.", ("\n", ".", ",")) == "David Seville"
