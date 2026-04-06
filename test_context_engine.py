"""Tests for context_engine.py"""

import pytest
from context_engine import (
    Message,
    ContextWindow,
    estimate_tokens,
    build_idf,
    _tokenize,
    _tfidf_vector,
    _cosine_similarity,
    _extractive_summary,
    score_message_importance,
    prune_context,
    build_context,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_nonempty():
    count = estimate_tokens("Hello world")
    assert count >= 1


def test_estimate_tokens_empty():
    # tiktoken returns 0 for empty string; the heuristic fallback clamps to 1.
    # Either way, the result must be non-negative.
    assert estimate_tokens("") >= 0


def test_estimate_tokens_longer_text_has_more_tokens():
    short = estimate_tokens("hi")
    long = estimate_tokens("This is a much longer sentence with many more words in it.")
    assert long > short


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_removes_stop_words():
    tokens = _tokenize("the cat sat on a mat")
    assert "the" not in tokens
    assert "on" not in tokens
    assert "a" not in tokens
    assert "cat" in tokens
    assert "sat" in tokens
    assert "mat" in tokens


def test_tokenize_lowercases():
    tokens = _tokenize("Python IS Great")
    assert "python" in tokens
    assert "great" in tokens


def test_tokenize_strips_punctuation():
    tokens = _tokenize("Hello, world!")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_empty_string():
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    idf = {"cat": 1.0, "sat": 1.0}
    tokens = ["cat", "sat"]
    v = _tfidf_vector(tokens, idf)
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    idf = {"cat": 1.0, "dog": 1.0}
    v_cat = _tfidf_vector(["cat"], idf)
    v_dog = _tfidf_vector(["dog"], idf)
    assert abs(_cosine_similarity(v_cat, v_dog)) < 1e-9


def test_cosine_similarity_empty():
    assert _cosine_similarity({}, {"cat": 1.0}) == 0.0
    assert _cosine_similarity({"cat": 1.0}, {}) == 0.0


def test_build_idf_gives_rare_terms_higher_weight():
    corpus = [
        ["cat", "sat"],
        ["cat", "dog"],
        ["cat", "bird"],
    ]
    idf = build_idf(corpus)
    # "cat" appears in all 3 docs → lower IDF than "sat" (1 doc)
    assert idf["cat"] < idf["sat"]


# ---------------------------------------------------------------------------
# score_message_importance
# ---------------------------------------------------------------------------

def test_score_bounds():
    msg = Message(role="user", content="hello", token_count=1)
    score = score_message_importance(msg, 0, 1)
    assert 0.0 <= score <= 1.0


def test_system_scores_higher_than_user_same_position():
    sys_msg = Message(role="system", content="You are helpful.", token_count=4)
    usr_msg = Message(role="user", content="You are helpful.", token_count=4)
    s_sys = score_message_importance(sys_msg, 1, 3)
    s_usr = score_message_importance(usr_msg, 1, 3)
    assert s_sys > s_usr


def test_recent_message_scores_higher():
    msg = Message(role="user", content="test", token_count=1)
    score_early = score_message_importance(msg, 0, 5)
    score_late = score_message_importance(msg, 4, 5)
    assert score_late > score_early


def test_relevant_message_scores_higher_with_query():
    idf_corpus = [
        _tokenize("black holes warp spacetime"),
        _tokenize("pasta cooking boiling water"),
        _tokenize("black holes hawking radiation"),
    ]
    idf = build_idf(idf_corpus)
    query_vector = _tfidf_vector(_tokenize("black holes hawking radiation"), idf)

    relevant = Message(role="assistant", content="Black holes warp spacetime.", token_count=5)
    irrelevant = Message(role="assistant", content="Pasta cooking boiling water.", token_count=5)

    s_rel = score_message_importance(relevant, 1, 3, query_vector, idf)
    s_irr = score_message_importance(irrelevant, 1, 3, query_vector, idf)
    assert s_rel > s_irr


# ---------------------------------------------------------------------------
# ContextWindow
# ---------------------------------------------------------------------------

def test_context_window_add_message():
    window = ContextWindow(max_tokens=20)
    msg = Message(role="user", content="hi", token_count=5)
    assert window.add_message(msg) is True
    assert window.total_tokens == 5
    assert window.remaining_tokens == 15


def test_context_window_rejects_oversized_message():
    window = ContextWindow(max_tokens=4)
    msg = Message(role="user", content="hi there", token_count=5)
    assert window.add_message(msg) is False
    assert len(window.messages) == 0


def test_context_window_to_dict():
    window = ContextWindow(max_tokens=100)
    window.add_message(Message(role="user", content="hello", token_count=2))
    d = window.to_dict()
    assert d["max_tokens"] == 100
    assert d["total_tokens"] == 2
    assert len(d["messages"]) == 1


# ---------------------------------------------------------------------------
# prune_context
# ---------------------------------------------------------------------------

def test_prune_context_no_op_when_within_budget():
    window = ContextWindow(max_tokens=100)
    for i in range(3):
        window.messages.append(Message(role="user", content=f"msg {i}", token_count=5))
    result = prune_context(window, 100)
    assert len(result.messages) == 3


def test_prune_context_drops_messages_to_fit():
    window = ContextWindow(max_tokens=100)
    for i in range(5):
        window.messages.append(Message(role="user", content=f"message number {i}", token_count=10))
    result = prune_context(window, 30)
    assert result.total_tokens <= 30


def test_prune_context_preserves_order():
    window = ContextWindow(max_tokens=100)
    for i, role in enumerate(["system", "user", "assistant", "user", "assistant"]):
        window.messages.append(Message(role=role, content=f"msg {i}", token_count=5))
    original = list(window.messages)
    result = prune_context(window, 20)
    # Every kept message must appear in the same relative order as in the original
    kept_positions = [original.index(m) for m in result.messages]
    assert kept_positions == sorted(kept_positions)


def test_prune_context_writes_importance_scores():
    window = ContextWindow(max_tokens=100)
    for i in range(4):
        window.messages.append(Message(role="user", content=f"sentence {i}", token_count=10))
    prune_context(window, 20)
    # All original messages should have scores written (prune_context mutates them)
    for msg in window.messages:
        assert 0.0 <= msg.importance_score <= 1.0


# ---------------------------------------------------------------------------
# _extractive_summary
# ---------------------------------------------------------------------------

def test_extractive_summary_basic():
    msgs = [
        Message(role="user", content="Quantum physics studies subatomic particles.", token_count=7),
        Message(role="assistant", content="Black holes warp spacetime near the event horizon.", token_count=9),
    ]
    summary = _extractive_summary(msgs, max_summary_tokens=30)
    assert summary is not None
    assert summary.role == "system"
    assert summary.content.startswith("[Prior context]")
    assert summary.token_count <= 30


def test_extractive_summary_empty():
    assert _extractive_summary([], max_summary_tokens=50) is None


def test_extractive_summary_respects_budget():
    msgs = [
        Message(role="user", content=" ".join(["word"] * 100), token_count=100),
    ]
    budget = 20
    summary = _extractive_summary(msgs, max_summary_tokens=budget)
    if summary:
        assert summary.token_count <= budget


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

def test_build_context_all_fit():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    window = build_context(messages, max_tokens=4096, reserve_tokens=500)
    assert len(window.messages) == 2


def test_build_context_prunes_when_over_budget():
    messages = [{"role": "user", "content": f"Message number {i} with some words."} for i in range(20)]
    window = build_context(messages, max_tokens=60, reserve_tokens=10)
    assert window.total_tokens <= 50


def test_build_context_summarize_flag():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about quantum physics and black holes."},
        {"role": "assistant", "content": "Quantum physics studies subatomic particles. Black holes warp spacetime."},
        {"role": "user", "content": "What about cooking pasta?"},
        {"role": "assistant", "content": "Boil water, add salt, cook pasta 8-10 minutes."},
        {"role": "user", "content": "Back to black holes — what is Hawking radiation?"},
    ]
    window = build_context(messages, max_tokens=60, reserve_tokens=10, summarize=True)
    has_summary = any(m.content.startswith("[Prior context]") for m in window.messages)
    assert has_summary
    assert window.total_tokens <= 50


def test_build_context_summarize_summary_after_system():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about quantum physics and black holes."},
        {"role": "assistant", "content": "Quantum physics studies subatomic particles. Black holes warp spacetime."},
        {"role": "user", "content": "What about cooking pasta?"},
        {"role": "assistant", "content": "Boil water, add salt, cook pasta 8-10 minutes."},
        {"role": "user", "content": "Back to black holes — what is Hawking radiation?"},
    ]
    window = build_context(messages, max_tokens=60, reserve_tokens=10, summarize=True)
    contents = [m.content for m in window.messages]
    summary_indices = [i for i, c in enumerate(contents) if c.startswith("[Prior context]")]
    system_indices = [i for i, m in enumerate(window.messages) if m.role == "system" and not m.content.startswith("[Prior context]")]
    if summary_indices and system_indices:
        # Summary should appear after the original system message
        assert summary_indices[0] > system_indices[0]
