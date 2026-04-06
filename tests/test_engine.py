"""Tests for the ContextEngine."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_engine.engine import ContextEngine, ContextItem
from context_engine.token_counter import count_tokens_approximate


class TestTokenCounter:
    def test_empty_string(self):
        assert count_tokens_approximate("") == 0

    def test_single_word(self):
        assert count_tokens_approximate("hello") >= 1

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        count = count_tokens_approximate(text)
        assert count >= 9  # at least one token per word


class TestEviction:
    def _make_engine(self, max_tokens=20):
        return ContextEngine(max_tokens=max_tokens, model="gpt-4")

    def test_evicts_lowest_priority(self):
        engine = self._make_engine(max_tokens=10)
        # Add two items with different priorities
        engine.items = [
            ContextItem(role="user", content="hi", priority=0.5, token_count=5),
            ContextItem(role="user", content="bye", priority=1.0, token_count=5),
        ]
        evicted = engine._evict_lowest_priority()
        assert evicted is not None
        assert evicted.content == "hi"
        assert len(engine.items) == 1
        assert engine.items[0].content == "bye"

    def test_fifo_on_priority_tie(self):
        """When priorities are equal, the oldest (first) item is evicted."""
        engine = self._make_engine()
        engine.items = [
            ContextItem(role="user", content="first", priority=1.0, token_count=5),
            ContextItem(role="user", content="second", priority=1.0, token_count=5),
            ContextItem(role="user", content="third", priority=1.0, token_count=5),
        ]
        evicted = engine._evict_lowest_priority()
        assert evicted.content == "first"
        assert engine.items[0].content == "second"
        assert engine.items[1].content == "third"

    def test_evict_from_empty(self):
        engine = self._make_engine()
        assert engine._evict_lowest_priority() is None

    def test_add_evicts_when_over_budget(self):
        """Adding items beyond the budget triggers eviction of the oldest low-priority item."""
        engine = self._make_engine(max_tokens=13)
        engine.items = [
            ContextItem(role="user", content="old low pri", priority=0.1, token_count=8),
            ContextItem(role="user", content="keep me", priority=1.0, token_count=5),
        ]
        # total=13; adding "reply" (1 token) pushes to 14 > 13, so eviction must happen
        engine.add(role="assistant", content="reply", priority=1.0)
        contents = [i.content for i in engine.items]
        assert "old low pri" not in contents
        assert "keep me" in contents


if __name__ == "__main__":
    import unittest

    # Simple runner without pytest dependency
    import traceback
    results = {"passed": 0, "failed": 0}

    for cls in [TestTokenCounter, TestEviction]:
        obj = cls()
        for name in dir(cls):
            if name.startswith("test_"):
                try:
                    getattr(obj, name)()
                    print(f"  PASS  {cls.__name__}.{name}")
                    results["passed"] += 1
                except Exception:
                    print(f"  FAIL  {cls.__name__}.{name}")
                    traceback.print_exc()
                    results["failed"] += 1

    print(f"\n{results['passed']} passed, {results['failed']} failed")
    sys.exit(0 if results["failed"] == 0 else 1)
