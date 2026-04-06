"""Tests for ContextWindow priority-based eviction."""

import pytest
from context_engine.context_manager import ContextWindow


def _big(n: int) -> str:
    """Return a string whose approximate token count equals n."""
    # count_tokens_approx: tokens ~ number of 4-char words
    return ("word " * n).strip()


class TestPinnedEviction:
    def test_unpinned_evicted_before_pinned(self):
        """Pinned messages survive eviction; unpinned ones are removed first."""
        window = ContextWindow(max_tokens=50)
        window.add_message("user", _big(10), pinned=True)   # pinned, kept
        window.add_message("user", _big(10))                 # unpinned, evictable
        window.add_message("user", _big(10))                 # unpinned, evictable
        # Add a message that pushes over the budget
        window.add_message("user", _big(25))

        # The pinned message must still be present
        assert any(m.pinned for m in window.messages)
        assert window.total_tokens <= window.max_tokens

    def test_system_messages_never_evicted(self):
        """System messages are implicitly pinned and never evicted."""
        window = ContextWindow(max_tokens=30)
        window.add_message("system", _big(5))
        window.add_message("user", _big(5))
        window.add_message("assistant", _big(5))
        # Overflow: force eviction
        window.add_message("user", _big(20))

        system_msgs = [m for m in window.messages if m.role == "system"]
        assert len(system_msgs) == 1, "System message was incorrectly evicted"

    def test_oldest_unpinned_evicted_first(self):
        """Oldest unpinned message is evicted before newer ones."""
        window = ContextWindow(max_tokens=40)
        window.add_message("user", "first unpinned message here please")
        window.add_message("user", "second unpinned message here")
        # Force eviction by adding a large message
        window.add_message("user", _big(30))

        contents = [m.content for m in window.messages]
        assert "first unpinned message here please" not in contents, (
            "Oldest unpinned should have been evicted first"
        )

    def test_clear_keeps_pinned_by_default(self):
        """clear() retains pinned and system messages by default."""
        window = ContextWindow(max_tokens=200)
        window.add_message("system", "you are helpful")
        window.add_message("user", "hello", pinned=True)
        window.add_message("assistant", "hi there")

        window.clear()

        roles = [m.role for m in window.messages]
        assert "system" in roles
        assert any(m.pinned for m in window.messages)
        assert not any(m.role == "assistant" for m in window.messages)

    def test_clear_removes_pinned_when_flag_false(self):
        """clear(keep_pinned=False) removes pinned non-system messages."""
        window = ContextWindow(max_tokens=200)
        window.add_message("system", "you are helpful")
        window.add_message("user", "important", pinned=True)
        window.add_message("assistant", "ok")

        window.clear(keep_system=True, keep_pinned=False)

        assert len(window.messages) == 1
        assert window.messages[0].role == "system"

    def test_within_budget_no_eviction(self):
        """No messages are evicted when total tokens fit within the budget."""
        window = ContextWindow(max_tokens=200)
        window.add_message("user", "short")
        window.add_message("assistant", "also short")
        assert len(window.messages) == 2
        assert window.total_tokens <= 200
