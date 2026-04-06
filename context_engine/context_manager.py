"""Context window management for LLM conversations."""

from dataclasses import dataclass, field
from typing import Literal

from .tokenizer import count_tokens_approx, truncate_to_token_limit


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str
    pinned: bool = False

    @property
    def token_count(self) -> int:
        # ~4 overhead tokens per message for role/formatting
        return count_tokens_approx(self.content) + 4


@dataclass
class ContextWindow:
    """Manages a sliding context window for LLM conversations.

    Tracks messages and their token counts, automatically evicting
    older messages when the token budget is exceeded.

    Supports pinning messages so they are never evicted during overflow.
    System messages are always treated as pinned. Additional messages
    (e.g. important user instructions) can be pinned via ``pinned=True``
    when calling :meth:`add_message`.
    """

    max_tokens: int
    messages: list[Message] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    @property
    def remaining_tokens(self) -> int:
        return self.max_tokens - self.total_tokens

    def add_message(self, role: Role, content: str, pinned: bool = False) -> None:
        """Add a message, evicting oldest unpinned messages if needed.

        Args:
            role: The role of the message sender.
            content: The message text.
            pinned: If True, this message will never be evicted to make room
                for new messages.  System messages are always pinned regardless
                of this flag.
        """
        msg = Message(role=role, content=content, pinned=pinned)
        self.messages.append(msg)
        self._evict_if_needed()

    def _is_evictable(self, msg: Message) -> bool:
        """Return True if a message may be removed during overflow eviction."""
        return msg.role != "system" and not msg.pinned

    def _evict_if_needed(self) -> None:
        """Remove oldest evictable messages until we're within the token budget.

        Eviction order:
        1. Oldest messages that are neither system messages nor pinned.
        2. If no evictable messages remain, truncate the last pinned/system
           message as a last resort so the window never exceeds its budget.
        """
        while self.total_tokens > self.max_tokens:
            # Find the oldest evictable message
            evicted = False
            for i, msg in enumerate(self.messages):
                if self._is_evictable(msg):
                    self.messages.pop(i)
                    evicted = True
                    break

            if not evicted:
                # All remaining messages are pinned/system — truncate the last
                # one to make the window fit as a safety valve.
                if self.messages:
                    last = self.messages[-1]
                    budget = self.max_tokens - sum(
                        m.token_count for m in self.messages[:-1]
                    )
                    last.content = truncate_to_token_limit(last.content, max(0, budget - 4))
                break

    def to_api_messages(self) -> list[dict]:
        """Serialize messages to the format expected by the Claude API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self, keep_system: bool = True, keep_pinned: bool = True) -> None:
        """Clear the context window.

        Args:
            keep_system: If True, retain system messages.
            keep_pinned: If True, retain pinned messages.
        """
        self.messages = [
            m for m in self.messages
            if (keep_system and m.role == "system") or (keep_pinned and m.pinned)
        ]
